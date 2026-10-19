"""Profile config isolation across registration, turns and unscoped workers (#1603).

The host stub models secret_scope's context-local accessors and home-only plugin
discovery. Its get_secret deliberately raises: the plugin must not call it.
"""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def host(monkeypatch):
    home_var = ContextVar("home", default=None)
    scope_var = ContextVar("scope", default=None)
    bound_var = ContextVar("bound_home", default=None)
    host = SimpleNamespace(routed=True, builds=[])
    scope = ModuleType("agent.secret_scope")
    scope.current_secret_scope = scope_var.get
    scope.current_secret_scope_home = bound_var.get
    scope.serves_routed_profile = lambda: host.routed
    scope.is_multiplex_active = lambda: host.routed

    def build(home):
        host.builds.append(home)
        path = home / ".env"
        return dict(line.split("=", 1) for line in path.read_text().splitlines()) if path.exists() else {}

    def raising_get_secret(*args):
        raise RuntimeError("UnscopedSecretError: must never reach this accessor")

    scope.build_profile_secret_scope = build
    scope.get_secret = raising_get_secret
    monkeypatch.setitem(sys.modules, "agent.secret_scope", scope)
    import agent
    monkeypatch.setattr(agent, "secret_scope", scope, raising=False)
    cli = ModuleType("hermes_cli")
    cli.__path__ = []
    config = ModuleType("hermes_cli.config")
    config.get_hermes_home = lambda: home_var.get() or Path(os.environ["HERMES_HOME"])
    config.load_config = lambda: {"context": {"engine": "lcm-x"}}
    cli.config = config
    monkeypatch.setitem(sys.modules, "hermes_cli", cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config)

    @contextmanager
    def bind(home, mapping=None, *, bound_home=None):
        tokens = (home_var.set(home), scope_var.set(mapping),
                  bound_var.set(str(bound_home or home) if mapping is not None else None))
        try:
            yield
        finally:
            home_var.reset(tokens[0])
            scope_var.reset(tokens[1])
            bound_var.reset(tokens[2])

    host.bind = bind
    return host


@pytest.fixture
def load_profile():
    names = []
    engines = []

    def load():
        name = f"issue1603_profile_{len(names)}"
        names.append(name)
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py",
                                                    submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        ctx = SimpleNamespace(register_context_engine=engines.append)
        module.register(ctx)
        return module, engines[-1]

    yield load
    for engine in engines:
        engine.shutdown()
    for key in list(sys.modules):
        if any(key == name or key.startswith(name + ".") for name in names):
            sys.modules.pop(key, None)


@pytest.mark.parametrize("discovery_scope", [True, False])
def test_two_homes_keep_registration_values(host, load_profile, tmp_path, monkeypatch, discovery_scope):
    """Different profiles stay isolated through model switches and worker reads."""
    monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "120000")
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture-launch")
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    profiles = []
    for index, absolute, ratio, embeddings in [(0, 250000, 0.61, True), (1, 160000, 0.82, False)]:
        home = tmp_path / f"profile-{index}"
        home.mkdir()
        mapping = {"LCM_ABSOLUTE_THRESHOLD_TOKENS": str(absolute),
                   "LCM_CONTEXT_THRESHOLD": str(ratio),
                   "LCM_EMBEDDINGS_ENABLED": str(embeddings).lower(),
                   "VOYAGE_API_KEY": f"fixture-routed-{index}",
                   "LCM_EMBEDDING_PROVIDER": "voyage", "LCM_EMBEDDING_MODEL": "voyage-4"}
        (home / ".env").write_text("\n".join(f"{key}={value}" for key, value in mapping.items()))
        with host.bind(home, mapping if discovery_scope else None):
            module, engine = load_profile()
        profiles.append((home, module, engine, mapping, absolute, ratio, embeddings))

    monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "99000")
    for home, module, engine, mapping, absolute, ratio, embeddings in profiles:
        assert engine._hermes_home == str(home)
        assert engine._config.context_threshold == ratio
        assert engine._config.embeddings_enabled is embeddings
        # A routed miss must retain the default, not the launch profile's true.
        assert engine._config.preanswer_evidence_enabled is False
        mapping["LCM_ABSOLUTE_THRESHOLD_TOKENS"] = "1"  # registration copied it
        config = sys.modules[module.__name__ + ".config"]
        provider_module = sys.modules[module.__name__ + ".embedding_provider"]

        def later():
            assert config._hermes_config_path() == home / "config.yaml"
            assert config.lcm_env_get("VOYAGE_API_KEY") == f"fixture-routed-{home.name[-1]}"
            assert config.lcm_env_get("LCM_PREANSWER_EVIDENCE_ENABLED", "missing") == "missing"
            for window in (262144, 500000, 1000000):
                engine._set_context_length(window, source="issue1603", model="fixture", provider="custom")
                assert engine.threshold_tokens == absolute
                assert engine.context_threshold == ratio
            provider = provider_module.resolve_provider(engine._config)
            headers_seen = []

            def transport(*args, **kwargs):
                headers_seen.append(kwargs["headers"]["Authorization"])
                return provider_module.HttpResponse(200, {}, json.dumps({"data": [{"embedding": [0.1, 0.2]}]}).encode())

            provider._transport = transport
            provider.embed_query("fixture query")
            assert headers_seen == [f"Bearer fixture-routed-{home.name[-1]}"]

        # Executor threads start with no ContextVars; test both it and a later
        # turn mistakenly carrying the sibling scope. The module is home-owned.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(later).result()
        with host.bind(tmp_path / "sibling", {"VOYAGE_API_KEY": "fixture-sibling"}):
            later()


def test_yaml_and_unscoped_misses(host, load_profile, tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture-launch")
    monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", "120000")
    home = tmp_path / "yaml-home"
    home.mkdir()
    (home / "config.yaml").write_text("lcm:\n  context_threshold: 0.67\n")
    with host.bind(home):
        module, engine = load_profile()
    assert engine._config.context_threshold == 0.67
    config = sys.modules[module.__name__ + ".config"]
    assert config.lcm_env_get("VOYAGE_API_KEY", "missing") == "missing"
    engine._set_context_length(200000, source="issue1603", model="fixture", provider="custom")
    assert engine.threshold_tokens == 134000


def test_home_only_discovery_rejects_foreign_bound_mapping(host, load_profile, tmp_path):
    home = tmp_path / "wanted"
    home.mkdir()
    (home / ".env").write_text("LCM_CONTEXT_THRESHOLD=0.69\n")
    with host.bind(home, {"LCM_CONTEXT_THRESHOLD": "0.12"}, bound_home=tmp_path / "foreign"):
        _, engine = load_profile()
    assert engine._config.context_threshold == 0.69


def test_scope_failure_uses_defaults(host, load_profile, tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.35")
    monkeypatch.setenv("VOYAGE_API_KEY", "fixture-launch")

    def fail(*args):
        raise RuntimeError("host scope unavailable")

    monkeypatch.setattr(sys.modules["agent.secret_scope"], "build_profile_secret_scope", fail)
    with host.bind(tmp_path):
        module, engine = load_profile()
    assert engine._config.context_threshold == 0.75
    assert sys.modules[module.__name__ + ".config"].lcm_env_get("VOYAGE_API_KEY", "missing") == "missing"


@pytest.mark.parametrize("has_scope_api", [True, False])
def test_no_scope_snapshot_matches_main(host, load_profile, monkeypatch, tmp_path, has_scope_api):
    """All dataclass fields equal the main d411e404 fixture, not selected values."""
    fixture = json.loads((ROOT / "tests/fixtures/issue_1603_single_profile.json").read_text())
    for key, value in fixture["env"].items():
        monkeypatch.setenv(key, value)
    # Config-only: the artificial containment base is deliberately not a real
    # home. Loading an engine here would conflate config parsing with storage.
    host.routed = False
    if not has_scope_api:
        monkeypatch.setitem(sys.modules, "agent.secret_scope", None)
        import agent
        monkeypatch.delattr(agent, "secret_scope")
    name = "issue1603_snapshot_config"
    spec = importlib.util.spec_from_file_location(name, ROOT / "config.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    with host.bind(tmp_path):
        assert asdict(module.LCMConfig.from_env()) == fixture["expected"]


def test_environment_read_inventory():
    """AST inventory, including aliased getenv/environ access in plugin modules."""
    allowed = {
        ("config.py", "lcm_env"): "sole resolver: explicit single-profile/old-host fallback",
        ("config.py", "from_env"): "identity check to distinguish routed snapshot from process mapping",
        ("embedding_provider.py", "__init__"): "FASTEMBED_CACHE_PATH: process-wide model download cache",
        ("inactive_record.py", "_process_start"): "LC_ALL=C subprocess for ps process identity, not profile configuration",
    }
    seen = set()
    failures = []
    excluded = {"tests", "bench", "benchmarking", "benchmarks", "scripts"}
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if any(part in excluded or part.startswith(".") for part in relative.parts) or relative.parts[0] == "agent":
            continue
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        os_names = {"os"}
        env_names, getenv_names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                os_names.update(alias.asname or "os" for alias in node.names if alias.name == "os")
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                env_names.update(alias.asname or alias.name for alias in node.names if alias.name == "environ")
                getenv_names.update(alias.asname or alias.name for alias in node.names if alias.name == "getenv")
        for node in ast.walk(tree):
            is_read = (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                       and node.value.id in os_names and node.attr in {"environ", "getenv"})
            is_read |= isinstance(node, ast.Name) and node.id in env_names | getenv_names
            if not is_read:
                continue
            parent = parents.get(node)
            while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = parents.get(parent)
            key = (relative.as_posix(), parent.name if parent else "<module>")
            seen.add(key)
            if key not in allowed:
                failures.append(f"{relative}:{node.lineno}: unclassified environment read")
            if key == ("config.py", "lcm_env"):
                assert isinstance(parents[node], ast.Return), "only the legacy mapping fallback is allowed"
            if key == ("config.py", "from_env"):
                assert isinstance(parents[node], ast.Compare) and isinstance(parents[node].ops[0], ast.IsNot)
            if key == ("embedding_provider.py", "__init__"):
                call = parents.get(parents.get(node))
                assert isinstance(call, ast.Call) and ast.literal_eval(call.args[0]) == "FASTEMBED_CACHE_PATH"
    assert not failures, "\n".join(failures)
    assert seen == set(allowed), "remove stale allowlist entries or check newly unclassified reads"
