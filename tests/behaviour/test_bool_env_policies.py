"""Legacy boolean policies, including readers loaded outside a package."""

import importlib.util
from pathlib import Path

import pytest

from hermes_lcm import config, db_bootstrap, identity_anchor
from hermes_lcm.engine import LCMEngine
from bench.instruments.reliability import plugin_tree


ROOT = Path(__file__).resolve().parents[2]


def load_plugin(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_boolean_edges(monkeypatch):
    """Pin whitespace, non-spellings, AST extraction and standalone loading."""
    plugins = [load_plugin(name) for name in ("bool_env_plugin", "hermes_lcm.bool_env_plugin")]
    for raw in (" \t\n", "t", "f", "enabled", "disabled", "2", "  OFF\n", " Yes\t"):
        on = raw == " Yes\t"
        off = raw == "  OFF\n"
        for default in (False, True):
            monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", raw)
            assert config._parse_bool_env("LCM_ENABLE_SLASH_COMMAND", default) is (
                True if on else False if off else default
            )
            for plugin in plugins:
                assert plugin._env_flag_enabled("LCM_ENABLE_SLASH_COMMAND", default) is on
        for key, reader in (
            ("LCM_IDENTITY_ANCHOR", identity_anchor.identity_anchor_enabled),
            ("LCM_FTS_INTEGRITY_BACKGROUND", db_bootstrap._background_integrity_enabled),
        ):
            monkeypatch.setenv(key, raw)
            assert reader() is not off
        assert plugin_tree.parse_bool(ROOT, "LCM_NATIVE_RECOVERY", raw)[0] is on
        monkeypatch.setenv("LCM_DISABLED_TOOLS", raw)
        expected = {raw.strip()} if raw.strip() else set()
        for plugin in plugins:
            assert plugin._disabled_tool_names() == expected
        assert LCMEngine._disabled_tool_names() == expected

    # This standalone live-provider authorization guard is deliberately exact,
    # and rejects before loading any plugin modules or constructing a provider.
    spec = importlib.util.spec_from_file_location("bool_env_eval", ROOT / "scripts/eval_retrieval_recall.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *a, **kw: pytest.fail("loaded plugin"))

    class Allowed(Exception):
        pass

    def reached_config():
        raise Allowed

    monkeypatch.setattr(config.LCMConfig, "from_env", reached_config)
    for raw in ("", " ", "true", "TRUE", "yes", "on", " 1 ", "garbage", "0"):
        monkeypatch.setenv("LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER", raw)
        with pytest.raises(RuntimeError, match="requires LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER=1"):
            module._load_live_provider()
    monkeypatch.setenv("LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER", "1")
    with pytest.raises(Allowed):
        module._load_live_provider()
