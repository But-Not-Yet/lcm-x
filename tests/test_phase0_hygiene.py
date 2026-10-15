"""Acceptance checks for import reporting and per-test environment isolation."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_lcm.config import ENV_FIELD_SPECS
from tests.phase0_hygiene import product_env_keys

ROOT = Path(__file__).resolve().parents[1]
HARNESS_KEYS = (
    "LCM_REAL_HERMES_SRC", "LCM_REAL_HERMES_PYTHON", "LCM_REAL_HERMES_LONG",
    "LCM_DIFFERENTIAL", "LCM_DIFFERENTIAL_ENABLED",
)


def test_product_env_derivation_covers_config_and_root_reads():
    keys = product_env_keys(ROOT)
    assert {spec.env_key for spec in ENV_FIELD_SPECS} <= keys
    assert {
        "LCM_LEAF_CHUNK_TOKENS", "LCM_HOST_MESSAGE_UID", "LCM_MODEL_THRESHOLDS",
        "LCM_DISABLED_TOOLS", "LCM_HERMES_BASE_DIR", "LCM_IDENTITY_ANCHOR",
        "LCM_ABSOLUTE_THRESHOLD_TOKENS", "LCM_EMBEDDING_BACKFILL_BATCH_SIZE",
        "LCM_SENSITIVE_PATTERNS", "LCM_SUMMARY_FALLBACK_MODELS",
    } <= keys
    assert not set(HARNESS_KEYS) & keys


def _run_pytest(tmp_path, source, *, injection="", env=None):
    probe = tmp_path / "test_probe.py"
    probe.write_text(source, encoding="utf-8")
    code = (
        "import sys, pytest\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        + injection
        + "\nraise SystemExit(pytest.main(['-p', 'tests.conftest', '-p', 'no:cacheprovider', "
        + f"'--rootdir', {str(tmp_path)!r}, '--confcutdir', {str(tmp_path)!r}, '-v', {str(probe)!r}]))\n"
    )
    return subprocess.run(
        ["nice", "-n", "19", sys.executable, "-c", code],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
    )


@pytest.mark.parametrize("module,error,allowed", [
    ("engine", "ModuleNotFoundError('missing host', name='agent')", True),
    ("engine", "ModuleNotFoundError('missing host', name='agent.context_engine')", True),
    ("engine", "ModuleNotFoundError('missing product dependency', name='broken_dependency')", False),
    ("engine", "ValueError('broken engine')", False),
    ("presets", "RuntimeError('broken presets')", False),
])
def test_bootstrap_reports_failures_and_rejects_unexpected_imports(tmp_path, module, error, allowed):
    injection = f"""
import importlib.machinery
original = importlib.machinery.SourceFileLoader.exec_module
def injected(loader, mod):
    if mod.__name__ == 'hermes_lcm.{module}':
        raise {error}
    return original(loader, mod)
importlib.machinery.SourceFileLoader.exec_module = injected
"""
    result = _run_pytest(tmp_path, "def test_probe(): pass\n", injection=injection)
    output = result.stdout + result.stderr
    assert result.returncode == (0 if allowed else pytest.ExitCode.USAGE_ERROR), output
    assert f"LCM import failure: hermes_lcm.{module}:" in output
    assert error.split("(")[0] in output
    if allowed:
        assert "missing host" in output
        assert "1 passed" in output
    else:
        assert "Unexpected plugin import failures" in output
        assert "1 passed" not in output


def test_ambient_product_env_is_removed_but_harness_controls_and_test_overrides_survive(tmp_path):
    keys = sorted(product_env_keys(ROOT))
    env = dict(os.environ)
    env.update(dict.fromkeys(keys, "ambient-product-sentinel"))
    env.update(dict.fromkeys(HARNESS_KEYS, "harness-sentinel"))
    ambient_home = tmp_path / "ambient-home"
    env["HERMES_HOME"] = str(ambient_home)
    result = _run_pytest(tmp_path, f"""
import os
from pathlib import Path

def test_first(monkeypatch, tmp_path):
    assert not set({keys!r}) & os.environ.keys()
    assert all(os.environ[k] == 'harness-sentinel' for k in {HARNESS_KEYS!r})
    assert Path(os.environ['HERMES_HOME']) == tmp_path / 'hermes-home'
    monkeypatch.setenv('LCM_LEAF_CHUNK_TOKENS', '1234')
    from hermes_lcm.config import LCMConfig
    assert LCMConfig.from_env().leaf_chunk_tokens == 1234

def test_second(tmp_path):
    assert not set({keys!r}) & os.environ.keys()
    assert Path(os.environ['HERMES_HOME']) == tmp_path / 'hermes-home'
""", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert not ambient_home.exists()
