"""Characterize fallback values and the existing configuration warning channel."""

import json
import logging

import pytest

from hermes_lcm import config, tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    for key in list(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


VALUES = [
    ("LCM_SURVIVAL_FIT", "survival_fit", raw, value)
    for raw, value in [(None, True), ("", True), (" \t", True), ("flase", True)]
    + [(raw, True) for raw in ("1", "true", "yes", "on", " TRUE ")]
    + [(raw, False) for raw in ("0", "false", "no", "off", " OFF ")]
] + [
    ("LCM_MAX_ASSEMBLY_TOKENS", "max_assembly_tokens", raw, value)
    for raw, value in [(None, 0), ("", 0), (" \t", 0), ("abc", 0), ("24k", 0),
                       ("1.5", 0), (" 24 ", 24), ("-1", -1), ("1_000", 1000)]
] + [
    ("LCM_SURVIVAL_RESERVE", "survival_reserve", raw, value)
    for raw, value in [(None, 0.15), ("", 0.15), (" \t", 0.15), ("abc", 0.15),
                       ("24k", 0.15), (" 0.25 ", 0.25), ("1e-2", 0.01), ("7.5", 7.5)]
]


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("key,field,raw,value", VALUES)
def test_fallback_values_unchanged(monkeypatch, key, field, raw, value):
    if raw is not None:
        monkeypatch.setenv(key, raw)
    assert getattr(config.LCMConfig.from_env(), field) == value


@pytest.mark.usefixtures("isolated_env")
def test_valid_env_warning_consumers(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("LCM_SURVIVAL_FIT", "true")
    monkeypatch.setenv("LCM_MAX_ASSEMBLY_TOKENS", "100000")
    monkeypatch.setenv("LCM_SURVIVAL_RESERVE", "0.15")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_TEMPORAL_ROLLUPS_ENABLED", "false")
    c = config.LCMConfig.from_env()
    c.database_path = str(tmp_path / "lcm.db")
    with caplog.at_level(logging.WARNING):
        engine = LCMEngine(config=c, hermes_home=str(tmp_path))
        try:
            engine.on_session_start("warning-channel", context_length=200000)
            assert engine.get_status()["config_source_warnings"] == []
            assert json.loads(tools.lcm_status({}, engine=engine))["config_source_warnings"] == []
            assert "config_source_warnings:" not in handle_lcm_command("status", engine)
            doctor = json.loads(tools.lcm_doctor({}, engine=engine))
            check = next(row for row in doctor["checks"] if row["check"] == "config_validation")
            assert check == {"check": "config_validation", "status": "pass",
                             "detail": "all settings within normal ranges"}
        finally:
            engine.shutdown()
    assert c.config_source_warnings == []
    assert not any("invalid env" in record.message for record in caplog.records)
