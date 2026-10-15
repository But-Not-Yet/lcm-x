"""Characterize the loading sources of the two retiring experiments."""

import pytest

from hermes_lcm import config, presets


FLAGS = (
    ("cache_friendly_condensation_enabled", "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED"),
    ("large_output_transcript_gc_enabled", "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED"),
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    for key in list(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.mark.parametrize("field,key", FLAGS)
@pytest.mark.parametrize("raw,expected", [(None, False), ("false", False), ("true", True)])
def test_env_flag_values(monkeypatch, field, key, raw, expected):
    if raw is not None:
        monkeypatch.setenv(key, raw)
    assert getattr(config.LCMConfig.from_env(), field) is expected


@pytest.mark.parametrize("field,key", FLAGS)
def test_direct_construction_does_not_warn(field, key):
    c = config.LCMConfig(**{field: True})
    assert getattr(c, field) is True
    assert c.config_source_warnings == []


@pytest.mark.parametrize("field,key", FLAGS)
def test_yaml_flag_is_ignored(tmp_path, field, key):
    (tmp_path / "config.yaml").write_text(f"lcm:\n  {field}: true\n")
    c = config.LCMConfig.from_env()
    assert getattr(c, field) is False
    assert c.ignored_config_yaml_lcm_keys == [field]
    assert c.config_source_warnings == []


def test_presets_do_not_enable_retiring_flags():
    for preset in presets.shipped_presets():
        for field, key in FLAGS:
            assert field not in preset.runtime_env
            assert field not in presets._PRESET_FIELDS
