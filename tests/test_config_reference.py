"""Pin the config metadata consumed by the configuration reference."""

import pytest

from hermes_lcm import config


def test_reference_scalar_specs_match_constructor_defaults():
    defaults = config.LCMConfig()
    assert len({spec.env_key for spec in config.ENV_FIELD_SPECS}) == len(config.ENV_FIELD_SPECS)
    for spec in config.ENV_FIELD_SPECS:
        value = getattr(defaults, spec.name)
        # Privacy's None means automatic provider-dependent protection.
        assert value is None or type(value) is spec.py_type


def test_reference_diagnostic_ranges():
    assert config._ENV_RANGES == {
        "context_threshold": (0.0, 1.0, False),
        "survival_reserve": (0.0, 0.9, True),
        "leaf_chunk_tokens": (1, None, True),
        "summary_timeout_ms": (0, None, False),
    }


def test_reference_diagnostic_choices():
    assert {name: row[0] for name, row in config._ENV_CHOICES.items()} == {
        "embedding_provider": (
            "voyage", "voyageai", "ollama", "fastembed", "fast-embed",
            "openai-compatible", "openai", "siliconflow",
        ),
        "preanswer_evidence_mode": ("off", "legacy_selective", "requirements_v1"),
        "embedding_storage_dtype": ("float32", "int8"),
    }


@pytest.mark.parametrize("key", (
    "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED", "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED",
))
def test_reference_deprecation_warning_bytes(monkeypatch, key):
    for name in tuple(config.os.environ):
        if name.startswith("LCM_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {})
    monkeypatch.setenv(key, "true")
    assert config.LCMConfig.from_env().config_source_warnings == [
        f"deprecated: {key} is set; this experiment is removed in v0.28.0 (#1246). "
        "Until then it behaves as before."
    ]
