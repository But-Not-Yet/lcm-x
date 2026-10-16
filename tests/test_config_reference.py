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
        f"deprecated: {key} is set; this experiment is removed in v0.29.0 (#1246). "
        "Until then it behaves as before."
    ]


@pytest.fixture
def reference():
    from scripts import gen_config_reference

    return gen_config_reference


def test_committed_reference_is_current(reference):
    document = reference.REFERENCE.read_text(encoding="utf-8")
    assert document == reference.update_reference(document), (
        "Configuration reference is stale; run python scripts/gen_config_reference.py"
    )


def test_generator_uses_metadata_and_constructor_defaults(monkeypatch, reference):
    from types import SimpleNamespace

    specs = (
        SimpleNamespace(name="limit", env_key="LCM_LIMIT", py_type=float),
        SimpleNamespace(name="mode", env_key="LCM_MODE", py_type=str),
        SimpleNamespace(name="privacy", env_key="LCM_PRIVACY", py_type=bool),
        SimpleNamespace(name="enabled", env_key="LCM_ENABLED", py_type=bool),
    )
    monkeypatch.setattr(reference, "ENV_FIELD_SPECS", specs)
    monkeypatch.setattr(reference, "LCMConfig", lambda: SimpleNamespace(
        limit=0.5, mode="", privacy=None, enabled=False,
    ))
    monkeypatch.setattr(reference, "_ENV_RANGES", {"limit": (0, 1, False)})
    monkeypatch.setattr(reference, "_ENV_CHOICES", {"mode": (("a", "b"), str.lower, "fallback")})
    monkeypatch.setattr(reference, "ENV_FIELD_DEPRECATIONS", {"enabled": ("v1", "v2")})
    assert reference.render_reference().splitlines()[2:] == [
        "| `LCM_LIMIT` | `float` | `0.5` | `> 0`, `<= 1` | — |",
        '| `LCM_MODE` | `str` | `""` | `"a"`, `"b"` | — |',
        "| `LCM_PRIVACY` | `bool` | `null` | — | — |",
        "| `LCM_ENABLED` | `bool` | `false` | — | Deprecated v1; removed v2 |",
    ]


def test_generator_ignores_environment_and_host_config(monkeypatch, reference):
    import sys

    expected = reference.render_reference()
    module = sys.modules[reference.LCMConfig.__module__]

    def forbidden(*args, **kwargs):
        pytest.fail("reference generation must not read host configuration")

    monkeypatch.setattr(module.LCMConfig, "from_env", forbidden)
    monkeypatch.setattr(module, "_load_hermes_config_yaml", forbidden)
    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "999")
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "synthetic-placeholder")
    assert reference.render_reference() == expected
    assert "synthetic-placeholder" not in expected


def test_generator_replaces_only_marked_bytes_and_is_idempotent(reference, monkeypatch, tmp_path):
    document = "# Before\n\n" + reference.START + "\nold\n" + reference.END + "\n\n# After\n"
    path = tmp_path / "configuration.md"
    path.write_text(document, encoding="utf-8")
    monkeypatch.setattr(reference, "REFERENCE", path)
    reference.main()
    generated = path.read_text(encoding="utf-8")
    assert generated == (
        "# Before\n\n" + reference.START + "\n" + reference.render_reference()
        + reference.END + "\n\n# After\n"
    )
    reference.main()
    assert path.read_text(encoding="utf-8") == generated


@pytest.mark.parametrize("document", (
    "no markers", "{start}", "{end}", "{end}\n{start}",
    "{start}\n{start}\n{end}", "{start}\n{end}\n{end}",
))
def test_generator_refuses_missing_duplicate_or_reversed_markers(reference, document):
    with pytest.raises(ValueError):
        reference.update_reference(document.format(start=reference.START, end=reference.END))
