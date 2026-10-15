"""Pin values independently of the range/choice/drop diagnostics added later."""

import math

import pytest

from hermes_lcm import config


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    for key in list(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


# key suffix, field, raw, value used, warning before this fix, warning after.
ROWS = [
    ("CONTEXT_THRESHOLD", "context_threshold", raw, value, malformed, warns)
    for raw, value, malformed, warns in [
        (None, .75, False, False), (".5", .5, False, False),
        ("1", 1., False, False), ("0", 0., False, True),
        ("-1", -1., False, True), ("1.01", 1.01, False, True),
        ("nan", float("nan"), False, True), ("inf", float("inf"), False, True),
        ("bad", .75, True, True), ("", .75, True, True),
    ]
] + [
    ("SURVIVAL_RESERVE", "survival_reserve", raw, value, malformed, warns)
    for raw, value, malformed, warns in [
        (None, .15, False, False), (".15", .15, False, False),
        ("0", 0., False, False), (".9", .9, False, False),
        ("-.1", -.1, False, True), (".91", .91, False, True),
        ("nan", float("nan"), False, True), ("inf", float("inf"), False, True),
        ("-inf", float("-inf"), False, True), ("bad", .15, True, True),
        ("", .15, False, False),
    ]
] + [
    (key, field, raw, value, malformed, warns)
    for key, field, default in [
        ("LEAF_CHUNK_TOKENS", "leaf_chunk_tokens", 24000),
        ("SUMMARY_TIMEOUT_MS", "summary_timeout_ms", 120000),
    ]
    for raw, value, malformed, warns in [
        (None, default, False, False), (str(default), default, False, False),
        ("1", 1, False, False), ("0", 0, False, True),
        ("-1", -1, False, True), ("bad", default, True, True),
        ("", default, True, True),
    ]
] + [
    (key, field, raw, default if raw is None else raw, False, warns)
    for key, field, default, values in [
        ("EMBEDDING_PROVIDER", "embedding_provider", "", [
            (None, False), ("", False), (" \t", False), (" VoYaGeAI ", False),
            ("voyage", False), ("ollama", False), ("fastembed", False),
            ("fast-embed", False), ("openai-compatible", False),
            ("openai", False), ("siliconflow", False), ("voyge", True),
        ]),
        ("PREANSWER_EVIDENCE_MODE", "preanswer_evidence_mode", "", [
            (None, False), ("", False), (" \t", False), ("off", False),
            (" LEGACY_SELECTIVE ", False), ("requirements_v1", False), ("bogus", True),
        ]),
        ("EMBEDDING_STORAGE_DTYPE", "embedding_storage_dtype", "float32", [
            (None, False), ("", False), (" \t", False), ("float32", False),
            (" INT8 ", False), ("float8", True),
        ]),
    ]
    for raw, warns in values
] + [
    ("EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS", "empty_lifecycle_gc_max_age_hours", raw,
     value, False, warns)
    for raw, value, warns in [
        (None, 24., False), ("24", 24., False), ("0", 0., False),
        ("-1", -1., False), ("inf", float("inf"), False),
        ("bad", 24., True), ("", 24., True),
    ]
] + [
    ("RECALL_ARM_WEIGHTS", "recall_arm_weights", raw, value, False, warns)
    for raw, value, warns in [
        (None, {"fts": .5, "summary": 1., "chunk": 1.}, False),
        (" FTS =0,summary=2", {"fts": 0., "summary": 2., "chunk": 1.}, False),
        ("fts=1,fts=2", {"fts": 2., "summary": 1., "chunk": 1.}, False),
        ("unknown=1", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("fts=nan", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("fts=inf", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("fts=-1", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("broken,fts=bad,chunk=.2", {"fts": .5, "summary": 1., "chunk": .2}, True),
        ("", {"fts": .5, "summary": 1., "chunk": 1.}, True),
    ]
] + [
    ("MODEL_THRESHOLDS", "model_thresholds", raw, value, False, warns)
    for raw, value, warns in [
        (None, {}, False), ("model:.5", {"model": .5}, False),
        (" model :1", {"model": 1.}, False), ("ns:model:.2", {"ns:model": .2}, False),
        ("model:0", {}, True), ("model:1.1", {}, True), ("model:nan", {}, True),
        ("model:inf", {}, True), ("broken,:.5,model:bad,ok:.3", {"ok": .3}, True),
        ("", {}, True),
    ]
]


@pytest.mark.parametrize("key,field,raw,value,before,after", ROWS)
def test_values_unchanged(monkeypatch, key, field, raw, value, before, after):
    if raw is not None:
        monkeypatch.setenv("LCM_" + key, raw)
    actual = getattr(config.LCMConfig.from_env(), field)
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(actual)
    else:
        assert actual == value


@pytest.mark.parametrize("key,field,raw,value,before,after", ROWS)
def test_warning_presence(monkeypatch, key, field, raw, value, before, after):
    if raw is not None:
        monkeypatch.setenv("LCM_" + key, raw)
    warnings = config.LCMConfig.from_env().config_source_warnings
    assert bool(warnings) is before
    if warnings:
        assert len(warnings) == 1
        assert "LCM_" + key in warnings[0]


def test_parsers_keep_dictionary_contract(caplog):
    assert config._parse_arm_weights("unknown=1,fts=nan,chunk=.2,fts=-1", {"fts": .5, "chunk": 1.}) == {
        "fts": .5, "chunk": .2,
    }
    assert [record.message for record in caplog.records] == [
        "LCM_RECALL_ARM_WEIGHTS: negative weight -1.0 for arm 'fts' is invalid; using default 0.5"
    ]
    assert config._parse_model_thresholds_env("broken,:.2,bad:nan,ok:.5") == {"ok": .5}
