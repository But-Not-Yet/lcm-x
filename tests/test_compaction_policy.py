"""The complete policy API is pure and agrees with the engine's ordered shells."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from hermes_lcm.compaction_policy import (
    PolicyFacts,
    assembly_token_cap,
    resolve_compaction_policy,
    threshold_tokens,
)


def facts(**changes):
    values = dict(
        raw_context_length=400_000, model="gpt-5.5", provider="openai-codex",
        configured_ratio=0.35, threshold_source="config_yaml:compression.threshold(floored)",
        model_thresholds={}, autoraise_enabled=True, max_assembly_tokens=0,
        reserve_tokens_floor=24_000, absolute_threshold_tokens=0,
        context_length_source="update_model",
    )
    return PolicyFacts(**(values | changes))


def test_complete_policy_values():
    assert asdict(resolve_compaction_policy(facts())) == {
        "raw_context_length": 400_000,
        "context_length": 272_000,
        "effective_context_length_cap": 272_000,
        "effective_context_length_reason": "codex_oauth_context_cap",
        "context_length_source": "update_model",
        "context_threshold": 0.85,
        "context_threshold_source": "codex_gpt55_autoraise",
        "context_threshold_autoraised": {"from": 0.35, "to": 0.85},
        "threshold_percent": 0.85,
        "threshold_tokens": 231_200,
        "disable_autoraise": False,
        "reserve_floor_disabled": False,
    }


@pytest.mark.parametrize("raw", [-1, 0, 1, 272_000, 400_000])
def test_absolute_and_disabled_reserve_flags(raw):
    result = resolve_compaction_policy(facts(
        raw_context_length=raw, absolute_threshold_tokens=130_000,
        reserve_tokens_floor=500_000,
    ))
    assert result.threshold_tokens == (130_000 if raw > 0 else 0)
    assert result.disable_autoraise is (raw > 0)
    assert result.reserve_floor_disabled is (raw > 0)


def test_frozen_records_and_repeatable_resolution(caplog):
    inputs = facts(model_thresholds={"gpt-5": 0.5, "gpt-5.5": 0.25})
    before = asdict(inputs)
    result = resolve_compaction_policy(inputs)
    assert result.context_threshold_source == "model_thresholds:gpt-5.5"
    assert result.context_threshold_autoraised == {"from": 0.35, "to": 0.25}
    assert result == resolve_compaction_policy(inputs)
    assert asdict(inputs) == before
    with pytest.raises(FrozenInstanceError):
        inputs.raw_context_length = 1
    with pytest.raises(FrozenInstanceError):
        result.threshold_tokens = 1
    assert caplog.records == []


@pytest.mark.parametrize("window,maximum,reserve,expected", [
    (0, 0, 10, (None, False)),
    (100, 0, 0, (None, False)),
    (100, 80, 10, (80, False)),
    (100, 80, 99, (1, False)),
    (100, 80, 100, (80, True)),
    (100, 0, 101, (None, True)),
])
def test_assembly_cap(window, maximum, reserve, expected):
    assert assembly_token_cap(window, maximum, reserve) == expected


@pytest.mark.parametrize("trigger,cap,absolute,expected", [
    (0, None, 0, 0), (0, 10, 0, 10), (-1, 10, 0, 10),
    (5, 10, 0, 5), (20, 10, 0, 10), (20, 10, 100, 100),
])
def test_threshold_token_precedence(trigger, cap, absolute, expected):
    assert threshold_tokens(trigger, cap, absolute) == expected
