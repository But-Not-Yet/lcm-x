"""Ingest's frozen observable protocol and independent observer inventory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour.ingest_trace_harness import (
    GOLDEN, OBSERVED_ATTRIBUTES, SCENARIOS, TRACE_GUIDANCE, IngestCanonical,
    engine_attributes, engine_module, first_difference, observe,
)


@pytest.mark.parametrize("case", SCENARIOS, ids=lambda case: case.name)
def test_ingest_golden(case: Any, tmp_path: Path) -> None:
    expected: dict[str, list[str]] = json.loads(GOLDEN.read_text())
    payloads: list[str] = []
    actual = observe(case, tmp_path, payloads=payloads)
    difference = first_difference(expected[case.name], actual)
    if difference is not None:
        (tmp_path / "actual-payload.json").write_text(json.dumps(payloads, ensure_ascii=False))
    assert difference is None, f"{case.name}: {difference}\n{TRACE_GUIDANCE}"


def test_observer_completeness(tmp_path: Path) -> None:
    discovered, methods = engine_attributes(Path(engine_module.__file__).parent)
    assert "_ingest_messages" in methods
    assert discovered <= OBSERVED_ATTRIBUTES, f"Unobserved engine state: {sorted(discovered - OBSERVED_ATTRIBUTES)}"
    assert "_ingest_cursor" in discovered
    assert "_identity_anchor_commit" in methods
    # The closure follows a future extracted phase and that phase's helper.
    (tmp_path / "engine.py").write_text(
        "class LCMEngine:\n"
        " def _ingest_messages(self): return self._moved_phase()\n"
        " def _moved_phase(self): return self._helper()\n"
        " def _helper(self): return self._trace_inventory_probe\n")
    extra, reachable = engine_attributes(tmp_path)
    assert extra == {"_trace_inventory_probe"}
    assert reachable == {"_ingest_messages", "_moved_phase", "_helper"}
    assert not extra <= OBSERVED_ATTRIBUTES


def test_fixture_scenario_inventory_and_size() -> None:
    data: dict[str, list[str]] = json.loads(GOLDEN.read_text())
    names = [case.name for case in SCENARIOS]
    assert len(names) == 190
    assert len(names) == len(set(names))
    assert set(data) == set(names), TRACE_GUIDANCE
    assert GOLDEN.stat().st_size <= 1_500_000
    assert all(len(json.dumps(trace).encode()) <= 40_000 for trace in data.values())


def test_first_divergence() -> None:
    assert first_difference(["BEGIN", "row", "COMMIT"], ["BEGIN", "row", "ROLLBACK"]) == (
        "first divergence event 2: expected 'COMMIT', got 'ROLLBACK'")
    assert first_difference(["row"], []) == "first divergence event 0: expected 'row', got '<end>'"
    assert first_difference([], []) is None


def test_occurrence_uids_and_structured_payload_bytes_stay_distinct(tmp_path: Path) -> None:
    canonical = IngestCanonical(tmp_path, [])
    first = {"role": "user", "content": {"mtime_ns": 100, "message_uid": "nested"},
             "message_uid": "12345678123412341234123456789abc", "timestamp": 10.}
    second = {**first, "message_uid": "fedcba9876543210fedcba9876543210"}
    assert canonical.value(first) != canonical.value(second)
    assert canonical.value(first)["content"] == first["content"]
    assert canonical.value({"content": "mtime_ns=123"}) == {"content": "mtime_ns=123"}
    assert canonical.value(first) != canonical.value({**first, "timestamp": 11.})
