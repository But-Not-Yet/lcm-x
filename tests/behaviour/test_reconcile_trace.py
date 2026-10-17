"""Frozen reconciliation behavior and an independent engine-state inventory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour.reconcile_trace_harness import (
    GOLDEN,
    OBSERVED_ATTRIBUTES,
    SCENARIOS,
    TRACE_GUIDANCE,
    engine_attributes,
    first_difference,
    observe,
    reconcile,
)


@pytest.mark.parametrize("case", SCENARIOS, ids=lambda case: case.name)
def test_reconcile_golden(case: Any, tmp_path: Path) -> None:
    expected: dict[str, list[str]] = json.loads(GOLDEN.read_text())
    actual = observe(case, tmp_path)
    difference = first_difference(expected[case.name], actual)
    assert difference is None, f"{case.name}: {difference}\n" + TRACE_GUIDANCE


def test_observed_attribute_inventory_is_complete() -> None:
    discovered = engine_attributes(Path(reconcile.__file__).read_text())
    assert discovered <= OBSERVED_ATTRIBUTES, (
        f"Unobserved engine state: {sorted(discovered - OBSERVED_ATTRIBUTES)}; update OBSERVED_ATTRIBUTES.\n"
        + TRACE_GUIDANCE)
    # Prove this guard is sensitive to new state used by a transitive helper,
    # independently of the frozen constant and without editing product source.
    source = Path(reconcile.__file__).read_text()
    source = source.replace('return f"{prefix}:', 'self._trace_inventory_probe\n        return f"{prefix}:', 1)
    assert "_trace_inventory_probe" in engine_attributes(source)
    assert "_trace_inventory_probe" not in OBSERVED_ATTRIBUTES


def test_fixture_scenario_inventory_and_size() -> None:
    data: dict[str, list[str]] = json.loads(GOLDEN.read_text())
    assert set(data) == {case.name for case in SCENARIOS}, (
        "Scenario inventory differs; update SCENARIOS and the fixture scenario keys together.\n" + TRACE_GUIDANCE)
    assert GOLDEN.stat().st_size <= 800_000, (
        "Fixture exceeds 800000 bytes; review the fixture size bound.\n" + TRACE_GUIDANCE)
    assert all(len(json.dumps(trace).encode()) <= 40_000 for trace in data.values()), (
        "A trace exceeds 40000 bytes; review the trace size bound.\n" + TRACE_GUIDANCE)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_first_divergence_reports_cursor_read_order_and_state(index: int) -> None:
    original = ["return 0", "store get_session_tail", "attr _last_ingest_reconciliation={}"]
    changed = list(original)
    changed[index] += " changed"
    assert first_difference(original, changed) == (
        f"first divergence event {index}: expected {original[index]!r}, got {changed[index]!r}")
    assert first_difference(original, original) is None
    assert "<end>" in str(first_difference(original, original[:1]))
