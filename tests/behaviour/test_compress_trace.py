"""One golden comparison per compaction branch-family scenario."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.behaviour.compress_trace_harness import GOLDEN, SCENARIOS, Scenario, first_difference, observe


@pytest.mark.parametrize("case", SCENARIOS, ids=lambda case: case.name)
def test_compress_golden(case: Scenario, tmp_path: Path) -> None:
    expected = json.loads(GOLDEN.read_text())[case.name]
    assert len(json.dumps(expected, indent=0).encode()) <= 40000, case.name
    assert GOLDEN.stat().st_size <= 1500000
    actual = observe(case, tmp_path)
    difference = first_difference(expected, actual)
    assert difference is None, f"{case.name}: first divergent event {difference}"
