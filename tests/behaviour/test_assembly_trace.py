"""Frozen assembly channels; intentional rebaseline only after a declared delta."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tests.behaviour.assembly_trace_harness import (
    EXTERNAL_ATTRIBUTES, GOLDEN, OBSERVED_ATTRIBUTES, SCENARIOS, Scenario,
    engine_attributes, engine_module, first_difference, observe,
)


def test_assembly_attribute_inventory() -> None:
    discovered = engine_attributes(inspect.getsource(engine_module))
    assert discovered == OBSERVED_ATTRIBUTES - EXTERNAL_ATTRIBUTES, (
        f'Assembly state inventory changed: added={sorted(discovered - OBSERVED_ATTRIBUTES)}, '
        f'removed={sorted(OBSERVED_ATTRIBUTES - EXTERNAL_ATTRIBUTES - discovered)}')


@pytest.mark.parametrize('case', SCENARIOS, ids=lambda case: case.name)
def test_assembly_golden(case: Scenario, tmp_path: Path) -> None:
    expected = json.loads(GOLDEN.read_text())[case.name]
    actual = observe(case, tmp_path)
    difference = first_difference(expected, actual)
    assert difference is None, f'{case.name}: first divergence: {difference}'
