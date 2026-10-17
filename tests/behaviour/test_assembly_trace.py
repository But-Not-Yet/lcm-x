"""Frozen assembly channels; intentional rebaseline only after a declared delta."""
from __future__ import annotations

import gc
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour.assembly_trace_harness import (
    EXTERNAL_ATTRIBUTES, GOLDEN, OBSERVED_ATTRIBUTES, SCENARIOS, TRACE_GUIDANCE, Scenario,
    dag_module, engine_attributes, engine_module, first_difference, observe,
)


def test_assembly_attribute_inventory() -> None:
    discovered = engine_attributes(inspect.getsource(engine_module))
    assert discovered == OBSERVED_ATTRIBUTES - EXTERNAL_ATTRIBUTES, (
        f'Assembly state inventory changed: added={sorted(discovered - OBSERVED_ATTRIBUTES)}, '
        f'removed={sorted(OBSERVED_ATTRIBUTES - EXTERNAL_ATTRIBUTES - discovered)}; '
        'update OBSERVED_ATTRIBUTES/EXTERNAL_ATTRIBUTES.\n' + TRACE_GUIDANCE)


@pytest.mark.parametrize('case', SCENARIOS, ids=lambda case: case.name)
def test_assembly_golden(case: Scenario, tmp_path: Path) -> None:
    expected = json.loads(GOLDEN.read_text())[case.name]
    actual = observe(case, tmp_path.resolve())
    difference = first_difference(expected, actual)
    assert difference is None, f'{case.name}: first divergence: {difference}\n' + TRACE_GUIDANCE


def test_assembly_defers_prior_cycle_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = importlib.import_module('hermes_lcm.vector_store')
    original = vectors.VectorStore._binary_fully_synced
    prior = dag_module.SummaryDAG(tmp_path / 'prior.db')
    prior.close()
    prior.cycle = prior
    pending = [prior]
    del prior
    threshold, enabled = gc.get_threshold(), gc.isenabled()

    def pressure(store: Any, *args: Any, **kwargs: Any) -> Any:
        gc.set_threshold(1000000, 1000000, 1000000)
        pending.clear()
        gc.set_threshold(1, 1, 1)
        # Force an automatic collection opportunity inside the SQL observation.
        allocations: list[list[object]] = [[] for _ in range(200)]
        assert len(allocations) == 200
        gc.set_threshold(*threshold)
        return original(store, *args, **kwargs)

    monkeypatch.setattr(vectors.VectorStore, '_binary_fully_synced', pressure)
    case = next(case for case in SCENARIOS if case.name == 'recall-on')
    try:
        actual = observe(case, tmp_path.resolve())
    finally:
        gc.set_threshold(*threshold)
    assert gc.isenabled() == enabled
    difference = first_difference(json.loads(GOLDEN.read_text())[case.name], actual)
    assert difference is None, f'{case.name}: first divergence: {difference}\n' + TRACE_GUIDANCE
