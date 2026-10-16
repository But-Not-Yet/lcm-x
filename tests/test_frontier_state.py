"""Characterize the legacy frontier attributes and engine copy boundaries."""

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


DEFAULTS: dict[str, Any] = {
    "_last_compacted_store_id": 0,
    "_ingest_cursor": 0,
    "_ingest_cursor_needs_reconcile": False,
    "_compress_commit_proof": None,
    "_last_emission_descriptors": None,
}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Any]:
    instance = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        yield instance
    finally:
        instance.shutdown()


def test_initial_values_and_assignment_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, Any]] = []

    def record(instance: Any, name: str, value: Any) -> None:
        if name in DEFAULTS:
            writes.append((name, value))
        object.__setattr__(instance, name, value)

    monkeypatch.setattr(LCMEngine, "__setattr__", record)
    instance = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        assert writes == list(DEFAULTS.items())
        for name, value in DEFAULTS.items():
            assert getattr(instance, name) is value
    finally:
        instance.shutdown()


@pytest.mark.parametrize("name,values", [
    ("_ingest_cursor", [0, 1001, -2]),
    ("_ingest_cursor_needs_reconcile", [False, True, False]),
    ("_last_compacted_store_id", [0, 2002, -3]),
    ("_compress_commit_proof", [None, {}, {"consulted": False}, None]),
    ("_last_emission_descriptors", [None, {}, {"rows": []}, None]),
])
def test_legacy_assignment_keeps_value_type_and_identity(
    engine: Any, name: str, values: list[Any],
) -> None:
    for value in values:
        setattr(engine, name, value)
        assert getattr(engine, name) is value
        assert type(getattr(engine, name)) is type(value)


def test_dict_mutations_and_dynamic_clear_keep_identity(engine: Any) -> None:
    proof: dict[str, Any] = {"consulted": False}
    descriptors: dict[str, Any] = {"rows": []}
    engine._compress_commit_proof = proof
    engine._last_emission_descriptors = descriptors
    engine._compress_commit_proof["consulted"] = True
    engine._compress_commit_proof["end_consumed"] = True
    engine._last_emission_descriptors["rows"].append("row")
    assert proof == {"consulted": True, "end_consumed": True}
    assert descriptors == {"rows": ["row"]}
    for name in ("_compress_commit_proof", "_last_emission_descriptors"):
        setattr(engine, name, None)
        assert getattr(engine, name) is None


@pytest.mark.parametrize("mode", ["clone", "deepcopy"])
def test_fresh_engines_reset_frontier_without_sharing(engine: Any, mode: str) -> None:
    engine._ingest_cursor = 1001
    engine._ingest_cursor_needs_reconcile = True
    engine._last_compacted_store_id = 2002
    engine._compress_commit_proof = {"parent": True}
    engine._last_emission_descriptors = {"parent": True}
    clone = engine.clone_for_agent() if mode == "clone" else copy.deepcopy(engine)
    try:
        for name, value in DEFAULTS.items():
            assert getattr(clone, name) is value
        clone._compress_commit_proof = {"child": True}
        clone._last_emission_descriptors = {"child": True}
        assert engine._compress_commit_proof == {"parent": True}
        assert engine._last_emission_descriptors == {"parent": True}
        assert engine._ingest_cursor == 1001
    finally:
        clone.shutdown()


def test_shallow_copy_matches_default_dict_and_aliases(engine: Any) -> None:
    engine._ingest_cursor = 1001
    engine._last_compacted_store_id = 2002
    engine._compress_commit_proof = {"consulted": False}
    engine._last_emission_descriptors = {"rows": []}
    # The default shallow-copy state transfer, independent of __copy__.
    default = type(engine).__new__(type(engine))
    default.__dict__.update(engine.__dict__)
    clone = copy.copy(engine)
    assert clone.__dict__.keys() == default.__dict__.keys()
    for key, value in default.__dict__.items():
        if key != "_frontier":
            assert clone.__dict__[key] is value
    for name in DEFAULTS:
        assert getattr(clone, name) is getattr(engine, name)
    clone._ingest_cursor = 3003
    clone._last_compacted_store_id = 4004
    clone._ingest_cursor_needs_reconcile = True
    assert engine._ingest_cursor == 1001
    assert engine._last_compacted_store_id == 2002
    assert engine._ingest_cursor_needs_reconcile is False
    clone._compress_commit_proof["consulted"] = True
    clone._last_emission_descriptors["rows"].append("row")
    assert engine._compress_commit_proof == {"consulted": True}
    assert engine._last_emission_descriptors == {"rows": ["row"]}
    clone._compress_commit_proof = None
    clone._last_emission_descriptors = None
    assert engine._compress_commit_proof == {"consulted": True}
    assert engine._last_emission_descriptors == {"rows": ["row"]}


@pytest.mark.parametrize("name", DEFAULTS)
@pytest.mark.parametrize("factory", [LCMEngine.__new__, object.__new__])
def test_uninitialized_engine_missing_then_set(name: str, factory: Any) -> None:
    instance = factory(LCMEngine)
    sentinel = object()
    assert not hasattr(instance, name)
    assert getattr(instance, name, sentinel) is sentinel
    with pytest.raises(AttributeError):
        getattr(instance, name)
    setattr(instance, name, DEFAULTS[name])
    assert hasattr(instance, name)
    assert getattr(instance, name) is DEFAULTS[name]


def test_shallow_copy_of_uninitialized_engine_keeps_missing_names() -> None:
    instance = object.__new__(LCMEngine)
    clone = copy.copy(instance)
    assert clone.__dict__ == instance.__dict__ == {}
    for name in DEFAULTS:
        assert not hasattr(clone, name)


def test_engine_has_no_custom_pickle_hooks_or_slots() -> None:
    for cls in LCMEngine.__mro__:
        if cls is not object:
            assert not {"__getstate__", "__setstate__", "__reduce__", "__reduce_ex__", "__slots__"}.intersection(vars(cls))
