"""One golden comparison per compaction branch-family scenario."""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from tests.behaviour.compress_trace_harness import (
    GOLDEN,
    OBSERVED_ATTRIBUTES,
    SCENARIOS,
    Scenario,
    compaction,
    first_difference,
    observe,
)


def _self_attributes(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            names.add(node.attr)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "setattr", "hasattr"} and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            names.add(node.args[1].value)
    return names


def _transitive_attributes(source: str) -> set[str]:
    methods = {
        node.name: node
        for cls in ast.parse(source).body
        if isinstance(cls, ast.ClassDef) and cls.name in {"CompactionMixin", "LCMEngine"}
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = ["_compress_impl"]
    visited: set[str] = set()
    names: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        method = methods[name]
        names.update(_self_attributes(method))
        for node in ast.walk(method):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                    and node.func.attr in methods):
                pending.append(node.func.attr)
    return names


def test_compress_observed_attributes_complete() -> None:
    """Phase extraction may move references, but must not shrink the observer."""
    engine_class = importlib.import_module("hermes_lcm.engine").LCMEngine
    discovered = _transitive_attributes(inspect.getsource(compaction))
    # Method fingerprints remain in the golden, but newly called methods are
    # not engine-run state. __dict__ is the reflection container, not a field.
    state_names = {
        name for name in discovered
        if name != "__dict__"
        and not callable(inspect.getattr_static(engine_class, name, None))
        and not isinstance(inspect.getattr_static(engine_class, name, None), (staticmethod, classmethod))
    }
    frozen = set(OBSERVED_ATTRIBUTES)
    assert tuple(sorted(frozen)) == OBSERVED_ATTRIBUTES, "Keep OBSERVED_ATTRIBUTES sorted and unique"
    missing = state_names - frozen
    assert not missing, (
        f"Unobserved compaction state: {sorted(missing)}. "
        "Add it to OBSERVED_ATTRIBUTES deliberately and regenerate the golden fixture."
    )
    # A name moved into another engine method/module is still valid. Only a
    # name absent from the whole engine class (including mixins) is stale.
    engine_names: set[str] = set()
    for cls in engine_class.__mro__:
        if not cls.__module__.startswith("hermes_lcm."):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        engine_names.update(_self_attributes(tree))
        engine_names.update(vars(cls))
    removed = frozen - engine_names
    assert not removed, f"Observed attributes removed from the whole engine class: {sorted(removed)}"


def test_transitive_attributes_follow_helpers_and_stop_at_module_boundary() -> None:
    source = '''
class CompactionMixin:
    def _compress_impl(self):
        return self._phase()
    def _phase(self):
        self._nested()
        return self._compress_impl()  # cycle
    def _nested(self):
        self._new_state = getattr(self, "_read_state", None)
        def closure():
            setattr(self, "_written_state", hasattr(self, "_optional_state"))
        self._external()
    def _unreachable(self):
        return self._unreachable_state
class OtherMixin:
    def _external(self):
        return self._other_module_state
'''
    assert _transitive_attributes(source) == {
        "_phase", "_nested", "_compress_impl", "_new_state", "_read_state",
        "_written_state", "_optional_state", "_external",
    }


@pytest.mark.parametrize("case", SCENARIOS, ids=lambda case: case.name)
def test_compress_golden(case: Scenario, tmp_path: Path) -> None:
    expected = json.loads(GOLDEN.read_text())[case.name]
    assert len(json.dumps(expected, indent=0).encode()) <= 40000, case.name
    assert GOLDEN.stat().st_size <= 1500000
    actual = observe(case, tmp_path)
    difference = first_difference(expected, actual)
    assert difference is None, f"{case.name}: first divergent event {difference}"
