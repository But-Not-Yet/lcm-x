"""Frozen reconciliation behavior and an independent engine-state inventory."""
from __future__ import annotations

import importlib
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour.reconcile_trace_harness import (
    GOLDEN,
    OBSERVED_ATTRIBUTES,
    SCENARIOS,
    TRACE_GUIDANCE,
    Canonical,
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
    names = [case.name for case in SCENARIOS]
    assert len(names) == len(set(names)), "Scenario names must be unique.\n" + TRACE_GUIDANCE
    assert set(data) == set(names), (
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


def test_direct_regeneration_isolates_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness = importlib.import_module("tests.behaviour.reconcile_trace_harness")
    from tests.phase0_hygiene import product_env_keys

    keys = product_env_keys(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "false")
    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "9999")
    monkeypatch.setenv("LCM_TRACE_UNKNOWN_SETTING", "inherited")
    destination = tmp_path / "trace.json"
    monkeypatch.setattr(harness, "GOLDEN", destination)
    monkeypatch.setattr(harness, "SCENARIOS", SCENARIOS[:1])
    monkeypatch.setattr(sys, "argv", ["trace", "--repo", str(harness.ROOT), "--out", str(destination)])

    def observe_clean(case: object, root: Path) -> list[str]:
        assert not {key for key in os.environ if key.startswith("LCM_") or key in keys}
        return ["clean settings"]

    monkeypatch.setattr(harness, "observe", observe_clean)
    harness.main()
    assert json.loads(destination.read_text()) == {SCENARIOS[0].name: ["clean settings"]}


@pytest.mark.parametrize("equals", [False, True], ids=["separate", "equals"])
def test_repo_option_binds_selected_tree(tmp_path: Path, equals: bool) -> None:
    root = Path(__file__).resolve().parents[2]
    selected = tmp_path / "selected"
    selected.mkdir()
    for path in root.iterdir():
        if path.suffix == ".py" or (path.is_dir() and not path.name.startswith(".")):
            (selected / path.name).symlink_to(path, target_is_directory=path.is_dir())
    option = [f"--repo={selected}"] if equals else ["--repo", str(selected)]
    script = """
import runpy, sys
try:
    runpy.run_module("tests.behaviour.reconcile_trace_harness", run_name="__main__")
except SystemExit as exc:
    if exc.code:
        raise
print("Product package:", sys.modules["hermes_lcm"].__path__[0])
"""
    result = subprocess.run([sys.executable, "-c", script, *option, "--help"],
                            cwd=root, capture_output=True, text=True, check=True)
    assert f"Product package: {selected.resolve()}" in result.stdout


@pytest.mark.parametrize("option", [["--repo"], ["--repo", "--out"], ["--repo="]])
def test_repo_option_rejects_missing_value(option: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", "tests.behaviour.reconcile_trace_harness", *option],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 2
    assert "argument --repo: expected" in result.stderr


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
@pytest.mark.parametrize("nonce", ["12345678-1234-1234-1234-123456789abc", "12345678123412341234123456789abc"])
def test_uuid_content_changes_remain_observable(tmp_path: Path, role: str, nonce: str) -> None:
    changed = nonce[:-1] + "d"
    original = Canonical(tmp_path, []).value({"role": role, "content": nonce})
    replacement = Canonical(tmp_path, []).value({"role": role, "content": changed})
    assert first_difference([json.dumps(original)], [json.dumps(replacement)]) is not None
    assert original["content"] == nonce
    # SQL and durable store snapshots use the same canonicalizer.
    assert Canonical(tmp_path, []).text(f"SELECT '{nonce}'") != Canonical(tmp_path, []).text(f"SELECT '{changed}'")
    assert Canonical(tmp_path, []).value([(17, role, nonce)])[0] == [17, role, nonce]
    assert Canonical(tmp_path, []).value({"content": [{"type": "text", "text": nonce}]}) == {
        "content": [{"type": "text", "text": nonce}]}


@pytest.mark.parametrize("field", ["message_uid", "_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid"])
def test_generated_identity_fields_normalize_without_changing_store_ids(tmp_path: Path, field: str) -> None:
    first, second = "12345678123412341234123456789abc", "fedcba9876543210fedcba9876543210"
    value = [first] if field.endswith("uids") else first
    replacement = [second] if field.endswith("uids") else second
    assert Canonical(tmp_path, []).value({field: value, "store_id": 17}) == Canonical(tmp_path, []).value({
        field: replacement, "store_id": 17})
