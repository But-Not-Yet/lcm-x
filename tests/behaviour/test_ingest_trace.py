"""Ingest's frozen observable protocol and independent observer inventory."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour.ingest_trace_harness import (
    GOLDEN, OBSERVED_ATTRIBUTES, SCENARIOS, TRACE_GUIDANCE, IngestCanonical,
    engine_attributes, engine_module, first_difference, observe,
    install_clock, reopen, search_observation, snapshot,
)
from tests.harness import engine_context, pin_char_counter


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


def test_search_observation_detects_unindexed_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=200000, conversation_id="trace-conversation", embeddings_enabled=False) as engine:
        first = [{"role": "user", "content": "indexed trace observation"}]
        engine._ingest_messages(first)
        assert search_observation(engine, first) == [[first[0]["content"], [[1, "S", first[0]["content"]]]]]
        # A real SQLite trigger failure leaves durable bytes but no FTS entry.
        engine._store._conn.execute("DROP TRIGGER msg_fts_insert")
        missing = {"role": "user", "content": "unindexed distinct needle"}
        engine._ingest_messages([*first, missing])
        observations = dict(search_observation(engine, [missing]))
        assert engine._store.get_session_count("S") == 2
        assert observations[missing["content"]] == []


def test_reopen_discards_instance_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=200000, conversation_id="trace-conversation", embeddings_enabled=False) as old, ExitStack() as engines:
        old._host_uid_lineage_cache = {"stale": True}
        old._host_uid_counters = {"stale": 7}
        old._engine_uids_pending = {"stale": True}
        fresh = reopen(old, tmp_path, engines)
        assert fresh is not old
        assert fresh._config is old._config
        for name in ("_host_uid_lineage_cache", "_host_uid_counters", "_engine_uids_pending"):
            assert "stale" not in getattr(fresh, name, {})


def test_snapshot_keeps_labelled_messages_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=200000, embeddings_enabled=False) as engine:
        message = {"role": "assistant", "content": "retained identity"}
        engine._last_active_replay_messages = [message]
        canonical = IngestCanonical(tmp_path, [])
        snapshot(engine, canonical)
        engine._last_active_replay_messages = []
        assert canonical.retained_messages[id(message)] is message


@pytest.mark.parametrize("name", [case.name for case in SCENARIOS if case.shape == "uid"])
def test_uid_scenarios_bind_durable_rows(name: str, tmp_path: Path) -> None:
    payloads: list[str] = []
    observe(next(case for case in SCENARIOS if case.name == name), tmp_path, payloads=payloads)
    state = json.loads(next(event.removeprefix("state-after ") for event in payloads
                            if event.startswith("state-after ")))
    bindings = state["tables"]["host_uid_bindings"]
    assert len(bindings) == (1 if name == "uid-bindings-new" else 4)
    assert all(row[3] == "canonical" for row in bindings)
    assert state["_host_uid_counters"]["unbound.agree_new"] == len(bindings)
    assert not any(key.startswith("skipped.") for key in state["_host_uid_counters"])
