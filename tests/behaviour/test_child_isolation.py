"""A bypassed child must leave its shared engine's foreground compress state alone."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest

from tests.harness import ScriptedSummarizer, assistant, engine_context, pin_char_counter, prose, user


# Audited child-path writes, plus the commit proof cleared before the bypass guard.
CHILD_WRITES = """
_compress_dropped_ignored_rows _compress_forced_overflow _compress_hidden_rows_consumed
_compress_host_rows_consumed _compress_occurrences _foreground_budget _hold_fit_only_requested
_host_fallback_compressor _host_fallback_session_id _ingest_cursor _last_compress_aborted
_last_compress_leaves _last_compression_noop_reason _last_compression_status _last_hidden_backlog
_last_stub_first_exit _no_progress_candidate _objective_only_noop _pending_emission_candidates
_preflight_automatic_request _preflight_below_threshold_cleanup_only
_preflight_cleanup_only_due_to_boundary_cooldown _pressure_yield_blocked_streak
_pressure_yield_invocation_verdict _pressure_yield_preflight_candidate _pressure_yield_streak_counted
_pressure_yield_tail_token_limit _stub_first_exit_now _survival_fit_emission _survival_fit_reason
_compress_commit_proof
""".split()


class ParentStateChanged(AssertionError):
    """Identify changes to the parent state captured around the child call."""


def _snapshot(engine):
    # Retain budget identity AND copy its fields, so in-place mutations are visible.
    snapshot = {name: (getattr(engine, name, None) if name in {
        "_foreground_budget", "_host_fallback_compressor"
    } else deepcopy(getattr(engine, name, None))) for name in CHILD_WRITES}
    budget = snapshot["_foreground_budget"]
    if budget is not None:
        snapshot["_foreground_budget"] = (
            budget, deepcopy({k: v for k, v in vars(budget).items() if k != "estimates"}),
            deepcopy(vars(budget.estimates)),
        )
    return snapshot


def _history():
    return [row for i in range(6) for row in (
        {**user(prose(800, seed=i)), "timestamp": float(2 * i + 1)},
        {**assistant(prose(800, seed=i + 10)), "timestamp": float(2 * i + 2)},
    )]


@pytest.mark.parametrize("schedule", ["sequential", "interleaved"])
@pytest.mark.parametrize("group", ["state", "proof", "status", "control"])
def test_child_preserves_parent_compress_state(tmp_path, monkeypatch, schedule, group):
    pin_char_counter(monkeypatch)
    summaries = ScriptedSummarizer().install(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="parent", conversation_id="parent",
                        fresh_tail_count=2, leaf_chunk_tokens=400,
                        large_output_externalization_enabled=False) as engine:
        # The native fallback is an external provider boundary; use its real deterministic trim.
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        view = _history()
        snapshots = []

        def child():
            engine._mark_thread_context_stateless("child")
            try:
                assert engine._bypasses_lcm_context_management()
                result = engine.compress([user("child request"), assistant("child reply")], force=True)
                assert result and engine.current_session_id == "parent"
                assert engine._store.get_session_count("child") == 0
            finally:
                engine._clear_thread_context_stateless("child")

        if schedule == "sequential":
            result = engine.compress(view, force=True)
            snapshots.append(_snapshot(engine))
            if group != "control":
                child()
            snapshots.append(_snapshot(engine))
        else:
            ready, finished = Barrier(2, timeout=10), Barrier(2, timeout=10)
            record_proof = engine._record_compress_commit_proof

            def pause_parent(messages, result):
                record_proof(messages, result)
                if not engine._bypasses_lcm_context_management():
                    snapshots.append(_snapshot(engine))
                    ready.wait()
                    finished.wait()
                    snapshots.append(_snapshot(engine))

            monkeypatch.setattr(engine, "_record_compress_commit_proof", pause_parent)

            def run_child():
                ready.wait()
                try:
                    if group != "control":
                        child()
                finally:
                    finished.wait()

            with ThreadPoolExecutor(max_workers=2) as pool:
                parent = pool.submit(engine.compress, view, force=True)
                delegated = pool.submit(run_child)
                delegated.result(timeout=15)
                result = parent.result(timeout=15)

        assert summaries.calls and len(result) < len(view)
        before, after = snapshots
        assert before["_compress_commit_proof"] is not None
        assert before["_last_compression_status"] == "compacted"
        if schedule == "interleaved":
            assert before["_foreground_budget"] is not None
        names = (set(CHILD_WRITES) if group == "control" else
                 {"_compress_commit_proof"} if group == "proof" else
                 {"_last_compression_status", "_last_compression_noop_reason"} if group == "status" else
                 set(CHILD_WRITES) - {"_compress_commit_proof", "_last_compression_status",
                                      "_last_compression_noop_reason"})
        changed = {name: (before[name], after[name]) for name in names if before[name] != after[name]}
        if changed:
            raise ParentStateChanged(changed)


@pytest.mark.parametrize("host_fails", [False, True])
def test_interleaved_child_stop_reason_and_native_state(tmp_path, monkeypatch, caplog, host_fails):
    """The native path uses a child cache and clock while a parent's proof is live."""
    import logging
    import sys
    import threading
    import time
    from types import ModuleType

    pin_char_counter(monkeypatch)
    ScriptedSummarizer().install(monkeypatch)
    clock = threading.local()
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0 + getattr(clock, "elapsed", 0.0))
    created = []

    class NativeCompressor:
        compression_count = 0
        _last_compress_aborted = False

        def __init__(self, **kwargs):
            created.append(self)

        def compress(self, messages, **kwargs):
            clock.elapsed = 7.5
            if host_fails:
                raise RuntimeError("scripted native failure")
            self.compression_count += 1
            return messages

    native = ModuleType("agent.context_compressor")
    native.ContextCompressor = NativeCompressor
    monkeypatch.setitem(sys.modules, "agent.context_compressor", native)
    with engine_context(tmp_path, window=200_000, session_id="parent", conversation_id="parent",
                        fresh_tail_count=2, leaf_chunk_tokens=400, temporal_rollups_enabled=False,
                        large_output_externalization_enabled=False) as engine:
        engine.model = "test-model"
        engine._last_compress_aborted = "parent sentinel"
        ready, finished = Barrier(2, timeout=10), Barrier(2, timeout=10)
        record_proof = engine._record_compress_commit_proof
        snapshots, threads = [], {}

        def pause_parent(messages, result):
            record_proof(messages, result)
            if engine._bypasses_lcm_context_management():
                return
            snapshots.append(_snapshot(engine))
            ready.wait()
            finished.wait()
            snapshots.append(_snapshot(engine))

        monkeypatch.setattr(engine, "_record_compress_commit_proof", pause_parent)

        def parent():
            threads["parent"] = threading.get_ident()
            return engine.compress(_history(), force=True)

        def child():
            threads["child"] = threading.get_ident()
            ready.wait()
            engine._mark_thread_context_stateless("child")
            try:
                view = [user("child request"), assistant("child reply")]
                result = engine.compress(view, force=True)
                assert result is view
                assert engine.last_compression_status == "host_fallback"
                assert engine.get_status()["last_compression_noop_reason"] == "LCM bypassed: auxiliary thread context"
                record_proof(view, result)  # Its own guard must not clear the parent's proof either.
                engine.last_compression_status = ""  # The host's writable reset is child-local too.
                assert engine.last_compression_status == ""
                assert engine._thread_context.foreground_budget is None
            finally:
                engine._clear_thread_context_stateless("child")
                finished.wait()

        with caplog.at_level(logging.INFO, logger="hermes_lcm"), ThreadPoolExecutor(max_workers=2) as pool:
            foreground, delegated = pool.submit(parent), pool.submit(child)
            delegated.result(timeout=15)
            assert foreground.result(timeout=15)
        assert len(created) == 1
        stops = {record.thread: record.getMessage() for record in caplog.records
                 if record.getMessage().startswith("LCM compaction stop:")}
        assert len(stops) == 2
        assert "reason=host_fallback leaves=0" in stops[threads["child"]]
        assert "host=7.5s" in stops[threads["child"]]
        assert "reason=compacted" in stops[threads["parent"]]
        assert " host=" not in stops[threads["parent"]]
        assert snapshots[0] == snapshots[1]
