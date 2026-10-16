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
    """Only the final isolation assertion may satisfy the expected defect."""


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
@pytest.mark.parametrize("group", [
    pytest.param("state", marks=pytest.mark.xfail(
        strict=True, raises=ParentStateChanged, reason="#1222 item 3: child resets parent compress state")),
    pytest.param("proof", marks=pytest.mark.xfail(
        strict=True, raises=ParentStateChanged, reason="#1222 item 3: child clears parent commit proof")),
    pytest.param("status", marks=pytest.mark.xfail(
        strict=True, raises=ParentStateChanged, reason="#1157: child overwrites parent diagnostic status")),
    "control",
])
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
