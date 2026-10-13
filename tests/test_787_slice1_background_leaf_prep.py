"""#787 slice 1: background leaf preparation (default off) and promotion at the threshold crossing.

The provider stands in at ``escalation._invoke_summary_llm``, so the real escalation chain, its breaker keys and
spend guards run on both the worker and the turn thread. Every test uses its own process-wide preparation state.
"""

from __future__ import annotations

import contextvars
import importlib
import os
import threading

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.leaf_prep as leaf_prep
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SummarySpendGuard
from tests.test_issue_622_slow_load import _Ctx, _Manager, _install_host, _load_plugin

WINDOW = 60_000  # T = 45,000; P = 36,000; L = 45,000 - max(2 x 2,000, 6,000) = 39,000
CHUNK = 2_000
PAD = " alpha beta gamma delta epsilon" * 64
SUMMARY = "Earlier turns.\nExpand for details about: turns"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(
        tokens,
        "count_tokens",
        lambda text: (
            0
            if not text
            else (
                tokens._fallback_token_estimate(text)
                if isinstance(text, str)
                else len(text) // 4 + 1
            )
        ),
    )
    state = leaf_prep._LeafPrep()
    monkeypatch.setattr(leaf_prep, "_LEAF_PREP", state)
    yield state
    state.wait_idle(10)


class _Provider:
    """Counts provider calls per thread; thread-safe; optional per-call hook."""

    def __init__(self, result=SUMMARY):
        self.result, self.calls, self.lock, self.hook = (
            result,
            [],
            threading.Lock(),
            None,
        )

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        with self.lock:
            self.calls.append(threading.current_thread().name)
        if self.hook is not None:
            self.hook()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def on(self, thread_name: str) -> int:
        with self.lock:
            return sum(1 for name in self.calls if name == thread_name)


@pytest.fixture
def provider(monkeypatch):
    fake = _Provider()
    monkeypatch.setattr(escalation, "_invoke_summary_llm", fake)
    return fake


def _engine(tmp_path, *, window: int = WINDOW, **config) -> LCMEngine:
    settings = {
        "database_path": str(tmp_path / "lcm.db"),
        "background_leaf_prep_enabled": True,
        "fresh_tail_count": 2,
        "leaf_chunk_tokens": CHUNK,
        "context_threshold": 0.75,
        "threshold_full_sweep_enabled": True,
        "large_output_externalization_enabled": False,
        "large_output_active_replay_stubbing_enabled": False,
        "temporal_rollups_enabled": False,
        "l3_truncate_tokens": 2,
        "incremental_max_depth": 0,
        **config,
    }
    if os.environ.get(
        "LCM_787_RED_RECEIPT"
    ):  # the red receipt: every engine runs with the feature off
        settings["background_leaf_prep_enabled"] = False
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start(
        "S", platform="telegram", context_length=window, conversation_id="conv"
    )
    return engine


def _view(turns: int) -> list[dict]:
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows.append(
            {
                "role": "user",
                "content": f"[T{i}] user turn{PAD}",
                "timestamp": float(i + 1),
            }
        )
        rows.append(
            {
                "role": "assistant",
                "content": f"reply to T{i}{PAD}",
                "timestamp": float(i + 1),
            }
        )
    return rows


def _turn_end(engine, view, *, tokens_now: int | None = None) -> None:
    """The post_llm_call work: ingest, then the enqueue; the worker runs to completion."""
    engine.ingest(view)
    engine.last_prompt_tokens = (
        tokens.count_messages_tokens(view) if tokens_now is None else tokens_now
    )
    engine._prepare_leaves_after_turn(view)
    assert leaf_prep._LEAF_PREP.wait_idle(10)


def _leaf_sources(engine) -> list[list[int]]:
    return [
        node.source_ids
        for node in engine._dag.get_session_nodes("S", limit=100_000)
        if node.depth == 0
    ]


def _frontier(engine) -> int:
    return int(engine._lifecycle.get_by_conversation("conv").current_frontier_store_id)


@pytest.fixture
def engine(tmp_path, provider):
    instance = _engine(tmp_path)
    try:
        yield instance
    finally:
        instance.shutdown()


def _crossing_view(engine) -> list[dict]:
    """A list just over the threshold (T = 45,000)."""
    turns = 1
    while tokens.count_messages_tokens(_view(turns)) <= engine.threshold_tokens:
        turns += 1
    return _view(turns)


# 1. off ------------------------------------------------------------------------------------------------------


def test_setting_defaults_off_and_parses_like_the_other_booleans(monkeypatch):
    assert LCMConfig().background_leaf_prep_enabled is False
    assert (
        LCMConfig().background_leaf_prep_max_calls,
        LCMConfig().background_leaf_prep_window_s,
    ) == (8, 600.0)
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "true")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_MAX_CALLS", "3")
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_WINDOW_S", "60")
    config = LCMConfig.from_env()
    assert config.background_leaf_prep_enabled is True
    assert (
        config.background_leaf_prep_max_calls,
        config.background_leaf_prep_window_s,
    ) == (3, 60.0)


def test_off_starts_no_thread_enqueues_nothing_and_never_promotes(
    tmp_path, provider, monkeypatch, _isolated
):
    engine = _engine(tmp_path, background_leaf_prep_enabled=False)
    monkeypatch.setattr(
        leaf_prep,
        "ready_chain",
        lambda *a: pytest.fail("promotion consulted while off"),
    )
    try:
        view = _crossing_view(engine)
        _turn_end(engine, view)
        assert _isolated._worker is None and _isolated.counters["jobs_enqueued"] == 0
        engine.compress(view, current_tokens=tokens.count_messages_tokens(view))
        assert provider.on("MainThread") >= 1
        assert "background_leaf_prep" not in engine.get_status()
    finally:
        engine.shutdown()


# 2. prepare, then promote --------------------------------------------------------------------------------------


def test_prepared_leaves_publish_at_the_crossing_with_no_model_call(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    _turn_end(engine, view)
    prepared = _isolated.counters["prepared"]
    assert (
        prepared >= 4
        and provider.on("lcm-leaf-prep") == prepared
        and provider.on("MainThread") == 0
    )

    host_tokens = tokens.count_messages_tokens(view)
    result = engine.compress(view, current_tokens=host_tokens)

    assert provider.on("MainThread") == 0  # 0 summariser calls on the turn
    leaves = _leaf_sources(engine)
    assert len(leaves) == _isolated.counters["promoted"] >= 4
    assert engine.last_compression_status == "compacted"
    sweep = engine._last_threshold_full_sweep
    assert sweep["stop_reason"] == "prepared_leaves_promoted" and sweep[
        "promoted_leaves"
    ] == len(leaves)
    low_water = leaf_prep.levels(engine.threshold_tokens, engine.context_length, CHUNK)[
        1
    ]
    freed = sum(
        tokens.count_messages_tokens(view[1 + sum(map(len, leaves[:i])) :][: len(ids)])
        for i, ids in enumerate(leaves)
    )
    assert host_tokens - freed + len(leaves) * tokens.count_tokens(SUMMARY) <= low_water
    assert tokens.count_messages_tokens(result) < engine.threshold_tokens
    assert _frontier(engine) == leaves[-1][-1]
    status = engine.get_status()["background_leaf_prep"]
    assert status["promoted"] == len(leaves) and status["prepared"] == prepared


def _prepared(engine, _isolated, view) -> list:
    _turn_end(engine, view)
    chain = leaf_prep.ready_chain(engine._store.db_path, "conv", _frontier(engine))
    assert len(chain) >= 4
    return chain


# 3. each reject reason -------------------------------------------------------------------------------------------


def test_row_content_update_rejects_with_coverage_digest_mismatch(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    changed = chain[0].covered_store_ids[1]
    engine._store.connection.execute(
        "UPDATE messages SET content = 'rewritten in place' WHERE store_id = ?",
        (changed,),
    )
    engine._store.connection.commit()
    view = [dict(m) for m in view]
    view[changed - 1]["content"] = (
        "rewritten in place"  # store ids follow the view (system prompt = 1)
    )

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    assert _isolated.counters["rejected_coverage_digest_mismatch"] == len(chain)
    assert _isolated.counters["promoted"] == 0
    assert provider.on("MainThread") >= 1  # the foreground path ran instead
    assert all(
        changed not in ids or ids != list(chain[0].covered_store_ids)
        for ids in _leaf_sources(engine)
    )


def test_summary_model_change_rejects_with_route_fingerprint_mismatch(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    engine._config.summary_model = "another-summariser"

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    assert _isolated.counters["rejected_route_fingerprint_mismatch"] == len(chain)
    assert _isolated.counters["promoted"] == 0


def test_foreground_compaction_in_between_rejects_with_frontier_moved(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    engine._config.background_leaf_prep_enabled = False
    result = engine.compress(
        view, current_tokens=tokens.count_messages_tokens(view)
    )  # the foreground wins
    engine._config.background_leaf_prep_enabled = True
    assert _frontier(engine) > 0

    engine.compress(result + view[-2:], current_tokens=engine.threshold_tokens + 1)

    assert _isolated.counters["rejected_frontier_moved"] == len(chain)
    assert _isolated.counters["promoted"] == 0


def test_session_rebind_rejects_with_binding_changed(engine, provider, _isolated):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    engine._session_id = "S-next"  # the leaves were prepared for S
    engine._compress_occurrences = None  # as compress() leaves it without a v4 proof

    leaf_prep_promotion = engine._promote_prepared_leaves(
        view, view, engine.threshold_tokens + 1, engine._new_foreground_budget()
    )

    assert leaf_prep_promotion is None
    assert _isolated.counters["rejected_binding_changed"] == len(chain)


def test_missing_host_row_rejects_with_host_position_gap(engine, provider, _isolated):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    gap = chain[0].covered_store_ids[2] - 1  # the view index of a covered row
    shown = view[:gap] + view[gap + 1 :]

    engine.compress(shown, current_tokens=tokens.count_messages_tokens(shown))

    assert _isolated.counters["rejected_host_position_gap"] == len(chain)
    assert _isolated.counters["promoted"] == 0


def test_cache_bound_evicts_the_least_recently_used_conversation(
    tmp_path, provider, _isolated, monkeypatch
):
    monkeypatch.setattr(leaf_prep, "_PER_PROCESS", 2)
    first = _engine(tmp_path)
    second = _engine(tmp_path / "b", database_path=str(tmp_path / "b.db"))
    try:
        _turn_end(first, _crossing_view(first))
        kept = _isolated.status()["cached_leaves"]
        assert kept >= 2
        _turn_end(second, _crossing_view(second))
        assert _isolated.counters["rejected_stale_evicted"] == kept
        assert leaf_prep.ready_chain(first._store.db_path, "conv", 0) == []
        assert leaf_prep.ready_chain(second._store.db_path, "conv", 0)
    finally:
        first.shutdown()
        second.shutdown()


# 4. no double publication ----------------------------------------------------------------------------------------


def test_rows_already_claimed_are_rejected_by_the_fence_never_published_twice(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    chain = _prepared(engine, _isolated, view)
    claimed = list(chain[0].covered_store_ids)
    # A competing publication claims the first leaf's rows; the lifecycle frontier has not moved yet.
    engine._dag.add_node(
        SummaryNode(
            session_id="S",
            depth=0,
            summary="competing leaf",
            token_count=3,
            source_token_count=100,
            source_ids=claimed,
            source_type="messages",
        )
    )

    engine.compress(view, current_tokens=tokens.count_messages_tokens(view))

    assert _isolated.counters["rejected_fence_conflict"] == len(chain)
    assert _isolated.counters["promoted"] == 0
    for store_id in claimed:
        assert sum(store_id in ids for ids in _leaf_sources(engine)) == 1


def test_promotion_then_foreground_never_covers_a_row_twice(
    engine, provider, _isolated
):
    view = _crossing_view(engine)
    _prepared(engine, _isolated, view)
    result = engine.compress(view, current_tokens=tokens.count_messages_tokens(view))
    longer = result + _view(40)[len(view) :]
    engine._config.background_leaf_prep_enabled = False
    engine.compress(longer, current_tokens=engine.threshold_tokens + 1)

    sources = _leaf_sources(engine)
    flat = [store_id for ids in sources for store_id in ids]
    assert len(flat) == len(set(flat)) and _isolated.counters["promoted"] >= 4


# 5. budget isolation ---------------------------------------------------------------------------------------------


def test_worker_failures_never_open_the_live_breaker_key(engine, provider, _isolated):
    provider.result = RuntimeError("provider down")
    _turn_end(engine, _crossing_view(engine))
    breaker = engine._summary_circuit_breaker
    assert provider.on("lcm-leaf-prep") >= 2
    assert not breaker.allows(leaf_prep.ROUTE_KEY_PREFIX)  # the worker's own key opened
    assert engine._summary_route_available()  # the live route did not
    assert _isolated.counters["prepared"] == 0


def test_worker_skips_while_the_live_route_is_open(engine, provider, _isolated):
    engine._summary_circuit_breaker.record_config_error("")
    view = _crossing_view(engine)
    _turn_end(engine, view)
    assert provider.calls == [] and _isolated.counters["prepared"] == 0
    engine._summary_circuit_breaker.record_success(
        ""
    )  # control: the route closes, the next turn prepares
    _turn_end(engine, view)
    assert _isolated.counters["prepared"] >= 1


@pytest.mark.parametrize("used,prepares", [(2, True), (3, False)])
def test_worker_needs_two_free_foreground_slots_and_never_records_into_them(
    engine, provider, _isolated, used, prepares
):
    engine._summary_spend_guard = SummarySpendGuard(max_calls=4, window_seconds=600.0)
    for _ in range(used):
        engine._summary_spend_guard.record_call()
    _turn_end(engine, _crossing_view(engine))
    assert bool(_isolated.counters["prepared"]) is prepares
    assert len(engine._summary_spend_guard._calls) == used


def test_worker_never_touches_foreground_estimates_or_leaf_provenance(
    engine, provider, _isolated
):
    engine._last_leaf_summary_model, engine._last_leaf_content_filter_error = (
        "sentinel",
        "sentinel",
    )
    engine._last_leaf_level_3_verbatim = "sentinel"
    _turn_end(engine, _crossing_view(engine))
    assert _isolated.counters["prepared"] >= 1
    assert (
        engine._foreground_estimates.calls == {}
        and engine._foreground_estimates.finalize == []
    )
    assert (
        engine._last_leaf_summary_model,
        engine._last_leaf_content_filter_error,
        engine._last_leaf_level_3_verbatim,
    ) == ("sentinel", "sentinel", "sentinel")
    assert engine._foreground_budget is None


def test_worker_has_its_own_spend_guard(engine, provider, _isolated):
    engine._config.background_leaf_prep_max_calls = 2
    _turn_end(engine, _crossing_view(engine))
    assert (
        provider.on("lcm-leaf-prep") == 2 and _isolated.counters["prepared"] == 2
    )  # then the guard is spent
    assert (
        engine._summary_spend_guard.remaining()
        == engine._config.summary_spend_max_calls
    )


def test_host_single_aux_slot_turns_the_worker_off(
    tmp_path, engine, provider, _isolated
):
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    max_concurrency: 1\n"
    )
    view = _crossing_view(engine)
    _turn_end(engine, view)
    assert provider.calls == [] and _isolated.counters["prepared"] == 0
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    max_concurrency: 2\n"
    )  # control
    _turn_end(engine, view)
    assert _isolated.counters["prepared"] >= 1


# 6. concurrency --------------------------------------------------------------------------------------------------


def _foreground_run(tmp_path, view):
    control = _engine(
        tmp_path / "control",
        database_path=str(tmp_path / "control.db"),
        background_leaf_prep_enabled=False,
    )
    try:
        control.ingest(view)
        result = control.compress(
            view, current_tokens=tokens.count_messages_tokens(view)
        )
        return (
            result,
            _leaf_sources(control),
            control._last_compress_leaves,
            dict(control._foreground_estimates.calls),
        )
    finally:
        control.shutdown()


def test_worker_during_foreground_compress_changes_neither_result_nor_budget(
    tmp_path, engine, provider, _isolated
):
    view = _crossing_view(engine)
    control = _foreground_run(tmp_path, view)
    started, release = threading.Event(), threading.Event()

    def hold_worker():
        if threading.current_thread().name == "lcm-leaf-prep":
            started.set()
            assert release.wait(10)

    provider.hook = hold_worker
    engine.ingest(view)
    engine.last_prompt_tokens = tokens.count_messages_tokens(view)
    engine._prepare_leaves_after_turn(view)
    assert started.wait(10)
    try:
        result = engine.compress(
            view, current_tokens=tokens.count_messages_tokens(view)
        )
    finally:
        release.set()
    assert _isolated.wait_idle(10)

    assert result == control[0] and _leaf_sources(engine) == control[1]
    assert engine._last_compress_leaves == control[2]
    assert {
        key: len(value) for key, value in engine._foreground_estimates.calls.items()
    } == {key: len(value) for key, value in control[3].items()}
    assert (
        _isolated.counters["prepared"] == 0
    )  # its frontier moved during the call: discarded


def test_at_most_one_worker_model_call_in_flight(engine, provider, _isolated):
    assert _isolated._call_slot.acquire(blocking=False)
    view = _crossing_view(engine)
    try:
        _turn_end(engine, view)
    finally:
        _isolated._call_slot.release()
    assert provider.calls == [] and _isolated.counters["prepared"] == 0
    _turn_end(engine, view)  # control: the slot is free
    assert _isolated.counters["prepared"] >= 1


# 7. context ------------------------------------------------------------------------------------------------------

_SCOPE = contextvars.ContextVar("lcm_787_scope", default="unscoped")


def test_worker_job_runs_in_the_enqueue_time_context(engine, provider, _isolated):
    seen = []
    provider.hook = lambda: seen.append((threading.current_thread().name, _SCOPE.get()))
    token = _SCOPE.set("profile-a")
    try:
        _turn_end(engine, _crossing_view(engine))
    finally:
        _SCOPE.reset(token)
    assert seen and set(seen) == {("lcm-leaf-prep", "profile-a")}


def test_post_llm_call_hook_enqueues_after_ingest_and_skips_interrupted_turns(
    tmp_path, monkeypatch, provider, _isolated
):
    monkeypatch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "1")
    _install_host(monkeypatch, tmp_path)
    module = _load_plugin("hermes_lcm_787_hook")
    jobs = []
    hook_prep = importlib.import_module(
        f"{module.__name__}.leaf_prep"
    )  # this load's own module state
    monkeypatch.setattr(hook_prep._LEAF_PREP, "enqueue", jobs.append)
    ctx = _Ctx(_Manager())
    module.register(ctx)
    engine = ctx.offered
    try:
        engine.on_session_start(
            "S", platform="telegram", conversation_id="conv", context_length=WINDOW
        )
        history = _crossing_view(engine)
        engine.last_prompt_tokens = tokens.count_messages_tokens(history)
        hook = ctx.hooks["post_llm_call"][0]
        hook(
            session_id="S",
            conversation_id="conv",
            platform="telegram",
            conversation_history=history,
            interrupted=True,
        )
        assert jobs == []
        hook(
            session_id="S",
            conversation_id="conv",
            platform="telegram",
            conversation_history=history,
        )
        assert [
            (job.session_id, job.conversation_id, job.frontier) for job in jobs
        ] == [("S", "conv", 0)]
        assert engine._store.get_session_count("S") == len(history)
    finally:
        engine.shutdown()


# 8. lifetime -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ending", ["shutdown", "rebind"])
def test_shutdown_and_rebind_clear_the_cache_and_discard_a_late_job(
    tmp_path, provider, _isolated, ending
):
    engine = _engine(tmp_path)
    view = _crossing_view(engine)
    _turn_end(engine, view)
    assert _isolated.status()["cached_leaves"] >= 4
    started, release = threading.Event(), threading.Event()

    def hold_worker():
        started.set()
        assert release.wait(10)

    provider.hook = hold_worker
    engine.ingest(
        view + _view(60)[len(view) :]
    )  # more rows: the next job prepares further
    engine.last_prompt_tokens = WINDOW  # far above L: the chain must grow
    engine._prepare_leaves_after_turn(view)
    assert started.wait(10)
    try:
        if ending == "shutdown":
            engine.shutdown()
        else:
            engine._rebind_storage_for_home(str(tmp_path / "other-home"))
    finally:
        release.set()
    assert _isolated.wait_idle(10)
    status = _isolated.status()
    assert status["cached_leaves"] == 0 and status["queued_jobs"] == 0
    if ending == "rebind":
        engine.shutdown()


# 9. levels -------------------------------------------------------------------------------------------------------


def test_levels_turn_off_on_small_windows_and_low_thresholds():
    assert leaf_prep.levels(45_000, 60_000, 2_000) == (36_000, 39_000)
    assert leaf_prep.levels(24_000, 32_000, 24_000) is None  # L <= 0 on a 32K window
    assert (
        leaf_prep.levels(24_000, 60_000, 2_000) is None
    )  # P = 0.25 x window <= 0.30 x window
    assert leaf_prep.levels(204_000, 272_000, 24_000) == (163_200, 156_000)


def test_small_window_prepares_nothing(tmp_path, provider, _isolated):
    engine = _engine(tmp_path, window=32_000, leaf_chunk_tokens=24_000)
    control = _engine(
        tmp_path / "c", database_path=str(tmp_path / "c.db"), leaf_chunk_tokens=CHUNK
    )
    try:
        _turn_end(engine, _crossing_view(engine))
        assert provider.calls == [] and _isolated.counters["prepared"] == 0
        _turn_end(
            control, _crossing_view(control)
        )  # control: the 60K window with a 2K chunk prepares
        assert _isolated.counters["prepared"] >= 1
    finally:
        engine.shutdown()
        control.shutdown()


def test_never_prepares_inside_the_fresh_tail(tmp_path, provider, _isolated):
    engine = _engine(tmp_path, fresh_tail_count=20)
    try:
        view = _crossing_view(engine)
        _turn_end(engine, view)
        chain = leaf_prep.ready_chain(engine._store.db_path, "conv", 0)
        assert chain
        tail_ids = set(
            range(len(view) - 20 + 1, len(view) + 1)
        )  # store ids follow the view
        assert not tail_ids & {
            store_id for leaf in chain for store_id in leaf.covered_store_ids
        }
    finally:
        engine.shutdown()


def test_sweep_chunk_matches_the_foreground_choice(engine):
    for raw in (0, 5_000, 50_000):
        assert leaf_prep.working_leaf_chunk_tokens(
            engine._config, engine.context_length, raw
        ) == engine._working_leaf_chunk_tokens(raw)
    engine._config.dynamic_leaf_chunk_enabled = True
    for raw in (0, 5_000, 50_000):
        assert leaf_prep.working_leaf_chunk_tokens(
            engine._config, engine.context_length, raw
        ) == engine._working_leaf_chunk_tokens(raw)


def test_spend_guard_remaining_records_nothing():
    guard = SummarySpendGuard(max_calls=3, window_seconds=600.0, backoff_seconds=60.0)
    assert guard.remaining() == 3
    guard.record_call()
    assert guard.remaining() == 2 and len(guard._calls) == 1
    guard.record_call()
    guard.record_call()  # fills the window: backoff
    assert guard.remaining() == 0
    assert SummarySpendGuard(max_calls=0).remaining() == float("inf")


def test_preparation_serializer_writes_no_payload_file(tmp_path):
    from hermes_lcm.engine import LeafPrepUnavailable, _serialize_leaf_input

    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=100,
        large_output_externalization_path=str(tmp_path / "payloads"),
    )
    rows = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "large tool output " * 40},
    ]

    def files():
        return (
            sorted(p.name for p in (tmp_path / "payloads").glob("*.json"))
            if (tmp_path / "payloads").exists()
            else []
        )

    with pytest.raises(LeafPrepUnavailable):
        _serialize_leaf_input(
            rows,
            config=config,
            hermes_home=str(tmp_path),
            session_id="S",
            externalize_write=False,
        )
    assert files() == []
    text, _clipped = _serialize_leaf_input(
        rows, config=config, hermes_home=str(tmp_path), session_id="S"
    )
    written = files()
    assert len(written) == 1  # the foreground writes the payload, as before
    again, _clipped = _serialize_leaf_input(
        rows,
        config=config,
        hermes_home=str(tmp_path),
        session_id="S",
        externalize_write=False,
    )
    assert (
        again == text and files() == written
    )  # on disk now: the same input, no new file


def test_a_newer_turn_supersedes_the_queued_job_of_its_conversation(
    engine, provider, _isolated
):
    started, release = threading.Event(), threading.Event()

    def hold_worker():
        started.set()
        assert release.wait(10)

    provider.hook = hold_worker
    other = _engine(
        engine._store.db_path.parent / "o",
        database_path=str(engine._store.db_path.parent / "o.db"),
    )
    try:
        _view_other = _crossing_view(other)
        other.ingest(_view_other)
        other.last_prompt_tokens = tokens.count_messages_tokens(_view_other)
        other._prepare_leaves_after_turn(_view_other)  # occupies the worker
        assert started.wait(10)
        view = _crossing_view(engine)
        engine.ingest(view)
        engine.last_prompt_tokens = tokens.count_messages_tokens(view)
        engine._prepare_leaves_after_turn(view)
        engine._prepare_leaves_after_turn(view)
        assert (
            _isolated.status()["queued_jobs"] == 1
            and _isolated.counters["jobs_superseded"] == 1
        )
    finally:
        release.set()
        assert _isolated.wait_idle(10)
        other.shutdown()
