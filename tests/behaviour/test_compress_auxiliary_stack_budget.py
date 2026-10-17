"""Keep late auxiliary detection reachable through the compression entry helper."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

harness = importlib.import_module("tests.harness")

# REALDEPTH C-r34.10-0.21.5-1405-delegate-G-1 has the maximum six host
# frames between plugin entry and the auxiliary frame (seven including it).
# Add one for public compress(), skipped by entering _compress_impl below;
# keep the auxiliary frame itself separate and the 32-frame budget unchanged.
OUTSIDE_FRAMES = 6 + 1


def _host_frames(engine: Any, view: list[dict[str, Any]], remaining: int) -> Any:
    if remaining > 1:
        return _host_frames(engine, view, remaining - 1)
    result = engine._compress_impl(view)
    assert engine._in_process_auxiliary_caller_generation("review-session") > 0
    assert engine._thread_context_session_id() == "review-session"
    assert engine.last_compression_noop_reason == "LCM bypassed below threshold: auxiliary thread context"
    return result


class _AuxiliaryAgent:
    session_id = "review-session"
    _parent_session_id = "foreground-session"
    _memory_write_origin = "background_review"

    def compress(self, engine: Any, view: list[dict[str, Any]]) -> Any:
        return _host_frames(engine, view, OUTSIDE_FRAMES)


@pytest.mark.parametrize("bound_session", ["review-session", "foreground-session"])
def test_begin_compression_preserves_late_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_session: str,
) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path, window=200_000, session_id=bound_session) as engine:
        view = [harness.user("auxiliary replay must stay out of durable history")]
        # compress() normally catches an already-marked caller before this
        # helper. Enter the real inner path to exercise its late-marker recheck
        # and its subsequent _current_auxiliary_prompt_tokens generation walk.
        # The review binding requires late generation-based registration;
        # the foreground binding matches the session-id walk via the parent.
        result = _AuxiliaryAgent().compress(engine, view)
        assert result == view
        assert engine._thread_context_has_auxiliary_session("review-session") == (
            bound_session == "review-session"
        )
        assert engine._store.get_session_count("review-session") == 0
        assert engine._store.get_session_count("foreground-session") == 0


class _LateAuxiliaryAgent:
    session_id = "review-session"
    _parent_session_id = "foreground-session"
    _memory_write_origin = ""

    def compress(self, engine: Any, view: list[dict[str, Any]], current_tokens: int) -> Any:
        return _phase_host_frames(engine, view, current_tokens, OUTSIDE_FRAMES)


def _phase_host_frames(engine: Any, view: list[dict[str, Any]], current_tokens: int, remaining: int) -> Any:
    if remaining > 1:
        return _phase_host_frames(engine, view, current_tokens, remaining - 1)
    return engine._compress_impl(view, current_tokens=current_tokens, force=False)


@pytest.mark.parametrize("phase", ["prepare", "condense"])
def test_sweep_phase_host_callbacks_preserve_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    """Keep the real path to each host boundary, including summary route wrappers.

    An already-marked child exits in P0. A host callback can mark its caller
    late; check both real walkers there, without replacing an engine method.
    The estimator and auxiliary-client response are the only host fakes.
    """
    import inspect
    import json
    import re
    import sys
    from types import ModuleType, SimpleNamespace

    from tests.behaviour.compress_trace_harness import PAD, SCENARIOS, dag, transcript

    survival = importlib.import_module("hermes_lcm.survival_fit")
    host_estimate = survival._host_estimate
    harness.pin_char_counter(monkeypatch)
    monkeypatch.setattr(survival, "_host_estimate", host_estimate)
    case = next(case for case in SCENARIOS if case.name == (
        "sweep-stub-first-exit" if phase == "prepare" else "sweep-pre-leaf-condensation"))
    settings = dict(fresh_tail_count=2, fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                    context_threshold=0.001, threshold_full_sweep_enabled=True,
                    max_assembly_tokens=100000, summary_model="trace-model", l3_truncate_tokens=2,
                    temporal_rollups_enabled=False, embeddings_enabled=False,
                    large_output_externalization_enabled=False,
                    large_output_active_replay_stubbing_enabled=False,
                    large_output_externalization_path=str(tmp_path / "payloads"))
    settings.update(dict(case.overrides))
    agent = _LateAuxiliaryAgent()
    boundary_parent = "_stub_first_exit" if phase == "prepare" else "_run_threshold_sweep_condensation"
    distances: dict[str, int] = {}
    answers: list[tuple[str, int]] = []
    paths: dict[str, list[str]] = {}
    walkers = {"_in_process_auxiliary_session_id_from_stack", "_in_process_auxiliary_caller_generation"}

    def profile(frame: Any, event: str, value: Any) -> None:
        if event != "return" or frame.f_code.co_name not in walkers:
            return
        chain, current = [], frame.f_back
        while current is not None:
            chain.append(current.f_code.co_name)
            if current.f_locals.get("self") is agent:
                break
            current = current.f_back
        if boundary_parent in chain and agent._memory_write_origin:
            assert current is not None
            name = frame.f_code.co_name
            if len(chain) > distances.get(name, 0):
                distances[name], paths[name] = len(chain), chain
            assert value == "review-session" if name.endswith("from_stack") else value > 0

    with harness.engine_context(tmp_path, window=200000, session_id="foreground-session", **settings) as engine:
        def boundary() -> None:
            frame = inspect.currentframe()
            chain = []
            while frame is not None:
                chain.append(frame.f_code.co_name)
                frame = frame.f_back
            if boundary_parent not in chain:
                return
            agent._memory_write_origin = "background_review"
            try:
                answers.append((engine._thread_context_session_id(),
                                engine._in_process_auxiliary_caller_generation("review-session")))
            finally:
                agent._memory_write_origin = ""

        def estimate(messages: Any) -> int:
            boundary()
            return 0  # real local counter still determines the measured size

        def call_llm(**kwargs: Any) -> Any:
            boundary()
            nonce = re.search(r'<lcm-summary nonce="([a-f0-9]+)">', kwargs["messages"][0]["content"])
            assert nonce is not None
            content = (f'<lcm-summary nonce="{nonce[1]}">\n'
                       'Earlier turns recorded source evidence and preserved the pending request.\n'
                       'Expand for details about: turns\n</lcm-summary>')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        metadata, auxiliary = ModuleType("agent.model_metadata"), ModuleType("agent.auxiliary_client")
        metadata.estimate_messages_tokens_rough = estimate
        auxiliary.call_llm = call_llm
        monkeypatch.setitem(sys.modules, metadata.__name__, metadata)
        monkeypatch.setitem(sys.modules, auxiliary.__name__, auxiliary)
        for i, depth in enumerate(case.seeded_depths):
            engine._dag.add_node(dag.SummaryNode(
                session_id="foreground-session", depth=depth, summary=f"Seed {i}. " + PAD,
                token_count=500, source_token_count=1000, source_ids=[],
                source_type="messages", created_at=100.0 + i))
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            agent.compress(engine, transcript(case), int(case.current_tokens or 0))
        finally:
            sys.setprofile(previous)
        assert answers and all(session == "review-session" and generation > 0 for session, generation in answers)
        assert set(distances) == walkers
        assert max(distances.values()) < 28, "keep calls within four frames of the limit in their original frame"
        print(json.dumps(dict(phase=phase, distances=distances, paths=paths), sort_keys=True))
