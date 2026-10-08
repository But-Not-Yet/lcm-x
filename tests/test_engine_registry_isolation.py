"""Lock the current isolation of the process-wide resident-engine shared surface.

Background: the "circuit contamination across sessions" story was refuted on
re-review (C2: the parallel failure did not reproduce; engines register per
session and are not shared across processes). The planned per-session-isolation
work was withdrawn, and these tests are the replacement lock: they pin the
*current* isolation guarantees of the shared registries and resolvers
(``engine_registry.py``) so any future cross-session leak fails immediately.

What is locked:
- two live engines in one process resolve strictly to their own session/lane;
- ingest through ``use_active_lcm_engine`` lands only in the selected engine's
  store, never in the sibling's;
- rebinding one engine never moves or shadows the other's registrations;
- a cold prototype cannot claim while any session engine is resident
  (``use_cold_lcm_engine`` -> RESIDENT_ENGINE_CONFLICT);
- the weak registries drop entries when the engines are unreferenced, so no
  stale engine stays resolvable after its last reference is gone.
"""

from __future__ import annotations

import gc

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.engine_registry import (
    ActiveEngineUseStatus,
    has_resident_lcm_engine,
    resolve_active_lcm_engine,
    use_active_lcm_engine,
    use_cold_lcm_engine,
)


def _engine(tmp_path, tag: str) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / f"iso-{tag}.db"))
    return LCMEngine(config=config)


@pytest.fixture
def pair(tmp_path):
    """Two live engines, each bound to its own session and conversation lane."""
    engine_a = _engine(tmp_path, "a")
    engine_b = _engine(tmp_path, "b")
    engine_a.on_session_start("iso-session-a", platform="cli", conversation_id="iso-conv-a")
    engine_b.on_session_start("iso-session-b", platform="cli", conversation_id="iso-conv-b")
    try:
        yield engine_a, engine_b
    finally:
        engine_a.shutdown()
        engine_b.shutdown()


def test_each_session_resolves_only_to_its_own_engine(pair):
    engine_a, engine_b = pair
    assert resolve_active_lcm_engine(session_id="iso-session-a") is engine_a
    assert resolve_active_lcm_engine(session_id="iso-session-b") is engine_b
    assert resolve_active_lcm_engine(conversation_id="iso-conv-a") is engine_a
    assert resolve_active_lcm_engine(conversation_id="iso-conv-b") is engine_b
    # A lane id that belongs to neither registration is not answered by the
    # sibling engine just because it is resident in the same process.
    assert resolve_active_lcm_engine(session_id="iso-session-c") is None
    assert resolve_active_lcm_engine(conversation_id="iso-conv-c") is None


def test_ingest_through_active_use_never_touches_the_sibling_store(pair):
    engine_a, engine_b = pair

    result_a = use_active_lcm_engine(
        lambda selected: selected.ingest(
            [{"role": "user", "content": "isolation payload for A"}]
        ),
        session_id="iso-session-a",
        conversation_id="iso-conv-a",
    )
    assert result_a.status is ActiveEngineUseStatus.USED

    rows_a = engine_a._store.search("isolation payload")
    assert len(rows_a) == 1
    assert rows_a[0]["session_id"] == "iso-session-a"
    # The sibling's own store saw nothing: no cross-engine ingest bleed.
    assert engine_b._store.search("isolation payload") == []

    result_b = use_active_lcm_engine(
        lambda selected: selected.ingest(
            [{"role": "user", "content": "isolation payload for B"}]
        ),
        session_id="iso-session-b",
        conversation_id="iso-conv-b",
    )
    assert result_b.status is ActiveEngineUseStatus.USED
    rows_b = engine_b._store.search("isolation payload")
    assert len(rows_b) == 1
    assert rows_b[0]["session_id"] == "iso-session-b"
    assert engine_a._store.search("isolation payload for B") == []


def test_rebinding_one_engine_leaves_the_other_registration_untouched(pair):
    engine_a, engine_b = pair
    engine_a.on_session_start(
        "iso-session-a2", platform="cli", conversation_id="iso-conv-a2"
    )
    # The rebound engine follows its new lane only.
    assert resolve_active_lcm_engine(session_id="iso-session-a2") is engine_a
    assert resolve_active_lcm_engine(session_id="iso-session-a") is None
    stale = use_active_lcm_engine(
        lambda selected: selected,
        session_id="iso-session-a",
        conversation_id="iso-conv-a",
    )
    assert stale.status is ActiveEngineUseStatus.ENGINE_NOT_RESIDENT
    # The sibling keeps its own registrations and its own engine.
    assert resolve_active_lcm_engine(session_id="iso-session-b") is engine_b
    assert resolve_active_lcm_engine(conversation_id="iso-conv-b") is engine_b


def test_cold_engine_cannot_claim_while_any_session_engine_is_resident(pair):
    cold = None
    try:
        cold = LCMEngine(config=LCMConfig(database_path=":memory:"))
        conflict = use_cold_lcm_engine(cold, lambda selected: "ran")
        assert conflict.status is ActiveEngineUseStatus.RESIDENT_ENGINE_CONFLICT
    finally:
        if cold is not None:
            cold.shutdown()


def test_cold_engine_claims_only_after_every_resident_engine_is_gone(tmp_path):
    cold = None
    try:
        cold = LCMEngine(config=LCMConfig(database_path=":memory:"))
        assert has_resident_lcm_engine() is False
        claimed = use_cold_lcm_engine(cold, lambda selected: "ran")
        assert claimed.status is ActiveEngineUseStatus.USED
        assert claimed.value == "ran"
    finally:
        if cold is not None:
            cold.shutdown()


def test_weak_registries_drop_unreferenced_engines(tmp_path):
    engine = _engine(tmp_path, "weak")
    engine.on_session_start("iso-session-weak", conversation_id="iso-conv-weak")
    try:
        assert resolve_active_lcm_engine(session_id="iso-session-weak") is engine
    finally:
        engine.shutdown()
        del engine
        gc.collect()
    assert resolve_active_lcm_engine(session_id="iso-session-weak") is None
    assert resolve_active_lcm_engine(conversation_id="iso-conv-weak") is None
