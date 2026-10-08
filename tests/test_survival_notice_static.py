"""The survival-fit notice in the system slot is a fixed template: its bytes never change between fits.

The system row sits at prompt offset ~0. A notice whose text varies per fit (message count, store-id
range) rewrites the shared prefix every round and wipes the provider's prefix cache (delta~0 risk).
The fitted system row must be byte-identical across consecutive fits; the per-fit detail (count, store
ids) stays in the log record and the ``_last_survival_fit`` metadata, never in the prompt.
"""

from __future__ import annotations

import sys

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import _NOTICE, _NOTICE_STATIC

PAD = " alpha beta gamma delta epsilon zeta" * 30


def _engine(tmp_path, window=800) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1600, context_threshold=0.75,
                                        database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _fit(engine, system, rows):
    """One direct fit over ``[system, *rows]``; returns the fitted list."""
    active = [system, *rows]
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = 0, True
    return engine._survival_fit(active, list(active), 0, "test", after_exception=True)


@pytest.fixture
def no_host(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.model_metadata", None)  # LCM's own character-based count


def test_two_fits_with_different_cuts_emit_byte_identical_notices(tmp_path, no_host):
    """The card's judgement: consecutive fits whose dynamic data (count, store ids) differs must still
    produce a byte-identical system row, so the assembled prompt's survival-fit line never diffs."""
    engine = _engine(tmp_path)
    try:
        system = {"role": "system", "content": "system prompt"}
        rows12 = [r for i in range(12) for r in _turn(f"T{i:03d}", 10.0 * i)]
        engine.ingest([system] + rows12)
        result1 = _fit(engine, system, rows12)
        assert engine._last_survival_fit is not None  # the fit fired
        record1 = engine._last_survival_fit["notice"]

        rows20 = [r for i in range(20) for r in _turn(f"U{i:03d}", 1000.0 * i)]
        engine.ingest([system] + rows20)
        result2 = _fit(engine, result1[0], rows20)  # the host replays the fitted system row
        assert engine._last_survival_fit is not None
        record2 = engine._last_survival_fit["notice"]

        assert record1 != record2  # the dynamic detail really differs between the two fits
        assert result1[0]["content"] == result2[0]["content"]  # the prompt line is byte-identical
        assert str(result1[0]["content"]).count("[LCM survival fit: ") == 1  # one notice, the newest
        assert str(result1[0]["content"]).endswith(_NOTICE_STATIC)
    finally:
        engine.shutdown()


def test_an_old_dynamic_notice_is_replaced_by_the_static_template(tmp_path, no_host):
    """A system row still carrying a pre-static fit's dynamic notice loses it: exactly one notice, the
    fixed template, remains."""
    engine = _engine(tmp_path)
    try:
        old = {"role": "system", "content": f"system prompt\n\n{_NOTICE.format(n=5, first=1, last=5)}"}
        rows = [r for i in range(12) for r in _turn(f"T{i:03d}", 10.0 * i)]
        engine.ingest([old] + rows)
        result = _fit(engine, old, rows)
        content = str(result[0]["content"])
        assert content.count("[LCM survival fit: ") == 1
        assert content.endswith(_NOTICE_STATIC)
        assert "(store ids" not in content  # the dynamic form left the prompt
        assert "store ids" in engine._last_survival_fit["notice"]  # and lives on in the record
    finally:
        engine.shutdown()


def test_list_content_system_row_is_byte_stable(tmp_path, no_host):
    """The list-content branch of ``_survival_with_notice``: the appended text part is the static
    template on every fit, whatever the cut."""
    engine = _engine(tmp_path)
    try:
        system = {"role": "system", "content": [{"type": "text", "text": "system prompt"}]}
        rows12 = [r for i in range(12) for r in _turn(f"T{i:03d}", 10.0 * i)]
        engine.ingest([system] + rows12)
        result1 = _fit(engine, system, rows12)
        rows20 = [r for i in range(20) for r in _turn(f"U{i:03d}", 1000.0 * i)]
        engine.ingest([system] + rows20)
        result2 = _fit(engine, result1[0], rows20)
        assert result1[0]["content"] == result2[0]["content"]
        assert result1[0]["content"][-1] == {"type": "text", "text": _NOTICE_STATIC}
    finally:
        engine.shutdown()
