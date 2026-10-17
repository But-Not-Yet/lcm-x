"""Keep late auxiliary detection reachable through the compression entry helper."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

harness = importlib.import_module("tests.harness")

# FRAMEDEPTH G01 (also G04-G16): the generation walk found at most one
# outside frame on main. Keep that many intervening host frames as well
# as the auxiliary caller itself; do not increase the walker's 32-frame budget.
OUTSIDE_FRAMES = 1


def _host_frames(engine: Any, view: list[dict[str, Any]], remaining: int) -> Any:
    if remaining > 1:
        return _host_frames(engine, view, remaining - 1)
    return engine._compress_impl(view)


class _AuxiliaryAgent:
    session_id = "review-session"
    _memory_write_origin = "background_review"

    def compress(self, engine: Any, view: list[dict[str, Any]]) -> Any:
        return _host_frames(engine, view, OUTSIDE_FRAMES)


def test_begin_compression_preserves_late_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path, window=200_000, session_id="review-session") as engine:
        view = [harness.user("auxiliary replay must stay out of durable history")]
        # compress() normally catches an already-marked caller before this
        # helper. Enter the real inner path to exercise its late-marker recheck
        # and its subsequent _current_auxiliary_prompt_tokens generation walk.
        result = _AuxiliaryAgent().compress(engine, view)
        assert result == view
        assert engine._thread_context_has_auxiliary_session("review-session")
        assert engine._store.get_session_count("review-session") == 0
        assert engine.last_compression_noop_reason == "LCM bypassed below threshold: auxiliary thread context"
