"""I1: a proven rotation replay survives a carried host stamp, without losing fresh repeats."""
from __future__ import annotations

import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.harness import engine_context


def message(text: str, stamp: float | None, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": text, "timestamp": stamp}


def lineage(root: Path, verified: bool = True) -> None:
    home = root / "home"
    home.mkdir(exist_ok=True)
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                       [("parent", None, "compression" if verified else "normal"), ("child", "parent", None)])


def test_rotation_replay_then_fresh_repeat(tmp_path: Path) -> None:
    """Real ingest and cold restart: the stored mis-stamp stays, the replay does not duplicate it."""
    lineage(tmp_path)
    previous, current = message("interrupted turn", 10.), message("next request", 20.)
    reply = message("reply", 21., "assistant")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([previous])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv",
                                boundary_reason="compression")
        engine.ingest([{**current, "timestamp": 10.}, reply])
        before = engine._store.get_session_messages("child")
        assert [r["observed_at"] for r in before] == [10., 21.]
    with engine_context(tmp_path, window=128000, session_id="child", conversation_id="conv") as engine:
        replay = [previous, current, reply]
        engine.ingest(deepcopy(replay))
        assert engine._store.get_session_messages("child") == before
        fresh = message("next request", 30.)
        engine.ingest([*deepcopy(replay), fresh])
        rows = engine._store.get_session_messages("child")
        assert [r["content"] for r in rows] == ["next request", "reply", "next request"]
        assert [r["observed_at"] for r in rows] == [10., 21., 30.]
        engine.ingest([*deepcopy(replay), fresh])
        assert engine._store.get_session_messages("child") == rows


@pytest.mark.parametrize("guard", ["accepted", "ancestry", "gap", "previous-unproven", "null",
                                   "equal-stamp", "consumed", "content", "uid", "consecutive"])
def test_carry_tolerance_requires_every_proof(tmp_path: Path, guard: str) -> None:
    """Exercise the audit plan on real storage; each missing proof stores the uncertain occurrence."""
    lineage(tmp_path, guard != "ancestry")
    previous = message("previous", 10.)
    current = message("current", 20.)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([previous])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        stored = [message("gap", 11., "assistant")] if guard == "gap" else []
        stored += [message("current", None if guard == "null" else 10.)]
        if guard == "consecutive":
            stored += [message("third", 20.)]
        engine.ingest(stored)
        if guard == "previous-unproven":
            previous = message("previous", 9.)
        if guard == "equal-stamp":
            current = message("current", 10.)
        if guard == "content":
            current = message("different", 20.)
        if guard == "uid":
            current["message_uid"] = "current-occurrence"
        view = [previous, current]
        if guard == "consumed":
            view = [previous, message("current", 10.), current]
        if guard == "consecutive":
            view += [message("third", 30.)]
        plan = engine._identity_anchor_prematch(view, view, len(view), 0)
        if guard in {"accepted", "equal-stamp"}:
            assert plan["cursor"] == len(view)
        elif guard == "consecutive":
            assert plan["cursor"] == 2  # the tolerated second row cannot prove the third
        else:
            assert plan["cursor"] <= len(view) - 1
        matched = [int(row["store_id"]) for rows in plan.get("matched", {}).values() for row in rows]
        assert len(matched) == len(set(matched))


def test_fresh_repeat_immediately_after_same_text(tmp_path: Path) -> None:
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([message("same", 10.)])
        engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        engine.ingest([message("same", 10.), message("same", 20.)])
        assert [r["observed_at"] for r in engine._store.get_session_messages("child")] == [20.]
