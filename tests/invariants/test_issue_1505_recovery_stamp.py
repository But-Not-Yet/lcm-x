"""I1: a live recovery-row rewrite explains one stored turn, never a fresh occurrence."""
from __future__ import annotations

import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.harness import ScriptedSummarizer, engine_context, pin_char_counter


def message(content: Any, stamp: float | None, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": content, "timestamp": stamp}


def lineage(root: Path, verified: bool = True) -> None:
    home = root / "home"
    home.mkdir(exist_ok=True)
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)",
                       [("parent", None, "compression" if verified else "normal"), ("child", "parent", None)])


def recover(engine: Any, end: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drive the real forced assembly/finalization and its publication, with no internal mocks."""
    current = message("[T32] rebuild the index " * 500, 20.)
    source = [message("[T31] earlier request", 10.), message("earlier reply", 11., "assistant"), current,
              {"role": "tool", "content": "orphan result", "tool_call_id": "orphan", "timestamp": 21.}]
    engine.ingest(source)
    result = engine._assemble_overflow_recovery_context(None, source, assembly_cap_override=120)
    result = engine._finalize_forced_overflow_result(source, result, assembly_cap_override=120)
    engine._record_compress_commit_proof(source, result)
    assert engine._last_compression_status == "overflow_recovery"
    assert result[-1]["content"].startswith("[LCM overflow recovery] Your latest message")
    assert any(d["kind"] == "recovery" for d in engine._compress_commit_proof["emissions"])
    if end:
        engine.on_session_end("parent", source)
    return result, current


def users(engine: Any) -> list[dict[str, Any]]:
    return [r for session in ("parent", "child") for r in engine._store.get_session_messages(session)
            if r["role"] == "user"]


@pytest.mark.parametrize("rotation", [False, True], ids=["in-place", "rotation"])
def test_live_recovery_then_fresh_repeat(tmp_path: Path, rotation: bool) -> None:
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        recovered, current = recover(engine, end=rotation)
        if rotation:
            engine.on_session_start("child", platform="cli", context_length=128000,
                                    conversation_id="conv", boundary_reason="compression", old_session_id="parent")
        recovered[-1]["timestamp"] = 30.
        # The host passes copies on its intermediate compress calls, retaining the original live row.
        engine.ingest(deepcopy(recovered))
        recovered[-1]["content"] = current["content"]
        view = [*recovered, message("reply", 31., "assistant")]
        engine.ingest(view)
        copies = [r for r in users(engine) if r["content"] == current["content"]]
        assert len(copies) == 1, [(r["store_id"], r["observed_at"]) for r in copies]
        assert copies[0]["observed_at"] == 20.
        fresh = message(current["content"], 40.)
        engine.ingest([*view, fresh])
        copies = [r for r in users(engine) if r["content"] == current["content"]]
        assert [r["observed_at"] for r in copies] == [20., 40.]


@pytest.mark.parametrize("guard", ["fresh-object", "nonadjacent", "null-copy", "changed-stamp",
                                   "wrong-role", "occurrence-uid", "new-user-after-copy", "unverified-scope",
                                   "two-recoveries", "consumed", "copy-alias", "recovery-alias"])
def test_missing_recovery_proof_stores_row(tmp_path: Path, guard: str) -> None:
    lineage(tmp_path, guard != "unverified-scope")
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        recovered, current = recover(engine)
        if guard == "two-recoveries":
            recovered, current = recover(engine)
        if guard == "unverified-scope":
            engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv")
        recovered[-1]["timestamp"] = 30.
        engine.ingest(deepcopy(recovered))
        copy = next(r for r in users(engine) if r["content"] == current["content"])
        if guard == "null-copy":
            engine._store._conn.execute("UPDATE messages SET observed_at=NULL WHERE store_id=?", (copy["store_id"],))
            engine._store._conn.commit()
        if guard == "copy-alias":
            engine._store.add_message_relations([[(copy["store_id"], "alt_stamp", None, None, 25.)]])
            # An alias alone does not identify a new, separately allocated host row.
            recovered[-1] = dict(recovered[-1])
        if guard == "recovery-alias":
            # The saved own stamp was 30; no alias or later timestamp can replace that proof.
            recovered[-1]["timestamp"] = 35.
        if guard == "fresh-object":
            recovered[-1] = dict(recovered[-1])
        if guard == "nonadjacent":
            recovered.insert(-1, message("inserted reply", 29., "assistant"))
        if guard == "changed-stamp":
            recovered[-1]["timestamp"] = 40.
        if guard == "occurrence-uid":
            recovered[-1]["message_uid"] = "a-new-occurrence"
        if guard == "new-user-after-copy":
            engine.ingest([*recovered, message("new request", 29.)])
        if guard == "wrong-role":
            recovered[-1]["role"] = "assistant"
        recovered[-1]["content"] = current["content"]
        if guard == "consumed":
            recovered.insert(0, dict(current))
        engine.ingest([*recovered, message("reply", 41., "assistant")])
        rows = [r for session in ("parent", "child") for r in engine._store.get_session_messages(session)]
        assert len([r for r in rows if r["content"] == current["content"]]) >= 2


def test_user_typed_recovery_text_is_not_emission(tmp_path: Path) -> None:
    from hermes_lcm.engine import _OVERFLOW_RECOVERY_PLACEHOLDER  # type: ignore[import-not-found]

    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest([message("same request", 10.)])
        fake = message(_OVERFLOW_RECOVERY_PLACEHOLDER, 20.)
        engine.ingest([fake])
        fake["content"] = "same request"
        engine.ingest([fake, message("reply", 21., "assistant")])
        assert [r["observed_at"] for r in users(engine) if r["content"] == "same request"] == [10., 20.]


def test_recovery_without_stored_copy_stores_new_text(tmp_path: Path) -> None:
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        recovered, _current = recover(engine)
        recovered[-1]["timestamp"] = 30.
        engine.ingest(deepcopy(recovered))
        recovered[-1]["content"] = "a request not yet stored"
        engine.ingest([*recovered, message("reply", 31., "assistant")])
        assert len([r for r in users(engine) if r["content"] == "a request not yet stored"]) == 1


@pytest.mark.parametrize("shape", ["whitespace", "structured", "model-switch", "todo"])
def test_fresh_identity_variants_at_recovery_stamp_are_stored(tmp_path: Path, shape: str) -> None:
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        recovered, current = recover(engine)
        recovered[-1]["timestamp"] = 30.
        engine.ingest(deepcopy(recovered))
        content = current["content"]
        if shape == "whitespace":
            content = "  " + content + "  "
        elif shape == "structured":
            content = [{"type": "text", "text": content}]
        elif shape == "model-switch":
            content = "[Note: model was just switched from model-a to model-b via router.]\n\n" + content
        else:
            content += "\n\n[System: todo list snapshot]\n- finish the request"
        # Equal stamp and matching normalized content do not make a fresh dict LCM's emitted row.
        fresh = {**recovered[-1], "content": content}
        before = len(users(engine))
        engine.ingest([*recovered[:-1], fresh, message("reply", 31., "assistant")])
        assert len(users(engine)) == before + 1


def assert_copies(engine: Any, text: str, expected: int = 1) -> None:
    copies = [r for r in users(engine) if r["content"] == text]
    assert len(copies) == expected, [(r["store_id"], r["observed_at"]) for r in copies]


def live_rewrite(engine: Any) -> tuple[list[dict[str, Any]], str]:
    view, current = recover(engine)
    view[-1]["timestamp"] = 30.
    engine.ingest(deepcopy(view))
    view[-1]["content"] = current["content"]
    view.append(message("reply", 31., "assistant"))
    engine.ingest(view)
    assert_copies(engine, current["content"])
    return view, current["content"]


@pytest.mark.parametrize("shape", ["rotation", "cold-compress", "cold-rotation", "trim", "steer",
                                    "compression-callback", "copied-compress"])
def test_recovery_alias_survives_delayed_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str) -> None:
    pin_char_counter(monkeypatch)  # type: ignore[no-untyped-call]
    ScriptedSummarizer("outage").install(monkeypatch)  # type: ignore[no-untyped-call]
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        view, text = live_rewrite(engine)
        view += [message("next request", 40.), message("next reply", 41., "assistant")]
        engine.ingest(view)
        assert_copies(engine, text)
        if shape.startswith("cold"):
            view = deepcopy(view)
        else:
            if shape in {"rotation", "compression-callback"}:
                engine.on_session_end("parent", view)
                engine.on_session_start("child" if shape == "rotation" else "parent", platform="cli",
                                        context_length=128000, conversation_id="conv", boundary_reason="compression",
                                        old_session_id="parent")
            if shape == "trim":
                view = view[1:]
            elif shape == "steer":
                view = [message("steering request", 39.), *view]
            else:
                view = engine.compress(deepcopy(view), current_tokens=100, force=True)
            assert_copies(engine, text)
            engine.ingest(deepcopy(view))
            assert_copies(engine, text)
    if shape.startswith("cold"):
        with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
            engine.ingest(view)
            assert_copies(engine, text)
            if shape == "cold-rotation":
                engine.on_session_end("parent", view)
                engine.on_session_start("child", platform="cli", context_length=128000, conversation_id="conv",
                                        boundary_reason="compression", old_session_id="parent")
            engine.compress(view, current_tokens=100, force=True)
            assert_copies(engine, text)


def test_recovery_alias_is_scoped_to_its_conversation(tmp_path: Path) -> None:
    """The alias names one row in one conversation; another conversation's event at that stamp is its own."""
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        _, text = live_rewrite(engine)
        assert [r["content"] for r in engine._store.find_rows_by_observed_at("conv", ["parent"], [30.])] == [text]
        assert engine._store.find_rows_by_observed_at("other-conv", ["parent"], [30.]) == []
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="other-conv") as engine:
        engine.ingest([{**message(text, 30.), "platform_message_id": "other-event"}, message("reply", 32., "assistant")])
        stored = [r for r in users(engine) if r["content"] == text]
        assert sorted(r["conversation_id"] for r in stored) == ["conv", "other-conv"]


@pytest.mark.parametrize("observe", [False, True], ids=["late-stamp", "observed-stamp"])
@pytest.mark.parametrize("key", ["message_uid", "platform_message_id", "event_id"])
def test_new_occurrence_key_on_reused_object_stores(tmp_path: Path, observe: bool, key: str) -> None:
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        view, current = recover(engine)
        view[-1]["timestamp"] = 30.
        if observe:
            engine.ingest(deepcopy(view))
        view[-1]["content"] = current["content"]
        view[-1][key] = "new-event"
        engine.ingest([*view, message("reply", 31., "assistant")])
        assert_copies(engine, current["content"], 2)


@pytest.mark.parametrize("guard", ["reused-dict-resend", "observed-then-new-stamp", "fresh-object", "position",
                                   "unknown-stamp", "no-recent-copy", "new-user"])
def test_stamp_first_seen_after_rewrite_stores(tmp_path: Path, guard: str) -> None:
    """Only the stamp observed while the generated bytes survive proves the rewrite. A stamp first seen after it
    cannot tell the host's in-place rewrite from a distinct identical resend placed in the reused dict (same
    index, no uid, assistant/tool-only followers), so the row is stored: duplication over loss."""
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        view, current = recover(engine)
        if guard == "observed-then-new-stamp":
            view[-1]["timestamp"] = 30.
            engine.ingest(deepcopy(view))
        view[-1]["content"] = current["content"]
        view[-1]["timestamp"] = {"unknown-stamp": None, "observed-then-new-stamp": 40.}.get(guard, 30.)
        if guard == "fresh-object":
            view[-1] = dict(view[-1])
        elif guard == "position":
            view.insert(0, message("another reply", 29., "assistant"))
        elif guard == "no-recent-copy":
            engine._identity_anchor_recent = []
        elif guard == "new-user":
            engine.ingest([message("a later request", 29.)])
        engine.ingest([*view, message("reply", 31., "assistant")])
        assert_copies(engine, current["content"], 2)


@pytest.mark.parametrize("abandoned", ["cancelled", "no-copy", "followers", "no-stamp", "session-switch"])
def test_stale_receipt_does_not_block_new_recoveries(tmp_path: Path, abandoned: str) -> None:
    lineage(tmp_path)
    with engine_context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        first, _ = recover(engine)
        if abandoned in {"no-copy", "followers"}:
            first[-1]["timestamp"] = 30.
            engine.ingest(deepcopy(first))
            if abandoned == "no-copy":
                first[-1]["content"] = "not stored"
            engine.ingest([*first, message("intervening request", 32.)])
        elif abandoned == "no-stamp":
            engine.ingest(deepcopy(first))
        elif abandoned == "session-switch":
            engine.on_session_start("elsewhere", platform="cli", conversation_id="other")
            engine.on_session_start("parent", platform="cli", conversation_id="conv")
        for turn in range(6):
            stamp = 100. + turn * 10
            current = message(f"next recovery {turn} " * 500, stamp)
            source = [message("earlier request", stamp - 2), message("reply", stamp - 1, "assistant"), current,
                      {"role": "tool", "content": "orphan", "tool_call_id": "orphan", "timestamp": stamp + 1}]
            engine.ingest(source)
            view = engine._assemble_overflow_recovery_context(None, source, assembly_cap_override=120)
            view = engine._finalize_forced_overflow_result(source, view, assembly_cap_override=120)
            engine._record_compress_commit_proof(source, view)
            view[-1]["timestamp"] = stamp + 2
            engine.ingest(deepcopy(view))
            view[-1]["content"] = current["content"]
            engine.ingest([*view, message("reply", stamp + 3, "assistant")])
            assert_copies(engine, current["content"])
        assert len(engine._identity_anchor_recoveries) <= 4
