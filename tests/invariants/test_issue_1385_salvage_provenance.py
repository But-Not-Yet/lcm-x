"""I1/I2/I8: a placeholder must not borrow an impossible salvage source (#1385)."""

import json

import pytest

from tests.test_issue_773_hermes_pruned_tool_replay import (
    CLEARED,
    SYSTEM,
    _call,
    _grouped,
    _rebind,
    _result,
    make_engine,  # noqa: F401
    plain_engine,
    stored,
)


@pytest.mark.parametrize("content", [
    ["list output " * 30],
    {"output": "dict output " * 30},
    int("9" * 250),
], ids=["untyped-list", "plain-dict", "number"])
def test_nonstring_source_cannot_hide_a_fresh_placeholder_pair(make_engine, content):  # noqa: F811
    """Canonical storage loses the content kind: fail closed for every non-string shape.

    Two newer outputs make X eligible by position; only source-kind provenance can reject it.
    Seed the durable representation directly so this also covers stores from older writers.
    """
    first = plain_engine(make_engine, fresh_tail_count=64)
    view = [SYSTEM, {"role": "user", "content": "original turn"}, _call("X"), _result("X", content),
            _call("Y"), _result("Y", "Y" * 300), _call("Z"), _result("Z", "Z" * 300)]
    for row in view:
        first._store.append(first._session_id, row, conversation_id=first._conversation_id)
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    before = stored(engine)
    assert before[3]["content"] == json.dumps(content, ensure_ascii=False, sort_keys=True)
    host = [dict(row, content=CLEARED) if i == 3 else row for i, row in enumerate(view)]

    engine._ingest_messages(host)

    rows = stored(engine)
    assert rows[:len(before)] == before
    added = rows[len(before):]
    assert any(row["role"] == "tool" and row["content"] == CLEARED for row in added)
    assert any(row.get("tool_calls") == host[2]["tool_calls"] for row in added)
    assert _grouped(rows)


@pytest.mark.parametrize("newer_tools", [0, 1, 2], ids=["newest", "second-newest", "true-salvage"])
@pytest.mark.parametrize("system", [False, True], ids=["effective-and-raw", "system-anchor"])
@pytest.mark.parametrize("proof", ["store", "host-rewrite", "native-recovery"])
def test_full_replay_uses_complete_list_placeholder_positions(make_engine, newer_tools, system, proof):  # noqa: F811
    """A full-length match cannot borrow a newest-two identity, even without a tool anchor.

    With two newer outputs the same occurrence really can be salvaged. Checking a candidate
    prefix's newest two instead of the complete list would duplicate that genuine replay.
    """
    first = plain_engine(make_engine, fresh_tail_count=64)
    view = ([SYSTEM] if system else []) + [{"role": "user", "content": "turn"},
                                         _call("X"), _result("X", "X" * 300)]
    for i in range(newer_tools):
        view += [_call(f"later-{i}"), _result(f"later-{i}", f"later output {i}")]
    first._ingest_messages(view)
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    if proof == "native-recovery":
        engine._remember_native_recovery_replay_snapshot_digest(engine._native_recovery_replay_snapshot_digest(view))
    before = stored(engine)
    host = [dict(row, content=CLEARED) if row.get("tool_call_id") == "X" else row for row in view]
    if proof == "host-rewrite":
        host = [dict(row, content="turn ") if row.get("role") == "user" else row for row in host]
        host += [{"role": "user", "content": "next turn"}]

    engine._ingest_messages(host)

    rows = stored(engine)
    assert rows[:len(before)] == before
    if newer_tools == 2:
        assert [row["content"] for row in rows[len(before):]] == (["next turn"] if proof == "host-rewrite" else [])
    else:
        added = rows[len(before):]
        assert any(row["role"] == "tool" and row["content"] == CLEARED for row in added)
        assert any(row.get("tool_calls") == _call("X")["tool_calls"] for row in added)
    assert _grouped(rows)
