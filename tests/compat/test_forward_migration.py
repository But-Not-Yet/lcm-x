"""A previous-GA writer's real store survives upgrade, repeat open and rollback."""

import hashlib
import json
import sqlite3

import pytest

from tests.compat.helpers import config_for, run_old

WRITE_AND_READ = r'''
if mode == "write":
    options.update(fresh_tail_count=4, leaf_chunk_tokens=1000, large_output_externalization_enabled=True)
    if data == "fleet":
        options.update(leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
                       large_output_externalization_enabled=True,
                       large_output_externalization_threshold_chars=12000,
                       large_output_active_replay_stubbing_enabled=True,
                       large_output_active_replay_stub_threshold_tokens=10000,
                       large_output_active_replay_stub_aged_threshold_tokens=2000)
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    if mode == "write":
        for index in range(3):
            session, conversation = f"session-{index}", f"conversation-{index}"
            engine.on_session_start(session, platform="cli", context_length=200_000,
                                    conversation_id=conversation)
            view = []
            for turn in range(12):
                view.extend([
                    {"role": "user", "content": [{"type": "text", "text": f"Question {index}/{turn}: " + "detail " * 800}],
                     "timestamp": 100.0 + 2 * turn},
                    {"role": "assistant", "content": f"Answer {index}/{turn}: " + "explanation " * 500,
                     "timestamp": 101.0 + 2 * turn},
                ])
            view.extend([
                {"role": "user", "content": "Run the large report.", "timestamp": 130.0},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "report", "type": "function", "function": {"name": "report", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "report", "name": "report", "content": "report data " * 5000},
                {"role": "assistant", "content": "The report is complete."},
            ])
            engine.ingest(view)
            engine.compress(view, current_tokens=100_000, force=True)
            # The default old reader externalizes summarizer input, so expose
            # the large result outside the protected tail on a second pass.
            view.extend([{"role": "user", "content": f"Follow-up {n}", "timestamp": 140.0 + n}
                         for n in range(6)])
            engine.compress(view, current_tokens=100_000, force=True)
            assert engine._dag.get_session_nodes(session), "healthy fake summary must produce a DAG"
        # Exercise the old survival fit separately from successful compaction;
        # all durable message writes still go through public ingest/compress.
        engine.on_session_start("survival", platform="cli", context_length=2000, conversation_id="survival-conv")
        view = [{"role": "user", "content": "retain " * 2000, "timestamp": 200.0},
                {"role": "assistant", "content": "answer " * 2000, "timestamp": 201.0}]
        engine.ingest(view)
        fitted = engine._survival_fit(view, view, engine._survival_measure(view), "compat_fixture")
        assert fitted != view
    sessions = [f"session-{index}" for index in range(3)] + ["survival"]
    rows = {session: engine._store.get_session_messages(session) for session in sessions}
    assert all(rows.values())
    nodes = {session: len(engine._dag.get_session_nodes(session)) for session in sessions[:3]}
    assert all(nodes.values())
    assert list((Path(home).parent / "payloads").glob("*")), "large output must be externalized"
    assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({"counts": {session: len(value) for session, value in rows.items()}, "nodes": nodes}))
finally:
    engine.shutdown()
'''


def snapshot(directory):
    with sqlite3.connect(directory / "lcm.db") as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        schema = {table: [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')] for table in tables}
        rows = conn.execute("SELECT session_id, conversation_id, store_id, role, content, tool_calls, tool_call_id "
                            "FROM messages ORDER BY store_id").fetchall()
        conversations = {}
        for session, conversation, *row in rows:
            conversations.setdefault((session, conversation), []).append(row)
        hashes = {key: (len(value), hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
                  for key, value in conversations.items()}
        nodes = conn.execute("SELECT * FROM summary_nodes ORDER BY node_id").fetchall()
        provenance = conn.execute("SELECT * FROM summary_node_provenance ORDER BY node_id").fetchall()
        assert nodes and provenance
        ids = {row[2] for row in rows}
        node_ids = {row[0] for row in nodes}
        for node in nodes:
            source_ids = json.loads(node[schema["summary_nodes"].index("source_ids")])
            source_type = node[schema["summary_nodes"].index("source_type")]
            assert source_ids and set(source_ids) <= (ids if source_type == "messages" else node_ids)
        payloads = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (directory / "payloads").rglob("*") if path.is_file()}
        return schema, hashes, nodes, provenance, payloads


@pytest.mark.parametrize("overlay", ["default", "fleet"])
def test_previous_ga_forward_migration(tmp_path, old_tree, host_root, monkeypatch, overlay):
    from hermes_lcm import db_bootstrap
    from hermes_lcm.engine import LCMEngine

    old = old_tree("v0.26.2")
    original = run_old(old, tmp_path, host_root, WRITE_AND_READ, mode="write", data=overlay)
    before = snapshot(tmp_path)
    assert len(before[1]) == 4 and before[-1]
    # Keep the asynchronous derived-index backfill out of the reopen comparison.
    monkeypatch.setattr(db_bootstrap, "dispatch_background_stem_backfill", lambda *args: None)
    first_open = None
    for opening in range(2):
        engine = LCMEngine(config=config_for(tmp_path), hermes_home=str(tmp_path / "home"))
        try:
            for index in range(3):
                engine.on_session_start(f"session-{index}", platform="cli", context_length=200_000,
                                        conversation_id=f"conversation-{index}")
                assert len(engine._dag.get_session_nodes(f"session-{index}")) == original["nodes"][f"session-{index}"]
            assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            engine.shutdown()
        after = snapshot(tmp_path)
        assert after[1:] == before[1:], "upgrade must preserve every conversation, DAG and payload"
        assert all(set(columns) <= set(after[0][table]) for table, columns in before[0].items())
        if opening:
            assert after == first_open, "second open must not change the persisted shape or data"
        first_open = after
    # #919's floor concerns replay of NEW short projections, not opening a store:
    # this candidate has emitted none. test_rollback_v0262_store defines this path.
    assert run_old(old, tmp_path, host_root, WRITE_AND_READ) == original
    assert snapshot(tmp_path)[1:] == before[1:]
