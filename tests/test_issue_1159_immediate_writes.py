"""#1159 DDL-rollback writes wait for a concurrent writer."""

import sqlite3
import threading
import time

import pytest

from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore


@pytest.mark.parametrize("kind", ["store", "dag"])
def test_ddl_rollback_write_waits_for_other_writer(tmp_path, kind):
    path = tmp_path / "write-wait.db"
    owner = MessageStore(path) if kind == "store" else SummaryDAG(path)
    conn = owner.connection

    def write(content):
        if kind == "store":
            return owner._append_protected_batch(
                "session", [{"role": "user", "content": content}]
            )[0]
        return owner.add_node(SummaryNode(session_id="session", summary=content))

    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    timer = None
    trace = []
    try:
        warm_id = write("warm cached insert")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE rolled_back_schema(value)")
        conn.rollback()
        conn.set_trace_callback(trace.append)
        holder.execute("BEGIN IMMEDIATE")
        timer = threading.Timer(0.3, holder.commit)
        timer.start()
        start = time.monotonic()
        try:
            stored_id = write("waited insert")
        except sqlite3.OperationalError as exc:
            pytest.fail(
                f"write failed after {time.monotonic() - start:.6f}s: "
                f"code={exc.sqlite_errorcode} {exc}"
            )
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2, (elapsed, trace)
        assert stored_id > warm_id
        stored = owner.get(stored_id) if kind == "store" else owner.get_node(stored_id)
        assert (stored["content"] if kind == "store" else stored.summary) == "waited insert"
        assert [sql for sql in trace if sql.startswith("BEGIN")] == ["BEGIN IMMEDIATE"]
        assert not conn.in_transaction
    finally:
        if timer is not None:
            timer.join()
        holder.rollback()
        holder.close()
        owner.close()
