"""Exact constructor traces captured before connection-recipe extraction."""

import json
from pathlib import Path
import sqlite3
import time

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.assertion_store import AssertionStore
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.query_view_store import QueryViewStore
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.store import MessageStore
from hermes_lcm.trajectory_store import CorpusIdentity, TrajectoryStore


CASES = ("trajectory", "trajectory_ro", "assertion", "assertion_ro", "rollup", "query_view")
GOLDEN = Path(__file__).with_name("fixtures") / "connection_recipes.json"


def prepare(kind, path):
    MessageStore(path).close()
    SummaryDAG(path).close()
    if kind.endswith("_ro"):
        construct(kind.removesuffix("_ro"), path).close()


def construct(kind, path):
    if kind.startswith("trajectory"):
        return TrajectoryStore(
            path,
            CorpusIdentity(
                dataset_name="example/recipes", dataset_revision="revision-1",
                harness_commit="harness-1", tier="small", domain="example",
                ingest_config_digest="ingest-1",
            ),
            asset_root=path.parent, read_only=kind.endswith("_ro"),
        )
    if kind.startswith("assertion"):
        return AssertionStore(path, read_only=kind.endswith("_ro"))
    return {"rollup": RollupStore, "query_view": QueryViewStore}[kind](path)


def trace_connections(monkeypatch, path, *, fail_statement=None):
    """Record calls without changing the kwargs presented by production code."""
    events = []
    connections = []
    original = sqlite3.connect

    class Connection(sqlite3.Connection):
        def __setattr__(self, name, value):
            if name in ("row_factory", "isolation_level"):
                events.append(["set", name, "Row" if value is sqlite3.Row else value])
            super().__setattr__(name, value)

        def execute(self, sql, parameters=()):
            events.append(["execute", sql, list(parameters)])
            if sql == fail_statement:
                raise sqlite3.OperationalError("recipe failure")
            return super().execute(sql, parameters)

        def executescript(self, sql):
            events.append(["executescript", sql])
            return super().executescript(sql)

        def commit(self):
            events.append(["commit"])
            return super().commit()

        def close(self):
            events.append(["close"])
            return super().close()

    def connect(*args, **kwargs):
        # Bootstrap verifiers also open scratch databases; those are not the
        # store connection whose recipe is being characterized.
        if str(args[0]) == ":memory:":
            return original(*args, **kwargs)
        from urllib.parse import quote

        database = str(args[0]).replace(quote(str(path), safe="/"), "<database>")
        database = database.replace(str(path), "<database>")
        events.append(["connect", [database, *args[1:]], kwargs])
        conn = original(*args, **kwargs, factory=Connection)
        connections.append(conn)
        events.append(["initial", conn.isolation_level, conn.row_factory])
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return events, connections


@pytest.fixture(autouse=True)
def deterministic_recipe_environment(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    monkeypatch.setattr(db_bootstrap, "sqlite_mmap_size", lambda: 268435456)


@pytest.mark.parametrize("kind", CASES)
def test_constructor_trace(kind, tmp_path, monkeypatch):
    path = tmp_path / "recipe #?.db"
    prepare(kind, path)
    events, connections = trace_connections(monkeypatch, path)
    store = construct(kind, path)
    try:
        assert events == json.loads(GOLDEN.read_text())[kind]
        assert len(connections) == 1
    finally:
        store.close()


@pytest.mark.parametrize("kind", ("trajectory", "assertion", "rollup", "query_view"))
def test_future_core_version_refused_before_writable_setup(kind, tmp_path, monkeypatch):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO metadata VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    events, connections = trace_connections(monkeypatch, path)
    try:
        with pytest.raises(db_bootstrap.SchemaVersionTooNewError) as caught:
            construct(kind, path)
        # Keep the failed constructor alive while inspecting its open path;
        # RollupStore's destructor separately checkpoints and closes it.
        assert "999" in str(caught.value)
        statements = [event[1] for event in events if event[0] == "execute"]
        assert statements
        assert all(sql.lstrip().upper().startswith(("SELECT", "PRAGMA TABLE_INFO"))
                   for sql in statements)
        assert not any(event[0] == "executescript" for event in events)
        assert path.read_bytes() == before
        assert events.count(["close"]) == (1 if kind == "query_view" else 0)
    finally:
        for conn in connections:
            conn.close()


@pytest.mark.parametrize("kind", ("trajectory_ro", "assertion_ro"))
def test_read_only_open_preserves_database_bytes(kind, tmp_path):
    path = tmp_path / "read only #?.db"
    prepare(kind, path)
    before = path.read_bytes()
    store = construct(kind, path)
    try:
        assert store.connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert path.read_bytes() == before
    finally:
        store.close()
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ("trajectory", "assertion", "rollup", "query_view"))
def test_setup_failure_cleanup_and_exception(kind, tmp_path, monkeypatch):
    path = tmp_path / "failure.db"
    events, connections = trace_connections(
        monkeypatch, path, fail_statement="PRAGMA busy_timeout=30000",
    )
    try:
        expected = RuntimeError if kind == "query_view" else sqlite3.OperationalError
        with pytest.raises(expected) as caught:
            construct(kind, path)
        prefix = "query-view schema ensure failed: " if kind == "query_view" else ""
        assert str(caught.value) == prefix + "recipe failure"
        assert events.count(["close"]) == (1 if kind == "query_view" else 0)
    finally:
        for conn in connections:
            conn.close()
