"""Shipped rollup defaults survive an older reader's additional compaction."""

from collections.abc import Callable
from pathlib import Path
import sqlite3

import pytest

from tests.compat.helpers import run_old
from tests.compat.storage_fixtures import (
    SESSION, VECTOR_TABLES, assert_complete_vectors, assert_preserved, bind, embed_all,
    install_fakes, payload_hashes, plugin, rebuild_rollup, seed_history, storage_config,
    table_hashes, warmup,
)

OLD_INGEST_AND_COMPRESS = r'''
from tests.compat.storage_fixtures import (
    SESSION, bind, install_fakes, seed_history, storage_config,
)
install_fakes()
config = storage_config(Path(db).parent, fleet=data == "fleet", embeddings=data == "fleet")
engine = LCMEngine(config=config, hermes_home=home)
try:
    bind(engine)
    before = engine._store.get_session_messages(SESSION)
    nodes = len(engine._dag.get_session_nodes(SESSION))
    assert before and nodes
    seed_history(engine, "old-reader")
    assert len(engine._store.get_session_messages(SESSION)) > len(before)
    assert len(engine._dag.get_session_nodes(SESSION)) > nodes
    assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({"messages": len(engine._store.get_session_messages(SESSION))}))
finally:
    engine.shutdown()
'''


@pytest.mark.parametrize("policy", ["defaults", "fleet"])
def test_shipped_rollups_rollback(
    tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path,
    monkeypatch: pytest.MonkeyPatch, policy: str,
) -> None:
    install_fakes(monkeypatch)
    config = storage_config(tmp_path, fleet=policy == "fleet", embeddings=policy == "fleet")
    assert config.temporal_rollups_enabled
    # Leave defaults untouched in the first arm; only isolate filesystem destinations.
    engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        seed_history(engine, "candidate")
        rebuild_rollup(engine)
        if policy == "fleet":
            assert engine.drain_embedding_maintenance(timeout=10)
            warmup(engine)
            embed_all(engine)
            assert_complete_vectors(engine)
        before = table_hashes(tmp_path, ("messages", "summary_nodes", "summary_node_provenance"))
        payloads = payload_hashes(tmp_path)
        assert payloads
        rollups = table_hashes(tmp_path, ("lcm_rollup_sources",))
        rollup_rows = table_hashes(tmp_path, ("lcm_rollups",))["lcm_rollups"]
        with sqlite3.connect(tmp_path / "lcm.db") as conn:
            ids = {row[0] for row in conn.execute("SELECT rollup_id FROM lcm_rollups")}
        vectors = table_hashes(tmp_path, VECTOR_TABLES) if policy == "fleet" else {}
    finally:
        engine.shutdown()
    report = run_old(old_tree("v0.26.2"), tmp_path, host_root, OLD_INGEST_AND_COMPRESS, data=policy)
    assert_preserved(tmp_path, before)
    assert_preserved(tmp_path, rollups)
    assert_preserved(tmp_path, vectors)
    assert payloads.items() <= payload_hashes(tmp_path).items()
    # Both versions understand these rows: publication invalidates them, rather
    # than deleting them. Freeze automatic rebuilding to inspect that durable debt.
    monkeypatch.setattr(plugin("engine").LCMEngine, "_schedule_rollup_maintenance", lambda *args: None)
    engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        assert len(engine._store.get_session_messages(SESSION)) == report["messages"]
        assert_preserved(tmp_path, before)
        assert_preserved(tmp_path, rollups)
        assert payloads.items() <= payload_hashes(tmp_path).items()
        rollup_store = plugin("rollup_store").RollupStore(tmp_path / "lcm.db")
        try:
            rollup_store.drain_invalidations()
        finally:
            rollup_store.close()
        conn = engine._store.connection
        rows = dict(conn.execute("SELECT rollup_id, status FROM lcm_rollups"))
        assert ids <= rows.keys()
        assert any(rows[rollup_id] == "stale" for rollup_id in ids)
        assert all(rows[rollup_id] in {"ready", "stale"} for rollup_id in ids)
        after = table_hashes(tmp_path, ("lcm_rollups",))["lcm_rollups"]
        assert all(after.get(row_id) == digest or rows.get(row_id) == "stale"
                   for row_id, digest in rollup_rows.items())
        assert conn.execute("SELECT count(*) FROM lcm_rollup_sources s LEFT JOIN summary_nodes n "
                            "ON s.node_id=n.node_id LEFT JOIN lcm_rollups r ON s.rollup_id=r.rollup_id "
                            "WHERE n.node_id IS NULL OR r.rollup_id IS NULL").fetchone()[0] == 0
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if policy == "fleet":
            assert engine.drain_embedding_maintenance(timeout=10)
            embed_all(engine)
            assert_complete_vectors(engine)
    finally:
        engine.shutdown()
