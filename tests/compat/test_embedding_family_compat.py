"""A5 embedding-family forward and rollback proof using pinned release code."""

from collections.abc import Callable
from contextlib import closing
from dataclasses import FrozenInstanceError
from pathlib import Path
import sqlite3
from typing import Any, cast

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.schema_families import COMPAT_REGISTRY, FINDING_KINDS, SchemaFinding
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
from tests.compat.helpers import READERS, run_old
from tests.test_embedding_schema_golden import CASES, Case, schema_for

# The previous-GA pin already fetched by the compatibility suite's CI job.
PREVIOUS_GA = "v0.26.2"
run_reader = cast(Callable[..., dict[str, Any]], run_old)
OLD_SETUP = '''
from hermes_lcm.vector_store import VectorStore, EmbeddingIdentity
from hermes_lcm.db_bootstrap import verify_embedding_schema
store = VectorStore(db, config=LCMConfig(embeddings_enabled=False))
identity = EmbeddingIdentity.canonical('fixture', 'fixture-model', '', 2, 'float32', 'little', 'summary')
assert verify_embedding_schema(store._conn) == []
'''
OLD_WRITE = OLD_SETUP + '''
from hermes_lcm.dag import SummaryDAG, SummaryNode
dag = SummaryDAG(db)
for i in (1, 2, 3):
    assert dag.add_node(SummaryNode(session_id='fixture', summary='synthetic', source_token_count=10,
                                   created_at=1.0, earliest_at=1.0, latest_at=1.0)) == i
dag.close()
store.register_profile('fixture-model', 'fixture', 2)
for key, vec in [('1', [1.0, 0.0]), ('2', [0.0, 1.0])]:
    store.record_embedding(key, 'summary', 'fixture-model', vec, identity=identity)
assert store._migration_step_present('embeddings_v1')
store.close()
print(json.dumps({'written': 2}))
'''
OLD_ROLLBACK = OLD_SETUP + '''
rows = store._load_vectors_for_ids(identity.identity_hash, 2, ['1', '2'])
assert dict(zip(rows[1], rows[3])) == {'1': [1.0, 0.0], '2': [0.0, 1.0]}
store.record_embedding('3', 'summary', 'fixture-model', [1.0, 0.0], identity=identity)
rows = store._load_vectors_for_ids(identity.identity_hash, 2, ['1', '2', '3'])
assert dict(zip(rows[1], rows[3])) == {'1': [1.0, 0.0], '2': [0.0, 1.0], '3': [1.0, 0.0]}
assert verify_embedding_schema(store._conn) == []
assert store._migration_step_present('embeddings_v1')
store.close()
print(json.dumps({'verifier': [], 'read': 2, 'after_write': 3}))
'''


def snapshot(path: Path) -> dict[str, list[tuple[Any, ...]]]:
    with closing(sqlite3.connect(path)) as conn:
        return {
            "tables": conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall(),
            "profile": conn.execute("SELECT * FROM lcm_embedding_profile ORDER BY identity_hash").fetchall(),
            "meta": conn.execute("SELECT * FROM lcm_embedding_meta ORDER BY embedded_id").fetchall(),
            "vectors": conn.execute("SELECT * FROM lcm_embedding_vectors ORDER BY embedded_id").fetchall(),
            "binary": conn.execute("SELECT * FROM lcm_embedding_binary ORDER BY embedded_id").fetchall(),
            "marker": conn.execute("SELECT * FROM lcm_migration_state WHERE step_name='embeddings_v1'").fetchall(),
        }


def test_embedding_forward(
    tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert run_reader(old_tree(PREVIOUS_GA), tmp_path, host_root, OLD_WRITE) == {"written": 2}
    path = tmp_path / "lcm.db"
    before = snapshot(path)
    assert len(before["profile"]) == 1 and len(before["vectors"]) == 2 and len(before["marker"]) == 1
    drops: list[str | None] = []
    connect = sqlite3.connect

    def authorize(action: int, name: str | None, column: str | None, database: str | None,
                  trigger: str | None) -> int:
        if action == sqlite3.SQLITE_DROP_TABLE:
            drops.append(name)
        return sqlite3.SQLITE_OK

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = cast(sqlite3.Connection, connect(*args, **kwargs))
        conn.set_authorizer(authorize)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    with closing(VectorStore(path, config=LCMConfig(embeddings_enabled=False))) as store:
        assert store._conn is not None
        assert db_bootstrap.embedding_schema_findings(store._conn) == []
        assert snapshot(path) == before
    assert drops == []


@pytest.mark.parametrize("reader", tuple(READERS))
def test_embedding_rollback(
    reader: str, tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path,
) -> None:
    path = tmp_path / "lcm.db"
    with closing(SummaryDAG(path)) as dag:
        for i in (1, 2, 3):
            assert dag.add_node(SummaryNode(session_id="fixture", summary="synthetic", source_token_count=10,
                                            created_at=1.0, earliest_at=1.0, latest_at=1.0)) == i
    with closing(VectorStore(path, config=LCMConfig(embeddings_enabled=False))) as store:
        store.register_profile("fixture-model", "fixture", 2)
        identity = EmbeddingIdentity.canonical("fixture", "fixture-model", "", 2, "float32", "little", "summary")
        for key, vector in (("1", [1.0, 0.0]), ("2", [0.0, 1.0])):
            store.record_embedding(key, "summary", "fixture-model", vector, identity=identity)
    before = snapshot(path)
    assert run_reader(old_tree(reader), tmp_path, host_root, OLD_ROLLBACK) == {
        "verifier": [], "read": 2, "after_write": 3,
    }
    after = snapshot(path)
    assert after["tables"] == before["tables"] and after["marker"] == before["marker"]
    assert [row for row in after["vectors"] if row[0] != "3"] == before["vectors"]


def test_embedding_registry(tmp_path: Path) -> None:
    assert tuple(COMPAT_REGISTRY) == ("embedding",)
    family = COMPAT_REGISTRY["embedding"]
    assert family.name == "embedding" and family.prefix == "lcm_embedding"
    assert family.marker == "embeddings_v1" and family.removal_version is None
    assert family.tables == db_bootstrap._REQUIRED_EMBEDDING_TABLES
    assert family.indexes == db_bootstrap._REQUIRED_EMBEDDING_INDEXES
    assert family.oldest_reader == next(iter(READERS))
    assert PREVIOUS_GA in READERS
    assert any(row["name"] == family.name and row["prefix"] == family.prefix
               for row in db_bootstrap._INTERIM_FEATURE_FAMILIES)
    with closing(VectorStore(tmp_path / "lcm.db", config=LCMConfig(embeddings_enabled=False))) as store:
        assert store._migration_step_present(family.marker)
    with pytest.raises(FrozenInstanceError):
        setattr(family, "marker", "changed")


@pytest.mark.parametrize("kind", ("missing object", "malformed table", "malformed constraints", "malformed index"))
def test_finding_string_round_trip(kind: str) -> None:
    assert FINDING_KINDS == ("missing object", "malformed table", "malformed constraints", "malformed index")
    finding = SchemaFinding(kind, "object")
    assert str(finding) == f"{kind}: object"
    assert SchemaFinding(*str(finding).split(": ", 1)) == finding
    assert len({finding, SchemaFinding(kind, "object")}) == 1
    with pytest.raises(FrozenInstanceError):
        setattr(finding, "name", "changed")


def test_unknown_finding_kind_rejected() -> None:
    with pytest.raises(ValueError):
        SchemaFinding("unknown", "object")


@pytest.mark.parametrize("case", [CASES[i] for i in (3, 7, 11, 12)], ids=lambda case: case[0])
def test_typed_findings(case: Case) -> None:
    with closing(schema_for(case)) as conn:
        findings = db_bootstrap.embedding_schema_findings(conn)
        assert findings and all(isinstance(finding, SchemaFinding) for finding in findings)
        assert [str(finding) for finding in findings] == list(case[3])
