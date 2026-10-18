"""#1536: rollback cursor growth must invalidate only the unproven coverage range."""

from collections.abc import Iterator
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

command = importlib.import_module("hermes_lcm.command")
chunking = importlib.import_module("hermes_lcm.chunking")
config_module = importlib.import_module("hermes_lcm.config")
vector_module = importlib.import_module("hermes_lcm.vector_store")

LONG = "The archive contains the complete synthetic migration history. " * 8
SHORT = "My bicycle is blue."
Store = tuple[Any, Any, Any, list[str]]


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    config = config_module.LCMConfig(
        database_path=str(tmp_path / "rollback.db"), embeddings_enabled=True,
        embedding_provider="ollama", embedding_model="model-a",
    )
    vectors = vector_module.VectorStore(config.database_path, config=config)
    vectors.ensure_chunk_schema()
    vectors.register_profile("model-a", "ollama", 2, task="chunk")
    identity = vector_module.EmbeddingIdentity.canonical(
        "ollama", "model-a", "", 2, "float32", "little", "chunk",
    )
    vectors.connection.execute(
        "CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT, "
        "session_id TEXT NOT NULL DEFAULT 'synthetic', source TEXT DEFAULT '', "
        "timestamp REAL NOT NULL DEFAULT 0, conversation_id TEXT DEFAULT '')"
    )
    texts: list[str] = []

    def embed(documents: list[str]) -> list[list[float]]:
        texts.extend(documents)
        return [[1.0, 0.0] for _ in documents]

    provider = SimpleNamespace(provider_id="ollama", model_id="model-a",
                               last_skipped_documents=[], embed_documents=embed)
    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_k: provider)
    yield config, vectors, identity, texts
    vectors.close()


def add_rows(store: Store, rows: list[tuple[int, str, str]], embedded: set[int]) -> None:
    _, vectors, identity, _ = store
    vectors.connection.executemany("INSERT INTO messages(store_id, role, content) VALUES(?, ?, ?)", rows)
    # c67d7633 chunking.py:226-229 skips <40-token turns. Its command.py:5020-5026
    # then advances MIN/MAX from published long rows and never revisits enclosed shorts.
    for row in rows:
        if row[0] in embedded:
            for chunk in chunking.chunk_message(*row, policy="full"):
                vectors.record_chunk_embedding(
                    chunk.chunk_id, "model-a", [1.0, 0.0], store_id=chunk.store_id,
                    chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                    char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity,
                )


def metadata(store: Store, key: str, value: int | None = None) -> str | None:
    _, vectors, identity, _ = store
    key = f"{key}:{identity.identity_hash}"
    if value is not None:
        vectors.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)", (key, str(value)))
    row = vectors.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def run(store: Store, *, limit: int = 32, incremental: bool = True) -> None:
    config, vectors, _, _ = store
    before = list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
    result = command._chunk_backfill_run(
        config, config.database_path, apply=True, limit=limit,
        retry_uncertain=False, policy="conversational", incremental=incremental,
    )
    assert isinstance(result, dict) and result["error"] is None and not result["failed"], result
    assert before == list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))


def counts(store: Store) -> dict[int, int]:
    _, vectors, identity, _ = store
    return dict(vectors.connection.execute(
        "SELECT store_id, COUNT(*) FROM lcm_chunk_meta WHERE identity_hash=? GROUP BY store_id",
        (identity.identity_hash,),
    ))


@pytest.mark.parametrize("bound", [None, 3])
def test_completed_marker_repairs_rollback_and_already_affected_store(store: Store, bound: int | None) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG)], {1, 2, 3})
    metadata(store, "chunk_policy_revision", 2)  # rc1 completed its original fill.
    if bound is not None:
        metadata(store, "chunk_policy_revision_through", bound)
    # v0.27.0 publishes long 5 but leaves short 4 enclosed; all marker keys survive.
    add_rows(store, [(4, "user", "I work Fridays."), (5, "assistant", LONG)], {5})
    for _ in range(3):
        run(store)
    assert counts(store) == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert store[3] == ["I work Fridays."]
    assert metadata(store, "chunk_policy_revision") == "2"
    assert metadata(store, "chunk_policy_revision_through") == "5"
    run(store)
    assert store[3] == ["I work Fridays."]


@pytest.mark.parametrize("bound", [None, 3])
def test_uncovered_status_is_read_only_and_reports_gapfill(store: Store, bound: int | None) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG),
                     (4, "user", SHORT), (5, "assistant", LONG)], {1, 2, 3, 5})
    metadata(store, "chunk_policy_revision", 2)
    if bound is not None:
        metadata(store, "chunk_policy_revision_through", bound)
    config, vectors, _, _ = store
    before = list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
    assert "embedding_backlog_chunks: 1 (gap-fill)" in command._embedding_maintenance_status_lines(engine)
    assert before == list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))


@pytest.mark.parametrize("scoped_skip", [False, True])
def test_mid_fill_skip_does_not_hide_new_rollback_rows(store: Store, scoped_skip: bool) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "user", "I work Fridays."),
                     (4, "user", "\t\n"), (5, "assistant", LONG)], {1, 3, 5})
    metadata(store, "chunk_gapfill_skip_below", 4)  # rc1 stopped after publishing short 3.
    if scoped_skip:
        metadata(store, "chunk_gapfill_skip_through", 5)
    add_rows(store, [(6, "user", "My cat is grey."), (7, "assistant", LONG)], {7})
    for _ in range(5):
        run(store, limit=1)
    assert counts(store) == {1: 1, 2: 1, 3: 1, 5: 1, 6: 1, 7: 1}
    assert store[3] == ["My cat is grey.", SHORT]
    assert metadata(store, "chunk_policy_revision_through") == "7"


def test_legacy_marker_with_full_coverage_is_revalidated_once(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG)], {1, 2, 3})
    metadata(store, "chunk_policy_revision", 2)
    run(store)
    assert metadata(store, "chunk_policy_revision_through") == "3"

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("steady coverage must not repeat gap-fill")

    monkeypatch.setattr(command, "_chunk_gapfill_rows", unexpected)
    run(store)
    assert not store[3]


def test_unpublished_gaps_do_not_advance_coverage_bound(store: Store) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "user", "I work Fridays."),
                     (4, "assistant", LONG)], {1, 4})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 1)
    run(store, limit=1)
    assert counts(store) == {1: 1, 3: 1, 4: 1}
    assert metadata(store, "chunk_policy_revision_through") == "1"
    for _ in range(3):
        run(store, limit=1)
    assert counts(store) == {1: 1, 2: 1, 3: 1, 4: 1}
    assert metadata(store, "chunk_policy_revision_through") == "4"


def test_full_backfill_stamps_coverage_bound(store: Store) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG)], {1, 3})
    run(store, incremental=False)
    assert counts(store) == {1: 1, 2: 1, 3: 1}
    assert metadata(store, "chunk_policy_revision_through") == "3"
