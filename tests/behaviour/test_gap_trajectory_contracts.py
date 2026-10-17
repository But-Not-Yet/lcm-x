"""State-index chunking through the public backfill path, with an offline codec."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

trajectory = importlib.import_module("hermes_lcm.trajectory_store")
tokens = importlib.import_module("hermes_lcm.tokens")
harness = importlib.import_module("tests.harness")


class ByteEncoder:
    """Deliberately splits UTF-8 codepoints; chunker must repair the boundary."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, token: int) -> bytes:
        return bytes([token])


class RecordingProvider:
    provider_id = "fixture"
    model_id = "byte-fixture"
    last_usage_tokens = 0

    def __init__(self) -> None:
        self.documents: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.documents.append(text)
        return [1.0, 0.0]


def make_store(tmp_path: Path, provider: RecordingProvider) -> Any:
    asset = tmp_path / "state.png"
    asset.write_bytes(b"fixture")
    identity = trajectory.CorpusIdentity(
        dataset_name="example/gaps", dataset_revision="fixture-v1", harness_commit="fixture-v1",
        tier="small", domain="web", ingest_config_digest="fixture-v1",
    )
    store = trajectory.TrajectoryStore(tmp_path / "lcm.db", identity, asset_root=tmp_path, embedding_provider=provider)
    state = trajectory.TrajectoryState(
        state_index=0, step=0, url="https://example.test/", incoming_action=None,
        thoughts="Inspect", text="visible 🐑 café 🐑 boundary", screenshot_path=asset,
    )
    store.insert(trajectory.TrajectorySource(
        trajectory_id="fixture", ordinal=0, goal="Inspect", start_url=state.url,
        outcome="done", states=(state,), source_payload={"id": "fixture"},
    ))
    store.finalize(["fixture"])
    return store


@pytest.mark.parametrize("budget", [1, 7])
def test_public_state_backfill_preserves_unicode_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int,
) -> None:
    harness.pin_char_counter(monkeypatch)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: ByteEncoder())
    provider = RecordingProvider()
    store = make_store(tmp_path, provider)
    try:
        if budget == 1:
            with pytest.raises(ValueError, match="Unicode character boundary"):
                store.build_state_semantic_index(document_token_budget=budget)
            assert store.active_state_semantic_profile() is None
        else:
            stats = store.build_state_semantic_index(document_token_budget=budget)
            assert stats["states_embedded"] == stats["chunked_states"] == 1
            assert all(len(text.encode("utf-8")) <= budget for text in provider.documents)
            # First request probes dimension; subsequent requests cover the exact document once.
            reconstructed = "".join(provider.documents[1:])
            assert "visible 🐑 café 🐑 boundary" in reconstructed
            assert "�" not in reconstructed
            calls = len(provider.documents)
            resumed = store.build_state_semantic_index(document_token_budget=budget)
            assert resumed["already_embedded"] == 1
            # observed behaviour; see REPORT row 1
            assert len(provider.documents) == calls + 1
            assert resumed["states_embedded"] == 0
            assert resumed["provider_calls"] == 1
        rows = store.connection.execute("SELECT text FROM lcm_trajectory_states").fetchall()
        assert [row[0] for row in rows] == ["visible 🐑 café 🐑 boundary"]
    finally:
        store.close()


def test_query_keeps_lexical_hits_when_attempt_introspection_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.pin_char_counter(monkeypatch)
    provider = RecordingProvider()
    store = make_store(tmp_path, provider)

    class BrokenIdentity:
        def __str__(self) -> str:
            raise ValueError("fixture identity introspection failure")

    try:
        store.build_semantic_index()
        monkeypatch.setattr(provider, "provider_id", BrokenIdentity())
        hits = store.query("visible")
        assert hits
        assert all(hit.match_kind == "fts" for hit in hits)
        assert store.last_semantic_attempt()["reason"] == "attempt_record_introspection_failed"
        assert store.semantic_attempt_counters()["fallbacks"] == 1
    finally:
        store.close()
