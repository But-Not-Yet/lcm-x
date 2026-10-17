"""Provider transport and optional dependency contracts without network or models."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

providers = importlib.import_module("hermes_lcm.embedding_provider")


def test_spend_guard_clear_reopens_without_advancing_time() -> None:
    guard = providers.EmbeddingSpendGuard(max_calls=2, backoff_seconds=100)
    guard.record_call(now=1)
    guard.record_call(now=1)
    assert not guard.allows(now=1)
    guard.clear()
    assert guard.allows(now=1)
    guard.record_call(now=1)
    assert guard.allows(now=1)


@pytest.mark.parametrize("name", ["voyage", "ollama", "openai-compatible"])
def test_http_documents_report_discovered_dimension(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def transport(**kwargs: Any) -> Any:
        calls.append(kwargs)
        body = {"embeddings": [[3.0, 4.0]]} if name == "ollama" else {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}
        return providers.HttpResponse(status=200, headers={}, body=json.dumps(body).encode())

    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-fixture")
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "synthetic-fixture")
    monkeypatch.setattr(providers, "count_tokens", lambda _text: 1)
    if name == "voyage":
        provider = providers.VoyageProvider("fixture", transport=transport)
    elif name == "ollama":
        provider = providers.OllamaProvider("fixture", transport=transport)
    else:
        provider = providers.OpenAICompatibleProvider("fixture", base_url="https://example.test/v1", transport=transport)
    assert provider.dim == 0
    assert provider.embed_documents(["document"]) == [[3.0, 4.0]]
    assert provider.dim == 2
    assert len(calls) == 1


@pytest.mark.parametrize("dispatch", [False, True])
def test_openai_compatible_batches_keep_input_indexes_and_dispatch_order(
    monkeypatch: pytest.MonkeyPatch, dispatch: bool,
) -> None:
    events: list[Any] = []

    def transport(**kwargs: Any) -> Any:
        events.append(kwargs["payload"]["input"])
        return providers.HttpResponse(status=200, headers={}, body=json.dumps({"data": [
            {"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]},
        ]}).encode())

    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "synthetic-fixture")
    provider = providers.OpenAICompatibleProvider("fixture", base_url="https://example.test/v1", transport=transport)
    batches = list(provider.embed_document_batches(["first", "second"], before_dispatch=events.append if dispatch else None))
    assert [batch.indexes for batch in batches] == [(0, 1)]
    assert batches[0].vectors == ((1.0, 0.0), (0.0, 1.0))
    assert events == ([(0, 1), ["first", "second"]] if dispatch else [["first", "second"]])
    assert list(provider.embed_document_batches([])) == []
    assert len(events) == (2 if dispatch else 1)


def test_fastembed_import_constructs_local_only_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[dict[str, Any]] = []

    class TextEmbedding:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

        def embed(self, texts: Any) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    module = ModuleType("fastembed")
    module.TextEmbedding = TextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    provider = providers.FastembedProvider("fixture", cache_dir=tmp_path)
    assert provider.embed_documents(["first"]) == [[1.0, 0.0]]
    assert provider.dim == 2
    assert constructed == [{"model_name": "fixture", "cache_dir": str(tmp_path), "threads": 2, "local_files_only": True}]
