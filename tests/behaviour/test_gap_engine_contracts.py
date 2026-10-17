"""Observable engine and operator behaviour for the coverage-gap inventory."""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

harness = importlib.import_module("tests.harness")
tools = importlib.import_module("hermes_lcm.tools")
command = importlib.import_module("hermes_lcm.command")
config = importlib.import_module("hermes_lcm.config")
denials = importlib.import_module("hermes_lcm.access_context.denials")
resolution = importlib.import_module("hermes_lcm.access_policy.resolution")
catalog = importlib.import_module("hermes_lcm.teams.catalog")
scope_storage = importlib.import_module("hermes_lcm.scope_storage")
externalize = importlib.import_module("hermes_lcm.externalize")
preanswer = importlib.import_module("hermes_lcm.preanswer_evidence")


@pytest.fixture(autouse=True)
def offline_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    harness.pin_char_counter(monkeypatch)


@pytest.mark.parametrize("allowed", [True, False])
def test_public_decision_reason_alias_preserves_blurring(allowed: bool) -> None:
    decision = denials.Decision.allow() if allowed else denials.Decision.deny(denials.DenialReason.SCOPE_FORBIDDEN)
    assert decision.reason == decision.denial_reason
    public = decision.public()
    assert public.reason == (None if allowed else denials.DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN)
    assert not public.detail


def test_engine_policy_writes_denial_to_durable_audit(tmp_path: Path) -> None:
    with harness.engine_context(tmp_path) as engine:
        catalog.ensure_teams_catalog(engine._store.connection)
        engine.lcm_teams_enabled = True
        engine.get_lcm_access_context = lambda: None
        policy = resolution.policy_for_engine(engine)
        decision = policy.authorize_operation(None, "write", {})
        policy.audit_decision(None, "write", decision.denial_reason, decision.public())
        events = catalog.read_audit_events(engine._store.connection)
        assert not decision.allowed
        assert len(events) == 1
        assert events[0]["allowed"] is False


def test_abandon_retrieval_removes_handle_without_changing_history(tmp_path: Path) -> None:
    with harness.engine_context(tmp_path, window=32000, adaptive_retrieval_enabled=True) as engine:
        engine.ingest([harness.user("Keep the visit record.")])
        before = harness.stored_rows(engine)
        started = json.loads(engine.handle_tool_call("lcm_retrieve", {
            "action": "start", "question": "Where did I travel?",
            "identity": {"intent_type": "travel memory", "operation": "evidence_only",
                         "subject_key": "user:self", "predicate_key": "travel.visit",
                         "scope_key": "personal", "time_mode": "none", "policy_version": "adaptive-v1"},
            "requirements": [{"slot_id": "visits", "description": "visited places", "minimum_refs": 1}],
        }))
        handle = started["retrieval_id"]
        abandoned = json.loads(engine.handle_tool_call("lcm_retrieve", {"action": "abandon", "retrieval_id": handle}))
        assert abandoned["status"] == "abandoned"
        assert abandoned["termination_reason"] == "abandoned by answerer"
        gone = json.loads(engine.handle_tool_call("lcm_retrieve", {"action": "status", "retrieval_id": handle}))
        assert "error" in gone
        assert harness.stored_rows(engine) == before


def test_unknown_preset_names_available_choices_without_mutation(tmp_path: Path) -> None:
    with harness.engine_context(tmp_path) as engine:
        before = deepcopy(vars(engine._config))
        response = command.handle_lcm_command("preset show missing-preset", engine)
        assert "error: unknown preset missing-preset" in response
        assert "available_presets:" in response
        assert "codex_gpt_long_context" in response
        assert vars(engine._config) == before


def test_legacy_threshold_wrapper_reads_profile_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("compression:\n  threshold: 0.93\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Public legacy wrapper is the sole caller of the inventory helper.
    assert config._hermes_compression_threshold(0.8) == 0.93


@pytest.mark.parametrize("content", [
    [{"type": "text", "content": {"value": "<think>private</think>visible"}}],
    [{"type": "output_text", "value": {"content": "<think>private</think>visible"}}],
    [{"type": "input_text", "text": {"text": "<think>private</think>visible"}}],
])
def test_structured_assistant_cleanup_keeps_durable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: Any,
) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path, window=32000) as engine:
        view = [harness.user("Question"), {"role": "assistant", "content": content}]
        original = deepcopy(view)
        engine.ingest(view)
        before = harness.stored_rows(engine)
        emitted = engine.compress(view, current_tokens=100, force=True)
        assert "private" not in json.dumps(emitted)
        assert "visible" in json.dumps(emitted)
        assert view == original
        assert harness.stored_rows(engine) == before
        assert "private" in before[-1]["content"]


def test_inspector_drops_oversize_section_as_valid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with harness.engine_context(tmp_path, window=32000) as engine:
        engine.ingest([harness.user("Keep source text.")])
        before = harness.stored_rows(engine)
        identity = {f"field-{index}": "x" * 200 for index in range(200)}
        status = engine.get_status()
        status["runtime_identity"] = identity
        monkeypatch.setattr(engine, "get_status", lambda: status)
        encoded = engine.handle_tool_call("lcm_inspect", {"limit": 1})
        response = json.loads(encoded)
        assert len(encoded) <= response["char_limit"]
        assert response["truncated"] is True
        assert response["read_only"] is True
        assert response["session_id"] == engine.current_session_id
        assert "runtime_identity" in response["truncation"]["omitted_top_level_sections"]
        assert response["messages"]["total"] == 1
        assert harness.stored_rows(engine) == before


def test_scope_writer_discovery_handles_interpolated_sql(tmp_path: Path) -> None:
    (tmp_path / "writer.py").write_text('''
def good(conn, value):
    conn.execute(f"INSERT INTO messages(session_id, access_scope) VALUES ({value}, ?)")
def bad(conn, value):
    conn.execute(f"INSERT INTO messages(session_id) VALUES ({value})")
''', encoding="utf-8")
    writers = scope_storage.enumerate_scope_writers(tmp_path)
    indexed = {writer.function: writer for writer in writers}
    assert indexed["good"].populates_access_scope
    assert not indexed["bad"].populates_access_scope
    assert {writer.table for writer in writers} == {"messages"}


def test_externalization_removes_partial_file_on_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config.LCMConfig()

    def fail_fsync(_fd: int) -> None:
        raise OSError("fixture fsync failure")

    externalize.get_large_output_storage_dir(cfg, hermes_home=str(tmp_path), create=True)
    monkeypatch.setattr(externalize.os, "fsync", fail_fsync)
    result = externalize.externalize_ingest_payload("durable candidate", config=cfg, hermes_home=str(tmp_path))
    assert result is None
    storage = externalize.get_large_output_storage_dir(cfg, hermes_home=str(tmp_path), create=False)
    assert not list(storage.iterdir())


@pytest.mark.parametrize("question, operation, kind", [
    ("What is the total of the two costs?", "sum", "missing_operand"),
    ("What is the difference between the two costs?", "difference", "missing_operand"),
    ("How many days between 2024-01-01 and 2024-01-03?", "date_interval", "missing_operand"),
])
def test_preanswer_names_missing_operand_without_fabricating_context(
    tmp_path: Path, question: str, operation: str, kind: str,
) -> None:
    with harness.engine_context(tmp_path) as engine:
        result = preanswer.build_preanswer_evidence(question, engine=engine, enabled=True)
        assert result["context"] is None
        missing = result["decision"]["missing_requirement"]
        assert missing["operation"] == operation
        assert missing["kind"] == kind
        assert missing["current_operands"] == 0


def test_scope_text_recovery_preserves_dynamic_marker() -> None:
    import ast

    # No visitor dispatch reaches BinOp/non-text nodes; exercise that orphan seam directly.
    for expression, expected in [
        ('"INSERT " + "INTO messages"', "INSERT INTO messages"),
        ('f"INSERT INTO messages VALUES ({value})"', "INSERT INTO messages VALUES ({expr})"),
        ('"INSERT " + value', None), ("7", None),
    ]:
        node = ast.parse(expression, mode="eval").body
        assert scope_storage._constant_text(node) == expected


@pytest.mark.parametrize("part, expected", [
    ({"type": "text", "text": "visible"}, True),
    ({"type": "text", "content": {"value": "visible"}}, True),
    ({"type": "text", "value": {"content": "visible"}}, True),
    ({"type": "text", "text": {"value": 1}}, False),
    ({"type": "text", "text": "<think>private</think>"}, False),
])
def test_orphan_visibility_predicate_matches_visible_text(part: dict[str, Any], expected: bool) -> None:
    sanitize = importlib.import_module("hermes_lcm.sanitize")
    # Current source has no caller of this predicate; there is no public entry point.
    assert sanitize._assistant_message_has_visible_content({"role": "assistant", "content": [part]}) is expected


@pytest.mark.parametrize("message, expected", [
    ({"role": "user", "content": ""}, False),
    ({"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]}, False),
    ({"role": "assistant", "content": "<think>private</think>"}, True),
    ({"role": "assistant", "content": "visible"}, False),
])
def test_orphan_active_cleanup_predicate(message: dict[str, Any], expected: bool) -> None:
    sanitize = importlib.import_module("hermes_lcm.sanitize")
    # The only engine caller is itself unreferenced in current source.
    assert sanitize._should_drop_active_assistant_message(message) is expected


def test_bypass_prefix_predicate_counts_only_matching_head(tmp_path: Path) -> None:
    with harness.engine_context(tmp_path, window=32000) as engine:
        view = [harness.user("first"), harness.assistant("answer")]
        engine._remember_lcm_bypass_message_prefix("S", view)
        # Predicate and count wrapper have no production caller beyond each other.
        assert engine._messages_match_lcm_bypass_prefix("S", view)
        assert not engine._messages_match_lcm_bypass_prefix("other", view)
        assert not engine._messages_match_lcm_bypass_prefix("S", [harness.user("different")])


def test_generated_payload_ref_is_recognized_without_file(tmp_path: Path) -> None:
    cfg = config.LCMConfig()
    result = externalize.externalize_ingest_payload("fixture payload", config=cfg, hermes_home=str(tmp_path))
    assert result is not None
    ref = result["path"].name
    result["path"].unlink()
    assert externalize.is_generated_payload_ref_name(ref)
    assert not externalize.is_generated_payload_ref_name("arbitrary.json")
    assert not externalize.is_generated_payload_ref_name("")


def test_evidence_pack_falls_back_without_partial_computation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = importlib.import_module("hermes_lcm.evidence_pack")
    with harness.engine_context(tmp_path, window=32000) as engine:
        text = "The invoice total is 20 USD."
        engine.ingest([harness.user(text)])
        sid = harness.stored_rows(engine)[0]["store_id"]
        monkeypatch.setattr(pack, "RESPONSE_CHAR_CAP", 800)
        result = json.loads(engine.handle_tool_call("lcm_evidence_pack", {
            "question": "What is the total of the one invoice?",
            "baseline_refs": [{"exact_ref": f"lcm:{sid}:0-{len(text)}", "quote": text, "value": 20, "unit": "USD"}],
        }))
        assert result["status"] == "fallback"
        assert result["evidence"] == []
        assert result["computation"] is None
        assert result["selection"] is None
        assert result["completeness"]["reason_code"] == "response_char_cap_exceeded"
        assert result["retrieval"]["status"] == "truncated"
        assert harness.stored_rows(engine)[0]["content"] == text


def test_recall_override_has_its_own_provider_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    configurations: list[tuple[str, str]] = []
    provider = SimpleNamespace(provider_id="fixture", model_id="fixture", embed_query=lambda _text: [1.0, 0.0])

    def resolve(cfg: Any) -> Any:
        configurations.append((cfg.embedding_provider, cfg.embedding_model))
        return provider

    with harness.engine_context(tmp_path, window=32000, embeddings_enabled=True, embedding_provider="original", embedding_model="fixture") as engine:
        engine._store.append("old", harness.user("remember the invoice"))
        monkeypatch.setattr(tools, "resolve_provider", resolve)
        interactive_cache = ("interactive", object())
        engine._lcm_embedding_provider_cache = interactive_cache
        for _ in range(2):
            response = json.loads(tools.lcm_recall({"query": "invoice", "include": "summaries"}, engine=engine, provider_override="fixture"))
            assert "error" not in response
        assert configurations == [("fixture", "fixture")]
        assert engine._config.embedding_provider == "original"
        assert engine._lcm_embedding_provider_cache is interactive_cache


def test_unreferenced_standalone_chunk_archive_keeps_rows(tmp_path: Path) -> None:
    vectors = importlib.import_module("hermes_lcm.vector_store")
    with harness.engine_context(tmp_path, window=32000, embeddings_enabled=True) as engine:
        engine.ingest([harness.user("Keep durable text.")])
        sid = harness.stored_rows(engine)[0]["store_id"]
        store = vectors.VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile("fixture", "fixture", 2)
            identity = store.capture_identity("fixture", provider="fixture")
            store.record_chunk_embedding(
                "chunk", "fixture", [1.0, 0.0], store_id=sid, chunk_index=0,
                char_start=0, char_end=18, token_estimate=5, identity=identity,
            )
            before = harness.stored_rows(engine)
            # The production caller always supplies a connection; the standalone branch has no caller.
            engine._archive_chunks_for_messages([sid])
            archived = store.connection.execute("SELECT archived FROM lcm_chunk_meta WHERE chunk_id='chunk'").fetchone()
            assert archived[0] == 1
            assert harness.stored_rows(engine) == before
        finally:
            store.close()
