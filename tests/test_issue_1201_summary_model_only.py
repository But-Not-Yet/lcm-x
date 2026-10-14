"""#1201: display-only summaries, visible carriers, and auxiliary prompt isolation."""

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm import assertion_extraction, escalation, extraction, host_evidence, selective_compiler, tools
from hermes_lcm import engine as engine_module
from hermes_lcm.assertion_store import SourceSnapshot
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_analysis import _merge_adjacent_assistant_messages
from hermes_lcm.reconcile import _emission_identity, _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from hermes_lcm.tokens import count_messages_tokens
from tests.test_overflow_reserves_sanitizer_stubs import (
    OBJECTIVE, add_summaries, assemble, assistant_calls, full_cap, result,
)


def _host(monkeypatch, key="model_only"):
    module = ModuleType("agent.context_compressor")
    if key is not None:
        module.MODEL_ONLY_DISPLAY_METADATA_KEY = key
    monkeypatch.setitem(sys.modules, "agent.context_compressor", module)
    resolver = getattr(engine_module, "_model_only_display_metadata_key", None)
    if resolver is not None:
        resolver.cache_clear()


@pytest.fixture(autouse=True)
def host(monkeypatch):
    _host(monkeypatch)
    yield
    resolver = getattr(engine_module, "_model_only_display_metadata_key", None)
    if resolver is not None:
        resolver.cache_clear()


@pytest.fixture
def engine(tmp_path, monkeypatch):
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "l.db"), fresh_tail_count=4,
        leaf_chunk_tokens=100, extraction_enabled=False,
        temporal_rollups_enabled=False,
    ), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("issue-1201", platform="cli", context_length=200_000)
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda *a, **k: ("Earlier work completed.\nExpand for details about: work", 1))
    yield instance
    instance.shutdown()


def _seed(engine):
    engine._dag.add_node(SummaryNode(
        session_id=engine._session_id, depth=0, summary="Earlier work completed.",
        token_count=10, source_token_count=100, source_ids=[], source_type="messages",
        created_at=1, earliest_at=1, latest_at=1, expand_hint="work",
    ))


def _history():
    return [{"role": "system", "content": "System instructions."}] + [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"Human turn {i}. " + "detail " * 60}
        for i in range(20)
    ]


def _compress(engine, messages):
    return engine.compress(messages, current_tokens=count_messages_tokens(messages), force=True)


def _summary_rows(output):
    return [row for row in output if " Summary (d" in str(row.get("content"))
            and "System instructions." not in str(row.get("content"))]


@pytest.mark.parametrize("key", ["model_only", "host_model_only"])
def test_r2_survival_fit_stamps_standalone_summary(engine, monkeypatch, key):
    _host(monkeypatch, key)
    _seed(engine)
    tail = [{"role": "user", "content": "Older request. " + "detail " * 200},
            {"role": "assistant", "content": "Older reply. " + "detail " * 200},
            {"role": "user", "content": "Newest human request."},
            {"role": "assistant", "content": "Newest human reply."}]
    engine.ingest(tail)
    incoming = engine._assemble_context(None, tail)
    remainder = engine._generated_context_carrier_remainder(incoming[0])
    assert remainder == tail[0]["content"]
    summary = incoming[0]["content"][:engine._verified_lcm_summary_prefix_end(incoming[0]["content"])]
    budget = count_messages_tokens([{"role": "user", "content": summary}, *tail[-2:]])
    engine._ingest_cursor = len(incoming)
    engine._ingest_cursor_needs_reconcile = False
    fitted = engine._survival_fit(incoming, incoming, 0, "recovery_attempt:test", request_cap=budget)
    assert fitted is not incoming and fitted[0]["content"] == summary
    assert fitted[0].get("display_metadata") == {key: True}
    assert fitted[1:] == tail[-2:]
    assert all(key not in row.get("display_metadata", {}) for row in fitted[1:])
    assert "display_metadata" not in incoming[0]  # a carrier holds human text
    metadata = {"origin": "survival", key: False}
    row = {"role": "user", "content": summary, "display_metadata": metadata}
    engine._survival_summary_identity(row, summary, None, "survival_summary")
    assert row["display_metadata"] == {"origin": "survival", key: True}
    assert metadata == {"origin": "survival", key: False}
    carrier = {"role": "user", "content": summary + "\n\nHuman request."}
    engine._survival_summary_identity(carrier, summary, None, "carrier")
    assert "display_metadata" not in carrier
    _host(monkeypatch, None)
    unsupported = {"role": "user", "content": summary}
    engine._survival_summary_identity(unsupported, summary, None, "survival_summary")
    assert "display_metadata" not in unsupported


@pytest.mark.parametrize("origin", [False, True], ids=["empty-metadata", "keep-other-metadata"])
@pytest.mark.parametrize("path", ["assistant-merge", "prepend-text", "prepend-list", "prepend-dict"])
def test_r2_human_text_merge_removes_stale_marker(origin, path):
    metadata = {"model_only": True, **({"origin": "host"} if origin else {})}
    row = {"role": "assistant", "content": "Earlier generated context.", "display_metadata": metadata}
    if path == "assistant-merge":
        human = {"role": "assistant", "content": "Human-visible reply."}
        generated = [row]
        merged = _merge_adjacent_assistant_messages([row, human], generated_rows=generated)[0]
        assert generated == []
    else:
        row["content"] = ("Human-visible reply." if path == "prepend-text" else
                          [{"type": "text", "text": "Human-visible reply."}] if path == "prepend-list" else
                          {"type": "text", "text": "Human-visible reply."})
        merged = LCMEngine._prepend_generated_context_to_message(row, "Generated context.")
    assert "Human-visible reply." in str(merged["content"])
    assert "model_only" not in merged.get("display_metadata", {})
    assert merged.get("display_metadata") == ({"origin": "host"} if origin else None)
    assert row["display_metadata"] is metadata and metadata["model_only"] is True
    if path == "assistant-merge":
        other = {"role": "assistant", "content": "More generated context."}
        generated = [row, other]
        merged = _merge_adjacent_assistant_messages([row, other], generated_rows=generated)[0]
        assert generated == [merged] and merged["display_metadata"]["model_only"] is True


def test_r2_over_cap_trim_tracks_generated_summary(engine, monkeypatch, caplog):
    nodes = add_summaries(engine)
    tail = [assistant_calls("call-0", "call-1"), result("call-0"), result("call-1")]
    cap = full_cap(engine, nodes, tail)
    original = LCMEngine._sanitize_active_context_messages

    def inflating(self, messages, **options):
        cleaned = original(self, messages, **options)
        if options.get("insert_missing_tool_stubs", True) and any(
            _PRESERVED_OBJECTIVE_CONTEXT_PREFIX in str(row.get("content", "")) for row in cleaned
        ):
            cleaned = [*cleaned, {"role": "user", "content": "unbudgeted " * 400}]
        return cleaned

    monkeypatch.setattr(LCMEngine, "_sanitize_active_context_messages", inflating)
    output = assemble(engine, tail, cap)
    assert OBJECTIVE not in "\n".join(str(row.get("content")) for row in output)
    assert "dropping the preserved user objective" in caplog.text
    summaries = [row for row in output if "orchard" in str(row.get("content"))]
    assert summaries and all(row.get("display_metadata") == {"model_only": True} for row in summaries)


def test_t1_compaction_marks_only_standalone_summary(engine):
    messages = _history()
    before = deepcopy(messages)
    output = _compress(engine, messages)
    summaries = _summary_rows(output)
    assert summaries
    assert all(row.get("display_metadata") == {"model_only": True} for row in summaries)
    assert all("display_metadata" not in row for row in output if row not in summaries)
    assert messages == before


@pytest.mark.parametrize("missing", ["attribute", "module"])
def test_t2_unsupported_host_full_output_unchanged(engine, monkeypatch, missing):
    _seed(engine)
    tail = [{"role": "user", "content": "A real request."}]
    _host(monkeypatch, None)
    if missing == "module":
        monkeypatch.setitem(sys.modules, "agent.context_compressor", None)
    # This is the complete base assembly, including role and summary formatting.
    output = engine._assemble_context(None, tail, persist=False)
    assert output == [
        {"role": "user", "content": "[Recent Summary (d0, node 1)]\nEarlier work completed.\n[Expand for details: work]"},
        *tail,
    ]


@pytest.mark.parametrize("shape", ["retained-fold", "retained-structured", "user-carrier"])
def test_t3_human_carriers_stay_visible(engine, shape):
    _seed(engine)
    retained = None
    if shape.startswith("retained"):
        retained = {"role": "user", "content": "The sole user objective."}
        tail = [{"role": "assistant", "content": "A real assistant reply."}]
        if shape == "retained-structured":
            tail[0]["content"] = [{"type": "text", "text": tail[0]["content"]}]
        engine.ingest([retained, *tail])
    else:
        tail = [{"role": "user", "content": "A real earlier request."},
                {"role": "assistant", "content": "A real assistant reply."},
                {"role": "user", "content": "A real latest request."}]
        engine.ingest(tail)
    output = engine._assemble_context(None, tail, retained_user_message=retained)
    assert any(" Summary (d" in str(row["content"]) and "A real" in str(row["content"])
               for row in output)
    assert all(not row.get("display_metadata", {}).get("model_only") for row in output)


@pytest.mark.parametrize("site", ["recall", "generated-context", "anchor", "assistant-merge"])
@pytest.mark.parametrize("cap", [None, 160], ids=["uncapped", "fit-candidates"])
def test_t1_other_standalone_consumers(engine, monkeypatch, site, cap):
    tail = [{"role": "user", "content": "A real request."}]
    retained = None
    if site in {"recall", "assistant-merge"}:
        _seed(engine)
        monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
            "role": "user", "content": "<relevant-memories>Retrieved context.</relevant-memories>",
            "display_metadata": {"origin": "recall"},
        })
    elif site == "generated-context":
        _seed(engine)
        retained = {"role": "user", "content": "The sole user objective."}
        tail = [{"role": "user", "content": "Another real request."}]
    else:
        monkeypatch.setattr(engine, "_latest_user_context_anchor_candidates",
                            lambda *a: [f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nAn omitted objective."])
    if site == "assistant-merge":
        tail = [{"role": "assistant", "content": "A real assistant reply."}]
        monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
            "role": "assistant", "content": "Retrieved context."})
    measured = []
    count = engine_module.count_message_tokens
    def measure(row):
        measured.append(deepcopy(row))
        return count(row)
    monkeypatch.setattr(engine_module, "count_message_tokens", measure)
    output = engine._assemble_context(None, tail, retained_user_message=retained, assembly_cap_override=cap)
    generated = [row for row in output if not any(
        text in str(row["content"]) for text in ("A real", "Another real", "sole user"))]
    assert generated
    assert all(row.get("display_metadata", {}).get("model_only") is True for row in generated)
    assert all(not row.get("display_metadata", {}).get("model_only")
               for row in output if "A real" in str(row["content"]))
    assert all(not row.get("display_metadata", {}).get("model_only") for row in measured)


def test_t1_cleanup_copy_of_standalone_recall_is_marked(engine, monkeypatch):
    _seed(engine)
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
        "role": "assistant", "content": "<think>Internal reasoning.</think>Retrieved context."})
    output = engine._assemble_context(None, [{"role": "user", "content": "A real request."}])
    recall = next(row for row in output if row["content"] == "Retrieved context.")
    assert recall["display_metadata"] == {"model_only": True}


def test_t3_same_text_host_cleanup_copy_stays_visible(engine, monkeypatch):
    _seed(engine)
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: {
        "role": "assistant", "content": "Retrieved context."})
    tail = [{"role": "assistant", "content": "<think>Internal reasoning.</think>Retrieved context.",
             "finish_reason": "incomplete"}, {"role": "user", "content": "A real request."}]
    output = engine._assemble_context(None, tail)
    human = next(row for row in output if row.get("finish_reason") == "incomplete")
    recall = next(row for row in output if row["content"] == "Retrieved context." and row is not human)
    assert "display_metadata" not in human
    assert recall["display_metadata"] == {"model_only": True}


def test_t4_host_display_projection_keeps_each_human_turn_once(engine):
    originals = _history()
    output = _compress(engine, originals)
    output_contents = {str(row["content"]) for row in output}
    compacted = [row for row in originals if str(row["content"]) not in output_contents]
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE messages(content TEXT, active INT, compacted INT, display_metadata TEXT)")
        for active, rows in [(0, compacted), (1, output)]:
            db.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", [
                (str(row["content"]), active, 1 - active, json.dumps(row.get("display_metadata")))
                for row in rows])
        visible = [row[0] for row in db.execute(
            "SELECT content FROM messages WHERE (active=1 OR compacted=1) "
            "AND COALESCE(json_extract(display_metadata, '$.model_only'),0)=0")]
    humans = [row["content"] for row in originals if row["role"] != "system"]
    assert all(Counter(visible)[content] == 1 for content in humans)
    assert not any(" Summary (d" in content for content in visible)


def test_t5_metadata_does_not_change_identity_or_reconciliation(engine, tmp_path, monkeypatch):
    originals = _history()
    marked = _compress(engine, originals)
    assert _summary_rows(marked)[0].get("display_metadata") == {"model_only": True}
    unmarked = deepcopy(marked)
    for row in unmarked:
        row.pop("display_metadata", None)
    assert [_emission_identity(row) for row in marked] == [_emission_identity(row) for row in unmarked]
    # Two cold engines consume the same durable first-compaction proof independently.
    engine.shutdown()
    results = []
    for index, messages in enumerate([marked, unmarked]):
        path = tmp_path / f"replay-{index}.db"
        with sqlite3.connect(engine._config.database_path) as source, sqlite3.connect(path) as target:
            source.backup(target)
        replay = LCMEngine(config=LCMConfig(database_path=str(path), fresh_tail_count=4,
                           leaf_chunk_tokens=100, temporal_rollups_enabled=False),
                           hermes_home=str(tmp_path / f"home-{index}"))
        replay.on_session_start("issue-1201", platform="cli", context_length=200_000)
        try:
            output = _compress(replay, deepcopy(messages))
            results.append((replay._last_ingest_reconciliation, replay._store.get_session_count("issue-1201"),
                            [_emission_identity(row) for row in output]))
        finally:
            replay.shutdown()
    assert results[0] == results[1]


def test_t6_existing_metadata_and_host_constant(engine, monkeypatch):
    _host(monkeypatch, "host_model_only")
    _seed(engine)
    metadata = {"origin": "recall", "host_model_only": False}
    recall = {"role": "user", "content": "Retrieved context.", "display_metadata": metadata}
    monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a: recall.copy())
    tail = [{"role": "user", "content": "A real request.", "display_metadata": {"origin": "human"}}]
    output = engine._assemble_context(None, tail)
    assert next(row for row in output if row["content"] == recall["content"])["display_metadata"] == {
        "origin": "recall", "host_model_only": True}
    assert output[-1]["display_metadata"] == tail[0]["display_metadata"]
    assert metadata == {"origin": "recall", "host_model_only": False}


@pytest.mark.parametrize("builder", ["summary-l1", "summary-l2", "extraction", "assertion",
                                    "host-evidence", "selective", "expansion"])
def test_t7_auxiliary_prompt_builders_never_forward_live_rows(engine, monkeypatch, tmp_path, builder):
    rows = [{"role": "user", "content": "A synthetic decision to preserve.",
             "display_metadata": {"model_only": True}, "display_kind": "summary"}]
    before = deepcopy(rows)
    calls = []
    module = ModuleType("agent.auxiliary_client")
    def call_llm(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    serialized = engine._serialize_messages(rows)
    # The public serializers take rows; the downstream auxiliary seams take text/envelopes.
    if builder.startswith("summary"):
        prompt = (escalation._build_l1_prompt(serialized, 100, 0) if builder == "summary-l1"
                  else escalation._build_l2_prompt(serialized, 100))
        escalation._call_llm_for_summary(prompt, 100)
    elif builder == "extraction":
        extraction.extract_before_compaction(serialized, str(tmp_path / "notes"))
    elif builder == "assertion":
        content = rows[0]["content"]
        snapshot = SourceSnapshot(1, "issue-1201", "cli", "user", content, 1,
                                  hashlib.sha256(content.encode()).hexdigest())
        prompt = assertion_extraction.build_structured_assertion_prompt(snapshot, SimpleNamespace(query_assertions=lambda **k: []))
        assertion_extraction._call_structured_assertion_llm(prompt, "", 1)
    elif builder == "host-evidence":
        host_evidence.call_auxiliary_selector({"question": "What was decided?", "baseline_evidence": rows})
    elif builder == "selective":
        selective_compiler.call_selective_auxiliary_selector({
            "status": "selector_required", "prompt": selective_compiler._prompt(
                {"operation": "latest_state", "facet": "decision"},
                [{**rows[0], "handle": "h1", "quote": rows[0]["content"], "date": None}])})
    else:
        tools._synthesize_expansion_answer(prompt="What was decided?", context_blocks=rows,
                                          model="", max_tokens=100, timeout=1)
    assert len(calls) == 1
    for outgoing in calls[0]:
        assert set(outgoing) == {"role", "content"}
        assert all(outgoing is not source and outgoing != source for source in rows)
    assert rows == before


def test_t8_second_compress_uses_public_row_noop_shortcut(engine, monkeypatch):
    output = _compress(engine, _history())
    assert _summary_rows(output)[0].get("display_metadata") == {"model_only": True}
    compared = []
    candidates = []
    compress_impl = engine._compress_impl
    def capture(*args, **kwargs):
        candidate = compress_impl(*args, **kwargs)
        candidates.append(candidate)
        return candidate
    monkeypatch.setattr(engine, "_compress_impl", capture)
    public_row = engine._public_compression_row
    def record(row):
        compared.append(row)
        return public_row(row)
    monkeypatch.setattr(engine, "_public_compression_row", record)
    second = _compress(engine, output)
    assert second is output
    assert len(candidates) == 1 and candidates[0] is not output
    assert len(compared) == 2 * len(output)
    assert all(public_row(left) == public_row(right) for left, right in zip(compared[::2], compared[1::2]))
    assert _summary_rows(second)[0]["display_metadata"] == {"model_only": True}
