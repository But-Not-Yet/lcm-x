"""#1221: the default prefix bound budgets summaries without dropping the objective."""

import json

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX as PREFIX


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(engine_module, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)

    def no_model_calls(**kwargs):
        pytest.fail("assembly and survival fit must not call a model")

    monkeypatch.setattr(engine_module, "summarize_with_escalation", no_model_calls)
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
                         hermes_home=str(tmp_path / "home"))
    instance.on_session_start("issue-1221", platform="cli", context_length=32_768)
    try:
        yield instance
    finally:
        instance.shutdown()


def objective_scenario(engine):
    """One leaf covers the objective and other rows; the 20-row tail is tool-only work."""
    system = {"role": "system", "content": "Synthetic system prompt."}
    objective = {"role": "user", "content": "Keep this objective verbatim. " + "plan " * 9600, "timestamp": 3.0}
    history = [
        {"role": "user", "content": "Earlier request.", "timestamp": 1.0},
        {"role": "assistant", "content": "Earlier answer.", "timestamp": 2.0},
        objective,
        {"role": "assistant", "content": "Starting the requested work.", "timestamp": 4.0},
    ]
    tail = [row for i in range(10) for row in (
        {"role": "assistant", "content": "Reading.", "timestamp": 5.0 + 2 * i, "tool_calls": [
            {"id": f"call-{i}", "type": "function", "function": {"name": "probe", "arguments": "{}"}}]},
        {"role": "tool", "content": f"result {i}: " + "data " * 1500, "tool_call_id": f"call-{i}",
         "name": "probe", "tool_name": "probe", "timestamp": 6.0 + 2 * i},
    )]
    source = [system, *history, *tail]
    engine.ingest(source)
    rows = engine._store.get_session_messages(engine.current_session_id)
    by_identity = {(row["role"], row["content"]): row for row in rows}
    history_rows = [by_identity[(message["role"], message["content"])] for message in history]
    node_id = engine._dag.add_node(SummaryNode(
        session_id=engine.current_session_id, depth=0, summary="Earlier work. " + "abcd" * 1990,
        token_count=2000, source_token_count=tokens.count_messages_tokens(history),
        source_ids=[row["store_id"] for row in history_rows], source_type="messages",
        expand_hint="synthetic history", created_at=30.0,
    ))
    engine._last_compacted_store_id = history_rows[-1]["store_id"]
    engine._pending_context_anchor_messages = source[1:]
    assert 12_000 <= tokens.count_message_tokens(objective) < 12_100
    assert 18_750 <= tokens.count_messages_tokens(tail) < 19_100
    assert len(engine._dag.get_node(node_id).source_ids) > 1
    assert by_identity[("user", objective["content"])]["store_id"] in engine._dag.get_node(node_id).source_ids
    return system, tail, source, objective


def test_large_objective_survives_default_bound_and_fits(engine):
    system, tail, source, objective = objective_scenario(engine)
    assembled = engine._assemble_context(system, tail, include_lcm_note=False)
    fitted = engine._survival_fit(source, assembled, tokens.count_messages_tokens(source), "issue-1221")
    assert sum(PREFIX in row.get("content", "") for row in fitted) == 1
    assert sum(objective["content"] in row.get("content", "") for row in fitted) == 1
    budget = engine._survival_fit_budget(source, tokens.count_messages_tokens(source))
    assert budget == 27_852
    assert engine._survival_measure(fitted) <= budget
    assert engine._last_survival_fit["uncovered_rows"] == 0


def test_128k_assembly_is_byte_identical_to_unbounded_rc5_path(engine):
    system, tail, source, _ = objective_scenario(engine)
    engine.context_length = 131_072
    assembled = engine._assemble_context(system, tail, include_lcm_note=False, persist=False)
    fitted = engine._survival_fit(source, assembled, tokens.count_messages_tokens(source), "issue-1221")
    engine.context_length = 0  # rc5 has no default bound on an unknown window.
    uncapped = engine._assemble_context(system, tail, include_lcm_note=False, persist=False)
    assert json.dumps(fitted, ensure_ascii=False).encode() == json.dumps(uncapped, ensure_ascii=False).encode()
