"""#1220: short stored tool rows can fit under pressure without losing their source."""

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS", "12000")
    monkeypatch.setenv("LCM_THRESHOLD_FULL_SWEEP_ENABLED", "true")
    monkeypatch.setenv("LCM_LEAF_CHUNK_TOKENS", "8000")
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(engine_module, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: ("Earlier summary.", 1))
    config = LCMConfig.from_env()
    config.database_path = str(tmp_path / "lcm.db")
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    instance.on_session_start("issue-1220", platform="cli", context_length=32_768)

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        return chunk, tokens.count_messages_tokens(chunk), "Earlier work.\nExpand for details about: work", 1, 0

    monkeypatch.setattr(instance, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    try:
        yield instance
    finally:
        instance.shutdown()


def _batch():
    return [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "earlier question", "timestamp": 1.0},
        {"role": "assistant", "content": "earlier answer", "timestamp": 2.0},
        {"role": "user", "content": "run the batch please", "timestamp": 3.0},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "probe", "arguments": f'{{"i": {i}}}'}}
            for i in range(128)
        ]},
        *[{"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i:04d}" + " x" * 597,
           "name": "probe", "tool_name": "probe", "timestamp": 4.0} for i in range(128)],
    ]


@pytest.mark.parametrize("reingest", [False, True], ids=["compress", "reingest"])
def test_parallel_short_results_fit_and_remain_stored(engine, reingest):
    messages = _batch()
    assert all(len(row["content"]) == 1199 for row in messages if row["role"] == "tool")
    current = tokens.count_messages_tokens(messages)
    fitted = engine.compress(messages, current_tokens=current)
    budget = engine._survival_fit_budget(messages, current)
    assert engine._survival_measure(fitted) <= budget
    marks = [row for row in fitted if row.get("role") == "tool"
             and survival_fit._PROJECTED_RE.fullmatch(row.get("content", ""))]
    assert marks
    assert not any(survival_fit._PROJECTED_RE.fullmatch(row.get("content", ""))
                   for row in fitted if row.get("role") == "user")
    assert any(messages[3]["content"] in row.get("content", "") for row in fitted)
    stored = engine._store.get_session_messages(engine.current_session_id)
    assert [(row["role"], row["content"]) for row in stored] == [
        (row["role"], row["content"]) for row in messages]
    if reingest:
        expected = {row["tool_call_id"]: row["store_id"] for row in stored if row["role"] == "tool"}
        for row in marks:
            source = engine._survival_projection_source(row, "tool", row["content"])
            assert source is not None and source["store_id"] == expected[row["tool_call_id"]]
        engine.ingest(fitted)
        assert engine._store.get_session_messages(engine.current_session_id) == stored


@pytest.mark.parametrize("role,content,mark_only", [
    ("tool", "s" * 1199, True),
    ("tool", "界" * 1201, True),  # The head plus mark would grow.
    ("tool", "s" * 2400, True),  # Pass 2 must use the original count after pass 1 shrinks it.
    ("user", "s" * 1199, False),
    ("tool", "s" * 2401, False),
    ("assistant", "s" * 1100, True),
    ("assistant", "s" * 2400, True),
    ("assistant", "s" * 2401, False),
    ("assistant", "short", False),
])
def test_limit_zero_uses_legacy_marks_only_for_eligible_tool_and_assistant_rows(engine, role, content, mark_only):
    message = {"role": role, "content": content, "timestamp": 5.0}
    if role == "tool":
        message.update(tool_call_id="call", name="probe", tool_name="probe")
    elif role == "assistant":
        message["tool_calls"] = [{"id": "call", "type": "function",
                                  "function": {"name": "probe", "arguments": "{}"}}]
    store_id = engine._store.append(engine.current_session_id, message)
    row = engine._store.get(store_id)
    source_tokens = tokens.count_message_tokens(message)
    if role == "assistant" and content == "short":
        assert source_tokens < 256
    else:
        assert source_tokens >= 256
    head = engine._survival_projected_fields(row, source_tokens, survival_fit._HEAD, survival_fit._TAIL)
    if content == "界" * 1201:
        assert tokens.count_message_tokens({**message, **head}) > source_tokens
    projected = engine._survival_projection([message], {id(message): store_id}, 0)[0]
    if mark_only:
        legacy = engine._survival_projected_fields(row, source_tokens, survival_fit._HEAD,
                                                   survival_fit._TAIL, legacy=True)
        assert projected["content"] == legacy["content"]
        assert survival_fit._PROJECTED_RE.fullmatch(projected["content"])
        if role == "assistant":
            assert projected["tool_calls"] == message["tool_calls"]
            assert tokens.count_message_tokens(projected) < tokens.count_message_tokens({**message, **head})
        source = engine._survival_projection_source(projected, role, projected["content"])
        assert source is not None and source["store_id"] == store_id
    elif role == "user" or source_tokens < 256:
        assert projected is message
    else:
        assert projected["content"] == head["content"]


def test_sequential_short_assistants_fit_during_summary_outage(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    monkeypatch.setattr(tokens, "_encoder", None)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    calls = []

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append(model)
        raise RuntimeError("synthetic summary route outage")

    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), temporal_rollups_enabled=False,
        reserve_tokens_floor=0, leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
        large_output_externalization_enabled=True, large_output_active_replay_stubbing_enabled=True,
        large_output_externalization_threshold_chars=12000,
        large_output_active_replay_stub_threshold_tokens=10000,
        large_output_active_replay_stub_aged_threshold_tokens=2000)
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    instance.on_session_start("synthetic-probe", platform="cli", context_length=32_768)
    prose = ("I will inspect the next record and compare its details with the earlier evidence. "
             "The result should tell us which step to take next and which facts still need checking. "
             "I will keep the source intact and report what the check actually shows. ")
    text = (prose * (1100 // len(prose) + 1))[:1100]
    messages = [{"role": "user", "content": "Please check these records.", "timestamp": 1.0}]
    for i in range(113):
        messages.extend([
            {"role": "assistant", "content": text, "timestamp": 2.0 + 2 * i,
             "tool_calls": [{"id": f"c{i}", "type": "function",
                             "function": {"name": "probe", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok", "tool_call_id": f"c{i}",
             "name": "probe", "tool_name": "probe", "timestamp": 3.0 + 2 * i},
        ])
    try:
        instance.ingest(messages)
        before = instance._store.get_session_messages(instance.current_session_id)
        assert [(row["role"], row["content"]) for row in before] == [
            (row["role"], row["content"]) for row in messages]
        current = tokens.count_messages_tokens(messages)
        assert current > instance.context_length
        fitted = instance.compress(messages, current_tokens=current)
        assert len(calls) == 2 and not instance._summary_route_available()
        budget = instance._survival_fit_budget(messages, current)
        assert instance._survival_measure(fitted) <= budget
        assert any(row["role"] == "assistant" and survival_fit._PROJECTED_RE.fullmatch(row.get("content", ""))
                   for row in fitted)
        assert any(row["role"] == "user" and row["content"] == before[0]["content"] for row in fitted)
        assert instance._store.get_session_messages(instance.current_session_id) == before
        instance.ingest(fitted)
        assert instance._store.get_session_messages(instance.current_session_id) == before
    finally:
        instance.shutdown()
