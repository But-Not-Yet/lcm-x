"""A first-sight stub is not host overhead: the #930 assembly site must not record its savings as overhead (#1012 F4
keeps the peak for the whole conversation)."""

from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_671_stub_first_exit import (  # noqa: F401
    THRESHOLD,
    exit_engine,
    make_engine,
    payload_of,
    run,
    summaries,
    text_view,
    tool_pair,
)

HOST = 1_000  # system prompt and tool schemas the host adds on top of the list


def crossing_turn(make_engine, summaries, caplog, tool_tokens):
    engine = exit_engine(
        make_engine,
        threshold_full_sweep_enabled=False,
        large_output_active_replay_stub_threshold_tokens=2_000,
    )
    first = run(engine, text_view(), caplog)
    assert engine.last_compression_status == "compacted"  # a summary prefix: the #930 block runs
    calls = len(summaries)
    view = first + [
        {"role": "user", "content": "read the big file"},
        *tool_pair("big", payload_of(tool_tokens, "zq")),
        {"role": "user", "content": "latest ask"},
        {"role": "assistant", "content": "latest answer"},
    ]
    current = count_messages_tokens(view) + HOST
    engine.should_compress(current)
    run(engine, view, caplog, current_tokens=current)
    return engine, current, len(summaries) - calls


def test_a_first_sight_stub_is_not_recorded_as_host_overhead(make_engine, summaries, caplog):
    engine, _, _ = crossing_turn(make_engine, summaries, caplog, tool_tokens=5_000)

    key, peak = engine._survival_overhead_observation
    assert key == engine._conversation_id
    assert peak <= HOST + 100, peak  # the host's own overhead, not the stubbed tool output


def test_the_stub_first_exit_takes_a_crossing_turn_with_a_large_new_tool_output(make_engine, summaries, caplog):
    engine, current, new_calls = crossing_turn(make_engine, summaries, caplog, tool_tokens=9_000)
    assert current >= THRESHOLD  # an automatic threshold pass

    assert engine._last_stub_first_exit is not None  # the stubbed list fits: no leaf, no model call
    assert new_calls == 0
    assert engine._survival_overhead_observation[1] <= HOST + 100
