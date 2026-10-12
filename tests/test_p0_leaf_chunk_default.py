"""v0.27.0 phase 0: larger default leaves with a fixed summary target."""

from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def test_p0_leaf_chunk_and_summary_defaults():
    config = LCMConfig()
    assert config.leaf_chunk_tokens == 24_000
    assert config.leaf_target_max_tokens == 2_000
    engine = SimpleNamespace(_config=config)
    assert [LCMEngine._leaf_target_tokens(engine, n) for n in (8_000, 20_000, 100_000)] == [2_000] * 3


@pytest.fixture
def threshold_sweep(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    config = LCMConfig(database_path=str(home / "lcm.db"))
    engine = LCMEngine(config=config, hermes_home=str(home))
    engine.on_session_start("phase-0", platform="cli", context_length=16_000)
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"FACT-{i} " + "alpha beta gamma delta " * 44}
        for i in range(48)
    ] + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i}"} for i in range(24)]

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        return chunk, count_messages_tokens(chunk), "Earlier facts.\nExpand for details about: facts", 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    try:
        yield engine, messages
    finally:
        engine.shutdown()


def test_p0_threshold_sweep_compacts_less_than_one_default_chunk(threshold_sweep):
    engine, messages = threshold_sweep
    boundary = engine._fresh_tail_boundary(messages)
    raw_tokens = count_messages_tokens(messages[:boundary.start])
    assert 11_000 < raw_tokens < 13_000
    assert raw_tokens < engine._config.leaf_chunk_tokens
    assert engine._config.threshold_full_sweep_enabled is True
    current_tokens = count_messages_tokens(messages)
    assert current_tokens > engine.threshold_tokens
    engine.compress(messages, current_tokens=current_tokens)
    assert any(n.depth == 0 for n in engine._dag.get_session_nodes("phase-0"))
    assert engine.get_status()["threshold_full_sweep"]["leaf_passes"] >= 1


def test_p0_sweep_prefix_target_follows_default_leaf_chunk(threshold_sweep):
    engine, messages = threshold_sweep
    assert engine._config.summary_prefix_target_tokens == 0
    engine.compress(messages, current_tokens=count_messages_tokens(messages))
    telemetry = engine.get_status()["threshold_full_sweep"]
    assert telemetry["summary_prefix_target_tokens"] == engine._config.leaf_chunk_tokens == 24_000
