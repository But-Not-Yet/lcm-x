"""Determinism guard for the assembly path (fix B, part 2).

Same state -> same bytes: two consecutive `_assemble_context(persist=False)`
runs over an unchanged store must be byte-identical.  Created-at ties among
same-depth uncondensed nodes must not make the selection order flip between
runs (the SQL tie-breaker is `node_id DESC`; this test pins the outcome).

This is a guard test: it locks the current, intended determinism of the
assembly selection, so a future ordering change that introduces instability
fails loudly here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

from hermes_lcm.config import LCMConfig


def _make_engine(tmp_path: Path, **overrides):
    # Same stub pattern as test_reconcile_tool_scaffold_excerpt.py (engine.py
    # imports agent.context_engine at module level).
    if "agent.context_engine" not in sys.modules:
        _agent_mod = ModuleType("agent")
        _agent_mod.__path__ = []
        _ce_mod = ModuleType("agent.context_engine")

        class _StubContextEngine:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.last_prompt_tokens = 0

            def get_status(self):
                return {}

        _ce_mod.ContextEngine = _StubContextEngine
        sys.modules["agent"] = _agent_mod
        sys.modules["agent.context_engine"] = _ce_mod

    _existing = sys.modules.get("hermes_lcm.engine")
    if _existing is not None and not hasattr(_existing, "LCMEngine"):
        sys.modules.pop("hermes_lcm.engine", None)
    from hermes_lcm.engine import LCMEngine

    defaults = dict(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=64,
    )
    defaults.update(overrides)
    config = LCMConfig(**defaults)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("assemble-det", platform="cli", context_length=1000000)
    return engine


def _seed(engine):
    from hermes_lcm.dag import SummaryNode

    sid = "assemble-det"
    engine._ingest_messages([
        {"role": "user", "content": "hello", "timestamp": 100.0},
        {"role": "assistant", "content": "world", "timestamp": 101.0},
    ])
    # depth 0: three nodes sharing one created_at (forces the tie-breaker)
    for i in range(3):
        engine._dag.add_node(SummaryNode(
            session_id=sid, depth=0, summary=f"d0-node-{i}", token_count=8,
            source_token_count=80, source_ids=[i], source_type="messages",
            created_at=1000.0, earliest_at=1000.0, latest_at=1000.0,
            expand_hint=f"d0 hint {i}",
        ))
    # depth 1: one node (not condensed into anything higher).  Its source_ids
    # must NOT collide with the depth-0 node ids just created (a real
    # reference would mark that node condensed and remove it from assembly).
    engine._dag.add_node(SummaryNode(
        session_id=sid, depth=1, summary="d1-node", token_count=8,
        source_token_count=80, source_ids=[9999], source_type="nodes",
        created_at=1200.0, earliest_at=1000.0, latest_at=1000.0,
        expand_hint="d1 hint",
    ))


def _dump(out) -> str:
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


def test_two_assemblies_byte_identical(tmp_path):
    engine = _make_engine(tmp_path)
    _seed(engine)
    out1 = engine._assemble_context(None, [], persist=False)
    out2 = engine._assemble_context(None, [], persist=False)
    assert _dump(out1) == _dump(out2)


def test_ties_are_stable_across_runs(tmp_path):
    """created_at ties: the rendered carrier order must not flip run to run."""
    engine = _make_engine(tmp_path)
    _seed(engine)
    first = _dump(engine._assemble_context(None, [], persist=False))
    for _ in range(4):
        again = _dump(engine._assemble_context(None, [], persist=False))
        assert again == first
    # Sanity: all three depth-0 carriers are present in the summary prefix.
    for i in range(3):
        assert f"d0-node-{i}" in first
    assert "d1-node" in first


def test_carrier_order_oldest_node_id_first_under_ties(tmp_path):
    """Under equal created_at, newest-selection keeps high node_id but the
    oldest-first display order puts smaller node_id first (pin of the SQL:
    ORDER BY created_at DESC, node_id DESC, then reversed)."""
    engine = _make_engine(tmp_path)
    _seed(engine)
    text = _dump(engine._assemble_context(None, [], persist=False))
    pos = {i: text.find(f"d0-node-{i}") for i in range(3)}
    assert all(p != -1 for p in pos.values())
    assert pos[0] < pos[1] < pos[2]
