"""Wiring test: the plugin's post_llm_call hook feeds prompt_snapshot.

Card t_8d410e97 task one. Proves the capture is actually on the production
channel (``__init__._on_post_llm_call``) and that it stays silent when the env
gate is off.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stub_agent(monkeypatch):
    agent_module = sys.modules.get("agent") or types.ModuleType("agent")
    agent_module.__path__ = []
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    ce = types.ModuleType("agent.context_engine")

    class ContextEngine:
        def on_session_reset(self):
            return None

    ce.ContextEngine = ContextEngine
    monkeypatch.setitem(sys.modules, "agent.context_engine", ce)
    monkeypatch.setattr(agent_module, "context_engine", ce, raising=False)


class _Ctx:
    def __init__(self):
        self.engine = None
        self.hooks = {}

    def register_context_engine(self, engine):
        self.engine = engine

    def register_hook(self, name, fn):
        self.hooks.setdefault(name, []).append(fn)


def _register(monkeypatch, tmp_path, module_name):
    _stub_agent(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT_PATH", str(tmp_path / "snap"))
    module = _load(module_name)
    ctx = _Ctx()
    module.register(ctx)
    assert ctx.engine is not None
    return module, ctx


def test_post_llm_call_feeds_prompt_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT", "1")
    module, ctx = _register(monkeypatch, tmp_path, "hermes_lcm_wiring_on")
    try:
        ps = importlib.import_module(module.__name__ + ".prompt_snapshot")
        ps._reset_for_tests()

        hook = ctx.hooks["post_llm_call"][-1]
        hook(
            session_id="wiring-session",
            conversation_id="c",
            platform="cli",
            conversation_history=[{"role": "user", "content": "wiring canary"}],
        )

        assert ps.drain_queue() == 1
        files = list((tmp_path / "snap" / "wiring-session").glob("turn_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["session_id"] == "wiring-session"
        assert data["platform"] == "cli"
        assert data["messages"][0]["row"]["content"] == "wiring canary"
    finally:
        ctx.engine.shutdown()


def test_post_llm_call_silent_when_gate_off(monkeypatch, tmp_path):
    monkeypatch.delenv("LCM_PROMPT_SNAPSHOT", raising=False)
    monkeypatch.delenv("LCM_PROMPT_SNAPSHOT_SESSIONS", raising=False)
    module, ctx = _register(monkeypatch, tmp_path, "hermes_lcm_wiring_off")
    try:
        ps = importlib.import_module(module.__name__ + ".prompt_snapshot")
        ps._reset_for_tests()

        ctx.hooks["post_llm_call"][-1](
            session_id="wiring-session",
            platform="cli",
            conversation_history=[{"role": "user", "content": "wiring canary"}],
        )
        assert ps.drain_queue() == 0
        assert list((tmp_path / "snap").glob("**/*.json")) == []
    finally:
        ctx.engine.shutdown()
