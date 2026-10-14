"""#1162: a retained occurrence that never resolves stopped every pass with "leaf would end before an unresolved
retained occurrence" (leaves=0), and nothing named the row or warned. The stop now names the blocking row (store_id,
role, kind) on the stop line, the no-op line, the sweep record and the status; a sanitizing pass keeps that reason;
doctor warns after consecutive over-threshold passes without a leaf.

Engine-level, on a scratch store with a stub summariser (the repro of the #1162 root-cause comment)."""

from __future__ import annotations

import json
import re
import logging

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
RETAINED = "leaf would end before an unresolved retained occurrence"


@pytest.fixture
def summaries(monkeypatch):
    captured: list[str] = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[tuple[int, str]] = []

    def emit(self, record):
        self.lines.append((record.levelno, record.getMessage()))


@pytest.fixture
def lines():
    handler, logger = _Lines(), logging.getLogger("hermes_lcm")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.lines
    logger.removeHandler(handler)
    logger.setLevel(level)


def _engine(tmp_path, sweep: bool, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "dynamic_leaf_chunk_max": 400, "threshold_full_sweep_enabled": sweep,
                "temporal_rollups_enabled": False, "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, tool: bool = False) -> list[dict]:
    rows = [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts}]
    if tool:
        call = {"id": f"call_{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        rows += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": f"call_{tag}", "content": f"result of {tag}{PAD}"}]
    return rows + [{"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _rows(engine) -> list[dict]:
    return engine._store.get_session_messages("S", limit=100_000)


def _frontier(engine) -> int:
    return int(engine._lifecycle.get_by_conversation("conv").current_frontier_store_id or 0)


def _passes(engine, view, n, lines, label="") -> tuple[list, list[dict]]:
    """``n`` threshold passes, each a foreground turn; the 600 s no-progress hold only paces passes (a real host
    retries after it), so it is cleared between them."""
    trace = []
    for _ in range(n):
        start = len(lines)
        view = engine.compress(view, current_tokens=10_000)
        engine.note_turn_complete()
        engine._no_progress_hold = None
        new = lines[start:]
        stop = next((msg for _lvl, msg in new if msg.startswith("LCM compaction stop:")), "")
        trace.append({
            "label": label, "status": engine._last_compression_status,
            "reason": engine._last_compression_noop_reason, "frontier": _frontier(engine),
            "leaves": int(stop.split("leaves=", 1)[1].split(" ", 1)[0]) if "leaves=" in stop else None,
            "stop": stop.split(" elapsed=", 1)[0], "line": stop, "rows": len(view),
            "warnings": [msg for lvl, msg in new if lvl >= logging.WARNING],
        })
    return view, trace


def _show(trace) -> str:
    return "\n".join(f"{t['label']} status={t['status']} frontier={t['frontier']} rows={t['rows']} | {t['stop']} "
                     f"| noop={t['reason']!r}" for t in trace)


def _reorder(tmp_path, sweep):
    """#581 test (c) shape: user row R is stored early; the host shows it at the END of its list."""
    engine = _engine(tmp_path, sweep, fresh_tail_count=1, leaf_chunk_tokens=1)
    r = {"role": "user", "content": "[R] retained occurrence" + PAD}
    stored = [*_turn("T1", 100.0), r, *[row for i in range(2, 6) for row in _turn(f"T{i}", 100.0 * i)]]
    engine.ingest(stored)
    view = [m for m in stored if m is not r] + [dict(r)]
    engine._ingest_cursor = len(view)
    return engine, view


def _restore(tmp_path, sweep):
    """The host rewrites an older tool output in place (Hermes' salvage text, #773): the ingest stores it and the
    unstamped rows after it again at the END, and the oldest host row binds to a newer store id."""
    engine = _engine(tmp_path, sweep, fresh_tail_count=4, leaf_chunk_tokens=400)
    view = [row for i in range(1, 13) for row in _turn(f"T{i}", 100.0 * i, tool=True)]
    view = engine.compress(view, current_tokens=10_000)
    engine.note_turn_complete()
    engine._no_progress_hold = None
    tools = [m for m in view if m.get("role") == "tool"]
    tools[0]["content"] = "[Old tool output cleared to save context space]"
    return engine, view + _turn("T13", 1300.0, tool=True)


def _scenario(name, tmp_path, sweep):
    return (_reorder if name == "reorder" else _restore)(tmp_path, sweep)


def _r_id(engine) -> int:
    return next(int(x["store_id"]) for x in _rows(engine) if x["content"].startswith("[R] retained"))


SCENARIOS = pytest.mark.parametrize("name", ["reorder", "restore"])
SWEEP = pytest.mark.parametrize("sweep", [True, False], ids=["sweep_on", "sweep_off"])


# -- observability ---------------------------------------------------------------------------------------------

@SCENARIOS
@SWEEP
def test_the_first_retained_stop_names_the_blocking_row(tmp_path, summaries, lines, name, sweep):
    """The stop line, the no-op line, the sweep record and the status all carry the blocking row (store_id, role,
    kind); with the sweep off a sanitizing pass no longer erases the reason."""
    engine, view = _scenario(name, tmp_path, sweep)
    try:
        trace, start = [], 0
        while len(trace) < 3 and not (trace and trace[-1]["reason"].startswith(RETAINED)):
            start = len(lines)
            view, step = _passes(engine, view, 1, lines, name)  # leaves may still form before the stop
            trace += step
        named = trace[-1]["reason"]
        assert named.startswith(RETAINED + " (store_id=") and named.endswith(" kind=shown)"), _show(trace)
        if name == "reorder":
            assert named == f"{RETAINED} (store_id={_r_id(engine)} role=user kind=shown)"
        store_id, role = named.split("(store_id=", 1)[1].split(" kind=", 1)[0].split(" role=")
        line = trace[-1]["line"]  # reason= exactly as on base (sweep on: the #581 text); the row is its own field
        assert trace[-1]["leaves"] == 0 and line.endswith(f" blocked=store_id:{store_id},role:{role},kind:shown")
        assert re.search(f"reason={RETAINED} leaves=0 progress=" if sweep else "reason=(noop|sanitized) leaves=0 "
                         "progress=", line) and "(store_id=" not in line, line
        assert f"LCM compression no-op: {named}" in [msg for _lvl, msg in lines[start:]]
        status = engine.get_status()
        assert status["last_compression_noop_reason"] == named
        assert status["zero_leaf_streak"]["last_reason"] == named
        if sweep:
            assert status["threshold_full_sweep"]["stop_reason"] == named
    finally:
        engine.shutdown()


def test_a_sanitizing_retained_stop_is_not_pressure_relief(tmp_path, summaries, lines, monkeypatch):
    """With the sweep off the restore stop returns a sanitized list: no leaf was written, so the fresh-tail
    pressure evidence is not reset and the reason is kept."""
    engine, view = _restore(tmp_path, False)
    relieved = []
    monkeypatch.setattr(engine, "_note_fresh_tail_pressure_relieved", lambda: relieved.append(1))
    try:
        trace = []
        while not (trace and trace[-1]["reason"].startswith(RETAINED)) and len(trace) < 4:
            relieved.clear()
            view, step = _passes(engine, view, 1, lines, "restore")  # leaves still form before the stop
            trace += step
        assert trace[-1]["status"] == "sanitized" and trace[-1]["reason"].startswith(RETAINED), _show(trace)
        assert relieved == []
    finally:
        engine.shutdown()


def test_doctor_warns_on_a_zero_leaf_streak_and_clears_after_a_leaf(tmp_path, summaries, lines):
    """Three consecutive over-threshold passes without a leaf warn with the last reason; a stored leaf clears it."""
    engine = _engine(tmp_path, False, fresh_tail_count=8, fresh_tail_pressure_yield_enabled=False)
    view = [row for i in range(1, 3) for row in _turn(f"T{i}", 100.0 * i)]  # all of it is fresh tail

    def doctor():
        report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
        return next(check for check in report["checks"] if check["check"] == "compaction_progress")

    try:
        assert doctor() == {"check": "compaction_progress", "status": "pass", "detail": {"passes": 0}}
        streaks = []
        for _ in range(3):
            view, _trace = _passes(engine, view, 1, lines)
            streaks.append((engine.get_status()["zero_leaf_streak"] or {}).get("passes"))
            if len(streaks) < 3:
                assert doctor()["status"] == "pass"
        assert streaks == [1, 2, 3]
        check = doctor()
        assert check["status"] == "warn" and check["detail"]["passes"] == 3
        assert check["detail"]["last_reason"] == "no eligible raw backlog outside fresh tail"
        report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
        assert any(item["check"] == "compaction_progress" and item["warning_only"] for item in report["guidance"])
        view, trace = _passes(engine, view + [row for i in range(3, 9) for row in _turn(f"T{i}", 100.0 * i)], 1,
                              lines)
        assert trace[0]["leaves"] >= 1, _show(trace)
        assert engine.get_status()["zero_leaf_streak"] is None and doctor()["status"] == "pass"
    finally:
        engine.shutdown()


# -- review of 559ce54f: the streak counts only real over-threshold passes; the stop line keeps one reason token --

def _streak(engine):
    return (engine.get_status()["zero_leaf_streak"] or {}).get("passes")


def _streak_of_three(tmp_path, lines):
    engine = _engine(tmp_path, False, fresh_tail_count=8, fresh_tail_pressure_yield_enabled=False)
    view = [row for i in range(1, 3) for row in _turn(f"T{i}", 100.0 * i)]  # all of it is fresh tail
    view, _trace = _passes(engine, view, 3, lines)
    assert _streak(engine) == 3
    return engine, view


def test_a_pass_at_or_under_the_threshold_clears_the_streak(tmp_path, summaries, lines):
    """Review 2: the streak is consecutive OVER-threshold passes: a pass the host observes under it starts it over,
    and so does the pressure-yield entry point that sees the session under the threshold."""
    engine, view = _streak_of_three(tmp_path, lines)
    try:
        engine.compress(view, current_tokens=1)
        assert _streak(engine) is None
        view, _trace = _passes(engine, view, 1, lines)
        assert _streak(engine) == 1
        engine._config.fresh_tail_pressure_yield_enabled = True
        engine._maybe_engage_fresh_tail_pressure_yield(view, 1, eligible_tokens=0)
        assert _streak(engine) is None
    finally:
        engine.shutdown()


def test_only_passes_that_select_a_leaf_count(tmp_path, summaries, lines):
    """Review 3: calls the #651 hold turns into cleanup-only maintenance run no leaf selection and never count; a
    new session that keeps the conversation id starts without the old streak."""
    engine = _engine(tmp_path, False, fresh_tail_count=8, fresh_tail_pressure_yield_enabled=False)
    view = [row for i in range(1, 3) for row in _turn(f"T{i}", 100.0 * i)]
    try:
        view = engine.compress(view, current_tokens=10_000)
        assert _streak(engine) == 1 and engine._no_progress_hold is not None
        for _ in range(2):  # held: the hold is not cleared between these calls
            view = engine.compress(view, current_tokens=10_000)
            assert engine._last_compression_status == "sanitized"
        assert _streak(engine) == 1
        engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv")
        assert _streak(engine) is None
    finally:
        engine.shutdown()


def test_the_stop_line_parsers_read_the_blocked_line(tmp_path, summaries, lines):
    """Review 4: the in-repo stop-line parsers (#605, the host fallback, #671) read a stop line with ``blocked=``."""
    from tests.test_host_fallback_stop_line import STOP as host_fallback
    from tests.test_issue_605_foreground_budget import STOP_LINE as foreground_budget
    from tests.test_issue_671_stub_first_exit import STOP as stub_first_exit

    engine, view = _reorder(tmp_path, False)
    try:
        view, trace = _passes(engine, view, 3, lines)
        line = next(t["line"] for t in trace if " blocked=" in t["line"])
        for parser in (foreground_budget, host_fallback, stub_first_exit):
            found = parser.search(line)
            assert found and found.group(1) == "noop" and found.group(2) == "0", (parser.pattern, line)
    finally:
        engine.shutdown()
