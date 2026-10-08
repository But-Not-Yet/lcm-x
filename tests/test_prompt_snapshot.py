"""Tests for prompt_snapshot — read-only, env-gated assembly-history dumper.

Card t_8d410e97, task one: capture the assembled conversation_history handed
to ``post_llm_call`` each turn (full rows + file sha256 + per-message row
hashes) so two adjacent production rounds can be diffed offline (delta
position + changed-byte list + mechanism classification).

Discipline pinned here:
  * env-gated, DEFAULT OFF (zero overhead until enabled);
  * session whitelist variant;
  * fail-open (never raises, never mutates the caller's history);
  * hot path is pure-memory; disk writes only via drain_queue();
  * filenames never clobber an earlier round after a process restart.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from hermes_lcm import prompt_snapshot as ps


def _history():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "[Recent Summary (d0, node 312)] tail"},
    ]


def _enable(monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT", "1")
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT_PATH", str(tmp_path))
    monkeypatch.delenv("LCM_PROMPT_SNAPSHOT_SESSIONS", raising=False)
    ps._reset_for_tests()


def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LCM_PROMPT_SNAPSHOT", raising=False)
    monkeypatch.delenv("LCM_PROMPT_SNAPSHOT_SESSIONS", raising=False)
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT_PATH", str(tmp_path))
    ps._reset_for_tests()

    assert ps.record("s1", _history()) == 0
    assert ps.drain_queue() == 0
    assert list(tmp_path.glob("**/*.json")) == []


def test_enabled_dumps_full_history_with_hashes(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    hist = _history()

    assert ps.record("s1", hist, conversation_id="c1", platform="cli") == 1
    assert ps.drain_queue() == 1

    files = list(tmp_path.glob("s1/turn_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))

    assert data["schema"] == "lcm-prompt-snapshot/v1"
    assert data["session_id"] == "s1"
    assert data["conversation_id"] == "c1"
    assert data["platform"] == "cli"
    assert data["turn"] == 1
    assert data["history_len"] == 3

    # Full rows preserved verbatim (read-only capture).
    assert [m["row"]["content"] for m in data["messages"]] == [
        "sys", "hello", "[Recent Summary (d0, node 312)] tail",
    ]
    assert [m["role"] for m in data["messages"]] == ["system", "user", "assistant"]

    # Per-message row hash.
    for m in data["messages"]:
        expected = hashlib.sha256(
            ps.canonical_json(m["row"]).encode("utf-8")
        ).hexdigest()
        assert m["row_sha256"] == expected

    # History-level hash over the ordered full rows.
    rows = [m["row"] for m in data["messages"]]
    expected_history = hashlib.sha256(
        ps.canonical_json(rows).encode("utf-8")
    ).hexdigest()
    assert data["history_sha256"] == expected_history


def test_turn_counter_increments_per_session(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    ps.record("s1", _history())
    ps.record("s1", _history())
    ps.record("s2", _history())
    assert ps.drain_queue() == 3

    s1 = sorted(p.name for p in tmp_path.glob("s1/turn_*.json"))
    s2 = sorted(p.name for p in tmp_path.glob("s2/turn_*.json"))
    assert s1 == ["turn_00001.json", "turn_00002.json"]
    assert s2 == ["turn_00001.json"]


def test_session_whitelist(monkeypatch, tmp_path):
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT", "")
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT_SESSIONS", "A")
    monkeypatch.setenv("LCM_PROMPT_SNAPSHOT_PATH", str(tmp_path))
    ps._reset_for_tests()

    assert ps.record("A", _history()) == 1
    assert ps.record("B", _history()) == 0
    assert ps.drain_queue() == 1
    assert list(tmp_path.glob("B/turn_*.json")) == []


def test_fail_open_on_bad_input(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)

    # Non-list / empty histories never raise and enqueue nothing.
    assert ps.record("s1", None) == 0
    assert ps.record("s1", "not-a-list") == 0

    # A non-JSON-serializable content value must still yield a valid dump.
    hist = [{"role": "user", "content": object()}]
    assert ps.record("s1", hist) == 1
    assert ps.drain_queue() == 1
    f = list(tmp_path.glob("s1/turn_*.json"))[0]
    data = json.loads(f.read_text(encoding="utf-8"))  # valid JSON, no crash
    assert data["history_len"] == 1


def test_read_only_does_not_mutate_caller_history(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    hist = _history()
    snapshot = copy.deepcopy(hist)

    ps.record("s1", hist)
    ps.drain_queue()
    assert hist == snapshot


def test_history_hash_changes_with_content(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    a = ps.build_snapshot("s1", _history(), turn=1)
    changed = _history()
    changed[1]["content"] = "changed"
    b = ps.build_snapshot("s1", changed, turn=1)
    assert a["history_sha256"] != b["history_sha256"]
    assert a["messages"][1]["row_sha256"] != b["messages"][1]["row_sha256"]


def test_restart_does_not_clobber_earlier_round(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    ps.record("s1", _history())
    ps.drain_queue()

    # Simulate a fresh process: turn counter resets to 1.
    ps._reset_for_tests()
    ps.record("s1", _history())
    ps.drain_queue()

    files = sorted(p.name for p in tmp_path.glob("s1/turn_*.json"))
    assert len(files) == 2
    assert files[0] == "turn_00001.json"
