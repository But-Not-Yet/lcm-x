"""prompt_snapshot.py — read-only assembly-history dumper (env-gated, default OFF).

Card ``t_8d410e97`` task one: the production main session's assembled
``conversation_history`` (the exact list handed to ``post_llm_call``) is
suspected to be rewritten mid-prompt (>32K) every round.  To settle that with
a byte-level diff, this module captures the *full* assembled history each
round + a file-level sha256 + per-message row hashes, so adjacent rounds can
be diffed offline (delta position, changed-byte list, mechanism class).

Why a separate module (not the tiered ``assembly_probe``):
  * the tiered probe only records summary-carrier ``head120`` — no full rows,
    no hashes — so it cannot answer "what exactly changed after 32K";
  * this module lives in ``hermes-lcm-x`` and does NOT touch the tiered plugin.

Discipline (mirrors ``tiered/assembly_probe`` and ``tiered/factcheck``):
  * **default OFF** — nothing happens unless ``LCM_PROMPT_SNAPSHOT`` is truthy
    (all sessions) or ``LCM_PROMPT_SNAPSHOT_SESSIONS`` whitelists session ids;
  * **read-only** — it never mutates the caller's history and never touches
    assembly behaviour; the hot path is pure-memory;
  * **fail-open** — any exception is swallowed; capture can never break a turn;
  * **hot path never touches disk** — ``record`` only builds + enqueues; a
    lazily-started daemon thread batches writes (``drain_queue``).

Layout: ``<root>/<session_id>/turn_<NNNNN>.json`` (one full snapshot per
turn).  ``LCM_PROMPT_SNAPSHOT_PATH`` overrides the root (default
``~/.hermes/cache/scratch/prompt_snapshot``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "lcm-prompt-snapshot/v1"

_DEFAULT_ROOT = ("~/.hermes", "cache", "scratch", "prompt_snapshot")
_DEFAULT_FLUSH_SECONDS = 5.0

_lock = threading.Lock()
_queue: List[Dict[str, Any]] = []
_turn_counts: Dict[str, int] = {}
_flush_thread: Optional[threading.Thread] = None

_FALSEY = ("", "0", "false", "False", "no", "off")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, tolerant of odd values."""
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _enabled_globally() -> bool:
    return str(os.environ.get("LCM_PROMPT_SNAPSHOT", "")).strip() not in _FALSEY


def _whitelist() -> set:
    raw = os.environ.get("LCM_PROMPT_SNAPSHOT_SESSIONS", "")
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def is_enabled(session_id: str) -> bool:
    """True when capture is on for this session (default: off)."""
    if _enabled_globally():
        return True
    allowed = _whitelist()
    return bool(allowed) and str(session_id) in allowed


def _root() -> Path:
    raw = os.environ.get("LCM_PROMPT_SNAPSHOT_PATH")
    if raw and str(raw).strip():
        return Path(str(raw).strip())
    return Path.home().joinpath(*_DEFAULT_ROOT)


def _flush_seconds() -> float:
    raw = os.environ.get("LCM_PROMPT_SNAPSHOT_FLUSH_SECONDS")
    if raw is None or str(raw).strip() == "":
        return _DEFAULT_FLUSH_SECONDS
    try:
        return max(0.05, float(raw))
    except ValueError:
        return _DEFAULT_FLUSH_SECONDS


def _safe_session_id(session_id: str) -> str:
    s = str(session_id) or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:200] or "unknown"


def _message_row(msg: Any) -> Dict[str, Any]:
    """Return a JSON-safe, verbatim copy of one history row."""
    if isinstance(msg, dict):
        return msg
    return {"_non_dict": True, "repr": str(msg)}


def _content_len(row: Dict[str, Any]) -> Optional[int]:
    content = row.get("content")
    return len(content) if isinstance(content, str) else None


def build_snapshot(
    session_id: str,
    history: Any,
    turn: int,
    conversation_id: str = "",
    platform: str = "",
) -> Dict[str, Any]:
    """Pure-memory snapshot of one assembled history. Never raises on bad rows."""
    rows: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = []
    if isinstance(history, list):
        for index, msg in enumerate(history):
            row = _message_row(msg)
            rows.append(row)
            messages.append({
                "index": index,
                "role": str(row.get("role", "")) if isinstance(row, dict) else "",
                "content_len": _content_len(row),
                "content_sha256": canonical_sha256(row.get("content")),
                "row_sha256": canonical_sha256(row),
                "row": row,
            })
    return {
        "schema": SCHEMA,
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": str(session_id),
        "conversation_id": str(conversation_id or ""),
        "platform": str(platform or ""),
        "turn": int(turn),
        "history_len": len(messages),
        "history_sha256": canonical_sha256(rows),
        "messages": messages,
    }


def record(
    session_id: str,
    history: Any,
    *,
    conversation_id: str = "",
    platform: str = "",
    turn: Optional[int] = None,
) -> int:
    """Build + enqueue one snapshot; never touches disk; returns rows queued.

    Returns 0 when disabled or on any failure (fail-open by design).
    """
    try:
        if not is_enabled(session_id):
            return 0
        if not isinstance(history, list) or not history:
            return 0
        with _lock:
            if turn is None:
                turn = _turn_counts.get(str(session_id), 0) + 1
                _turn_counts[str(session_id)] = turn
        snapshot = build_snapshot(
            session_id, history, int(turn),
            conversation_id=conversation_id, platform=platform,
        )
        with _lock:
            _queue.append(snapshot)
        _ensure_flush_thread()
        return 1
    except Exception:  # noqa: BLE001 - fail-open by design
        return 0


def _snapshot_path(session_id: str, turn: int) -> Path:
    return _root() / _safe_session_id(session_id) / f"turn_{int(turn):05d}.json"


def _flush_loop() -> None:
    delay = _flush_seconds()
    time.sleep(delay)
    while True:
        drain_queue()
        time.sleep(delay)


def _ensure_flush_thread() -> None:
    global _flush_thread
    if _flush_thread is not None and _flush_thread.is_alive():
        return
    _flush_thread = threading.Thread(
        target=_flush_loop, name="lcm-prompt-snapshot", daemon=True
    )
    _flush_thread.start()


def _write_snapshot(snapshot: Dict[str, Any]) -> None:
    """Disk write (writer thread only). Never overwrites an earlier round."""
    path = _snapshot_path(snapshot["session_id"], snapshot["turn"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # process restart reset the turn counter; keep both
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = path.with_name(f"{path.stem}_{stamp}.json")
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, default=str), encoding="utf-8"
    )


def drain_queue() -> int:
    """Write all queued snapshots to disk (append-style, one file each)."""
    with _lock:
        if not _queue:
            return 0
        rows, _queue[:] = list(_queue), []
    written = 0
    failed: List[Dict[str, Any]] = []
    for snapshot in rows:
        try:
            _write_snapshot(snapshot)
            written += 1
        except Exception:  # noqa: BLE001 - fail-open
            failed.append(snapshot)
    if failed:
        with _lock:
            _queue[:0] = failed  # put back; next flush retries
    return written


def _reset_for_tests() -> None:
    """Clear in-process state. Test-only; does not touch env or disk."""
    with _lock:
        _queue.clear()
        _turn_counts.clear()
