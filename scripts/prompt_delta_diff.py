#!/usr/bin/env python3
"""prompt_delta_diff.py — byte-level delta analysis between two prompt snapshots.

Card ``t_3c5f4b44``: locate the exact rewritten content and delta position in
the production main session's adjacent assembled prompts (52K-90K band), and
classify the change mechanism.  Consumes the output of
``prompt_snapshot.py::build_snapshot`` (schema ``lcm-prompt-snapshot/v1``);
does NOT re-implement the snapshotter and does NOT touch assembly code.

Three-stage pipeline (frozen per card pre-registration):
  1. delta localization  -- first divergence offset in the canonical byte
     sequence of the assembled rows (same bytes ``history_sha256`` hashes),
     reported both in canonical-JSON bytes and in row index.
  2. changed-byte list   -- per changed row: added/deleted/modified bytes and
     a content digest (role, head, node_id when present).
  3. mechanism class     -- (a) fresh-tail roll, (b) compaction row deletion,
     (c) summary rewrite, (d) cross-session carrier row, (e) other.

Usage:
  python scripts/prompt_delta_diff.py TURN_A.json TURN_B.json [--json-out OUT]

Read-only: never writes unless --json-out is given. Exit 0 on success.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "lcm-prompt-snapshot/v1"
CUTOFF_BYTES = 32768  # the 32K boundary under investigation

_SUMMARY_CARRIER = re.compile(r"^\s*\[(?:Recent|Global|Session)? ?Summary")
_EXPAND_TAIL = "Expand for details about:"
_NODE_RE = re.compile(r"node\s+(\d+)", re.IGNORECASE)


def canonical_json(obj: Any) -> str:
    """Must match prompt_snapshot.canonical_json byte-for-byte."""
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def load_snapshot(path: Path) -> Dict[str, Any]:
    snap = json.loads(path.read_text(encoding="utf-8"))
    if snap.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {snap.get('schema')!r}")
    return snap


def rows_of(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild the exact rows list whose canonical hash is history_sha256."""
    return [m.get("row", {}) for m in snap.get("messages", [])]


# ---------------------------------------------------------------- stage 1

def stage1_localize(rows_a: List[dict], rows_b: List[dict]) -> Dict[str, Any]:
    """First divergence: row index (hash alignment) + canonical byte offset."""
    hashes_a = [canonical_json(r) for r in rows_a]
    hashes_b = [canonical_json(r) for r in rows_b]

    first_row = None
    for i, (ra, rb) in enumerate(zip(hashes_a, hashes_b)):
        if ra != rb:
            first_row = i
            break
    if first_row is None and len(hashes_a) != len(hashes_b):
        first_row = min(len(hashes_a), len(hashes_b))

    bytes_a = canonical_json(rows_a).encode("utf-8")
    bytes_b = canonical_json(rows_b).encode("utf-8")
    byte_delta = 0
    limit = min(len(bytes_a), len(bytes_b))
    while byte_delta < limit and bytes_a[byte_delta] == bytes_b[byte_delta]:
        byte_delta += 1

    return {
        "first_divergent_row_index": first_row,
        "first_divergent_byte_offset": byte_delta,
        "bytes_a": len(bytes_a),
        "bytes_b": len(bytes_b),
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "delta_after_32k_cutoff": byte_delta >= CUTOFF_BYTES,
    }


# ---------------------------------------------------------------- stage 2

def row_digest(row: dict) -> Dict[str, Any]:
    content = row.get("content")
    text = content if isinstance(content, str) else canonical_json(content)
    node = _NODE_RE.search(text or "")
    return {
        "role": str(row.get("role", "")),
        "content_chars": len(text) if isinstance(text, str) else None,
        "content_bytes": len(text.encode("utf-8")) if isinstance(text, str) else None,
        "head80": (text or "")[:80].replace("\n", " "),
        "node_id": int(node.group(1)) if node else None,
        "is_summary_carrier": bool(_SUMMARY_CARRIER.match(text or "")),
    }


def stage2_change_list(rows_a: List[dict], rows_b: List[dict]) -> Dict[str, Any]:
    """difflib alignment over canonical row strings -> per-row change list."""
    sa = [canonical_json(r) for r in rows_a]
    sb = [canonical_json(r) for r in rows_b]
    sm = difflib.SequenceMatcher(a=sa, b=sb, autojunk=False)

    entries: List[Dict[str, Any]] = []
    total_added = total_deleted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("delete", "replace"):
            for k in range(i1, i2):
                d = row_digest(rows_a[k])
                total_deleted += d["content_bytes"] or 0
                entries.append({"op": "delete", "index_a": k, **d})
        if tag in ("insert", "replace"):
            for k in range(j1, j2):
                d = row_digest(rows_b[k])
                total_added += d["content_bytes"] or 0
                entries.append({"op": "insert", "index_b": k, **d})
        if tag == "replace":
            # pair up same-position modifications for byte-level accounting
            for k in range(min(i2 - i1, j2 - j1)):
                ta, tb = sa[i1 + k], sb[j1 + k]
                changed = sum(
                    1 for x, y in zip(ta.encode(), tb.encode()) if x != y
                ) + abs(len(ta.encode()) - len(tb.encode()))
                entries.append({
                    "op": "modify",
                    "index_a": i1 + k,
                    "index_b": j1 + k,
                    "changed_bytes_estimate": changed,
                    "before": row_digest(rows_a[i1 + k]),
                    "after": row_digest(rows_b[j1 + k]),
                })

    return {
        "entries": entries,
        "total_deleted_content_bytes": total_deleted,
        "total_added_content_bytes": total_added,
        "opcodes": [
            {"tag": t, "a": [i1, i2], "b": [j1, j2]}
            for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t != "equal"
        ],
    }


# ---------------------------------------------------------------- stage 3

def _carrier_node_id(row: dict) -> Optional[int]:
    content = row.get("content")
    if not isinstance(content, str):
        return None
    m = _NODE_RE.search(content)
    return int(m.group(1)) if m else None


def stage3_classify(
    rows_a: List[dict],
    rows_b: List[dict],
    stage2: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Classify each opcode region into mechanism candidates (a)-(e)."""
    findings: List[Dict[str, Any]] = []
    n_a, n_b = len(rows_a), len(rows_b)

    for oc in stage2["opcodes"]:
        tag = oc["tag"]
        i1, i2 = oc["a"]
        j1, j2 = oc["b"]
        span_a = rows_a[i1:i2]
        span_b = rows_b[j1:j2]

        carriers = [
            r for r in span_a + span_b if row_digest(r)["is_summary_carrier"]
        ]
        node_ids_a = {_carrier_node_id(r) for r in span_a if _carrier_node_id(r)}
        node_ids_b = {_carrier_node_id(r) for r in span_b if _carrier_node_id(r)}

        # (c) summary rewrite: carrier row(s) modified/replaced, node id changed
        if carriers and (node_ids_a != node_ids_b or tag == "replace"):
            findings.append({
                "mechanism": "c_summary_rewrite",
                "region": oc,
                "evidence": {
                    "node_ids_a": sorted(x for x in node_ids_a if x is not None),
                    "node_ids_b": sorted(x for x in node_ids_b if x is not None),
                    "carrier_heads": [row_digest(r)["head80"] for r in carriers][:2],
                },
            })

        # (b) compaction row deletion: many rows out, one summary carrier in
        if tag == "replace" and len(span_a) >= 3 and len(span_b) == 1 \
                and row_digest(span_b[0])["is_summary_carrier"]:
            findings.append({
                "mechanism": "b_compaction_row_deletion",
                "region": oc,
                "evidence": {
                    "rows_removed": len(span_a),
                    "carrier_in": row_digest(span_b[0])["head80"],
                },
            })

        # (a) fresh-tail roll: head deletions + tail insertions, no carriers.
        # Head eviction may merge with a carrier rewrite into one replace
        # opcode, so the head check also accepts replaces anchored at <=2.
        if i1 <= 2 and (
            (tag == "delete" and not carriers)
            or (tag == "replace" and len(span_a) > len(span_b)
                and any(not row_digest(r)["is_summary_carrier"] for r in span_a))
        ):
            findings.append({
                "mechanism": "a_fresh_tail_roll_head_eviction",
                "region": oc,
                "evidence": {"rows_evicted_from_head": len(span_a), "head_index": i1},
            })
        # tail append: insertion lands at the very end of B (row-count drift
        # makes a positional j1 check unreliable; j2 == len(B) is exact)
        if tag == "insert" and j2 == n_b and not carriers:
            findings.append({
                "mechanism": "a_fresh_tail_roll_tail_append",
                "region": oc,
                "evidence": {"rows_appended": len(span_b)},
            })

        # (d) cross-session carrier: carrier content references another session
        for r in span_b:
            content = r.get("content")
            if isinstance(content, str) and re.search(
                r"session[_ ]id|跨会话|cross-session", content, re.IGNORECASE
            ):
                findings.append({
                    "mechanism": "d_cross_session_carrier",
                    "region": oc,
                    "evidence": {"head80": content[:80].replace("\n", " ")},
                })
                break

    # fallback: unexplained non-equal regions -> (e)
    classified_regions = {json.dumps(f["region"], sort_keys=True) for f in findings}
    for oc in stage2["opcodes"]:
        if json.dumps(oc, sort_keys=True) not in classified_regions:
            findings.append({
                "mechanism": "e_other",
                "region": oc,
                "evidence": {"note": "region not matched by (a)-(d) heuristics; "
                                      "manual read of the listed rows required"},
            })
    return findings


# ---------------------------------------------------------------- main

def analyze(path_a: Path, path_b: Path) -> Dict[str, Any]:
    a, b = load_snapshot(path_a), load_snapshot(path_b)
    rows_a, rows_b = rows_of(a), rows_of(b)
    s1 = stage1_localize(rows_a, rows_b)
    s2 = stage2_change_list(rows_a, rows_b)
    s3 = stage3_classify(rows_a, rows_b, s2)
    return {
        "schema": "prompt-delta-diff/v1",
        "inputs": {
            "a": {"path": str(path_a), "session": a.get("session_id"),
                  "turn": a.get("turn"), "ts": a.get("ts"),
                  "history_sha256": a.get("history_sha256")},
            "b": {"path": str(path_b), "session": b.get("session_id"),
                  "turn": b.get("turn"), "ts": b.get("ts"),
                  "history_sha256": b.get("history_sha256")},
        },
        "same_session": a.get("session_id") == b.get("session_id"),
        "stage1_delta_localization": s1,
        "stage2_change_list": s2,
        "stage3_mechanism_classification": s3,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("turn_a", type=Path)
    ap.add_argument("turn_b", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    result = analyze(args.turn_a, args.turn_b)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
