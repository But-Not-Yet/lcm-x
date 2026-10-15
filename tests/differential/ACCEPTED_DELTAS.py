"""Narrow, issue-backed exceptions; never suppress candidate absolute checks."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AcceptedDelta:
    id: str
    issue: str
    layer: tuple[str, ...]
    oracle: str
    reason: str
    predicate: Callable


def ga_marked_user(case, ga, cand, failure=None):
    if case["layer"] == "projection":
        return ga["out_tokens"] <= case["limit"] < cand["out_tokens"] and bool(ga["user_marks"])
    failures = [(a, b) for a, b in zip(ga["calls"], cand["calls"])
                if a["tokens"] <= case["window"] < b["tokens"]]
    return bool(failures) and all(a["user_marks"] for a, _ in failures)


ACCEPTED_DELTAS = [AcceptedDelta(
    "D916_GA_MARKED_USER", "https://github.com/electricsheephq/lcm-x/issues/916",
    ("projection", "public"), "R1", "GA fit with a mark-only user row; #916 deliberately preserves its text. Residual room: #1225.",
    ga_marked_user,
)]


def i1123_leaf_trade(case, ga, cand, failure):
    """#1123 / PR #1129: only the observed 64K managed provider-call cell."""
    return (case.get("id") == "i1139_small_window_64k"
            and failure["detail"] == dict(metric="summarizer_calls", ga=46, candidate=65))


ACCEPTED_DELTAS.append(AcceptedDelta(
    "D1123_64K_LEAF_TRADE", "https://github.com/electricsheephq/lcm-x/issues/1123",
    ("public",), "R3", "#1123 / PR #1129: 24K leaves trade 46 for 65 provider calls at 64K; compress runs fall.",
    i1123_leaf_trade,
))


def i1123_40k_compress_trade(case, ga, cand, failure):
    """Separate from the 64K delta: the triaged 40K sweep/hold cadence."""
    return (case.get("id") == "i1139_small_window_40k"
            and failure["detail"] == dict(metric="compress_runs", ga=93, candidate=113))


ACCEPTED_DELTAS.append(AcceptedDelta(
    "D1123_40K_COMPRESS_TRADE", "https://github.com/electricsheephq/lcm-x/issues/1123",
    ("public",), "R3", "#1123 / PR #1129: 40K sweep/hold cadence costs 113 vs 93 runs; provider calls fall 166 to 165.",
    i1123_40k_compress_trade,
))
