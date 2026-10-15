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


def ga_marked_user(case, ga, cand):
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
