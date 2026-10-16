"""Differential and absolute checks return evidence, not tuned thresholds."""
from .ACCEPTED_DELTAS import ACCEPTED_DELTAS
from .trees import DEFAULT_BASELINE


def compare(case, ga, cand, baseline=DEFAULT_BASELINE):
    failures = []

    def check(oracle, condition, detail):
        if not condition:
            failures.append(dict(oracle=oracle, detail=detail, delta=None))

    if "error" in ga or "error" in cand:
        return [dict(oracle="DRIVER", detail={"ga": ga.get("error"), "candidate": cand.get("error")}, delta=None)]
    if case["layer"] == "projection":
        check("R1", not (ga["out_tokens"] <= case["limit"] < cand["out_tokens"]),
              dict(ga=ga["out_tokens"], candidate=cand["out_tokens"], limit=case["limit"]))
        check("P_USER", not cand["user_marks"], cand["user_marks"])
        check("P_INPUT", not cand["input_mutated"] and not ga["input_mutated"], "input mutated")
    else:
        check("CALLS", len(ga["calls"]) == len(cand["calls"]) == len(case["steps"]), "model call count differs")
        for i, (a, b) in enumerate(zip(ga["calls"], cand["calls"])):
            check("R1", not (a["tokens"] <= case["window"] < b["tokens"]),
                  dict(call=i, ga=a["tokens"], candidate=b["tokens"], window=case["window"]))
            for sentinel in ("⟦objective⟧", "⟦newest⟧"):
                check("R2", sentinel not in a["sentinels"] or sentinel in b["sentinels"], dict(call=i, sentinel=sentinel))
            check("A2", not b["pair_errors"], dict(call=i, errors=b["pair_errors"]))
        for metric in ("compress_runs", "summarizer_calls"):
            a = sum(call[metric] for call in ga["calls"])
            b = sum(call[metric] for call in cand["calls"])
            check("R3", b <= a * 1.10 + 2, dict(metric=metric, ga=a, candidate=b))
        check("C1", cand["uncovered_rows"] <= ga["uncovered_rows"] + 2,
              dict(ga=ga["uncovered_rows"], candidate=cand["uncovered_rows"]))
        lost = sorted(set(cand["produced"]) - set(cand["stored"]))
        check("A1", not lost, lost)
    for failure in failures:
        for delta in ACCEPTED_DELTAS:
            if (baseline in delta.baselines and case["layer"] in delta.layer
                    and failure["oracle"] == delta.oracle and delta.predicate(case, ga, cand, failure)):
                failure["delta"] = delta.id
                break
    return failures
