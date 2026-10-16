#!/usr/bin/env python3
"""1096-H T/P comparison; D7 bars and retention are reported, never gated.

Exit 0 PASS, 1 FAIL, 2 INVALID. Inputs are complete pmw_read JSON receipts.
"""
import argparse
import json
import math
from pathlib import Path


def arm(path, variant):
    data = json.loads(path.read_text())
    verdict = data["verdict"]
    if verdict["result"] not in ("PASS", "FAIL") or verdict["complete"] is not True:
        raise ValueError(f"{path.name}: INVALID or incomplete reader")
    bars = verdict["bars"]
    if len(bars) != 6 or any(type(v) is not bool for v in bars.values()):
        raise ValueError("missing D7 bars")
    if len(data["pooled"]) != 1:
        raise ValueError("expected one pooled arm per input")
    name, metrics = next(iter(data["pooled"].items()))
    for key in ("runs", "faulted_calls", "long_waits", "max_long_waits_in_one_turn", "dropped_rows", "survival_fits"):
        value = metrics[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid {key}")
    for key in ("c_per_100", "waited_turn_share", "mean_turn_wait_s", "max_turn_wait_s", "facts_304_mean"):
        value = metrics[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"missing or invalid {key}")
    runs = metrics["runs"]
    if not runs or len(data["per_run"]) != runs or metrics["fault_mode"] != variant:
        raise ValueError("incomplete runs or wrong pooled fault_mode")
    counts = []
    for run in data["per_run"].values():
        if run["status"] != "DONE" or run["fault_mode"] != variant:
            raise ValueError("incomplete run or wrong fault_mode")
        count = run["faulted_calls"]
        if type(count) is not int or count < 1 or (variant == "T" and count != 1):
            raise ValueError("fault did not fire as declared per run")
        counts.append(count)
    if sum(counts) != metrics["faulted_calls"]:
        raise ValueError("pooled faulted_calls disagrees with per-run counts")
    return {"arm": name, "d7": verdict, **metrics}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--variant", choices=("T", "P"), required=True)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {"variant": args.variant}
    try:
        base, candidate = arm(args.base, args.variant), arm(args.candidate, args.variant)
        out.update(base=base, candidate=candidate)
        passed = (candidate["dropped_rows"] == 0 and candidate["long_waits"] <= base["long_waits"]
                  if args.variant == "T" else
                  candidate["long_waits"] <= base["long_waits"] + 1 and candidate["max_long_waits_in_one_turn"] <= 1)
        out["result"] = "PASS" if passed else "FAIL"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        out.update(result="INVALID", error=str(exc))
    args.json.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[out["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
