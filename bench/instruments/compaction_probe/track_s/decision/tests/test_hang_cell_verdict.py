"""Synthetic reader receipts: no model calls or external material."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("hang_verdict", Path(__file__).resolve().parents[1] / "hang_cell_verdict.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def receipt(mode, counts=(1, 1)):
    return dict(verdict=dict(result="FAIL", complete=True, bars={str(i): False for i in range(6)}),
                pooled=dict(synthetic=dict(runs=2, faulted_calls=sum(counts), fault_mode=mode, long_waits=2,
                    max_long_waits_in_one_turn=1, max_turn_wait_s=115.0, dropped_rows=0, survival_fits=0,
                    c_per_100=10.0, waited_turn_share=0.1, mean_turn_wait_s=12.0, facts_304_mean=0.5)),
                per_run={f"synthetic|d{n}-r1": dict(status="DONE", fault_mode=mode, faulted_calls=c)
                         for n, c in enumerate(counts, 1)})


def invoke(tmp_path, mode, base, candidate):
    for name, data in (("base", base), ("candidate", candidate)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    output = tmp_path / "verdict.json"
    rc = v.main(["--base", str(tmp_path / "base.json"), "--candidate", str(tmp_path / "candidate.json"),
                 "--variant", mode, "--json", str(output)])
    return rc, json.loads(output.read_text())


@pytest.mark.parametrize("mode", ["T", "P"])
def test_pass_and_d7_never_gates(tmp_path, mode):
    base = receipt(mode)
    candidate = receipt(mode, (2, 1)) if mode == "P" else copy.deepcopy(base)
    if mode == "P":
        candidate["pooled"]["synthetic"].update(long_waits=3, dropped_rows=42)
    rc, out = invoke(tmp_path, mode, base, candidate)
    assert rc == 0 and out["result"] == "PASS"
    assert out["base"]["d7"]["result"] == "FAIL"
    assert out["candidate"]["facts_304_mean"] == 0.5


@pytest.mark.parametrize("mode,change", [("T", dict(dropped_rows=1)), ("T", dict(long_waits=3)),
                                        ("P", dict(long_waits=4)), ("P", dict(max_long_waits_in_one_turn=2))])
def test_fail(tmp_path, mode, change):
    base, candidate = receipt(mode), receipt(mode)
    candidate["pooled"]["synthetic"].update(change)
    assert invoke(tmp_path, mode, base, candidate)[0] == 1


@pytest.mark.parametrize("mode", ["T", "P"])
@pytest.mark.parametrize("bad", ["invalid", "incomplete", "missing", "no_fault", "mode", "mixed", "run_zero", "sum", "run_missing"])
@pytest.mark.parametrize("side", ["base", "candidate"])
def test_invalid(tmp_path, mode, bad, side):
    base, candidate = receipt(mode), receipt(mode)
    data = base if side == "base" else candidate
    x = data["pooled"]["synthetic"]
    if bad == "invalid":
        data["verdict"]["result"] = "INVALID"
    elif bad == "incomplete":
        data["verdict"]["complete"] = False
    elif bad == "missing":
        del x["max_turn_wait_s"]
    elif bad == "no_fault":
        x["faulted_calls"] = 0
    elif bad in ("mode", "mixed"):
        x["fault_mode"] = ("P" if mode == "T" else "T") if bad == "mode" else "mixed"
    elif bad == "run_zero":
        data["per_run"]["synthetic|d1-r1"]["faulted_calls"] = 0
    elif bad == "sum":
        x["faulted_calls"] += 1
    else:
        del data["per_run"]["synthetic|d1-r1"]
    assert invoke(tmp_path, mode, base, candidate)[0] == 2


def test_t_rejects_uneven_counts_even_with_equal_pool(tmp_path):
    assert invoke(tmp_path, "T", receipt("T"), receipt("T", (0, 2)))[0] == 2


def test_invalid_json(tmp_path):
    (tmp_path / "bad.json").write_text("{")
    assert v.main(["--base", str(tmp_path / "bad.json"), "--candidate", str(tmp_path / "missing.json"),
                   "--variant", "T", "--json", str(tmp_path / "out.json")]) == 2
