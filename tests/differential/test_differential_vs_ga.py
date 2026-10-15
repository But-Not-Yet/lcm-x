"""No differential imports or subprocesses until explicitly enabled."""
import os

import pytest

if os.environ.get("LCM_DIFFERENTIAL") != "1":
    pytest.skip("opt in with LCM_DIFFERENTIAL=1", allow_module_level=True)

from pathlib import Path

from .cases import exact_cases, generated_cases, projection_cases
from .run import run
from .trees import PrerequisiteUnavailable, skip_or_fail


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = os.environ.get("LCM_DIFFERENTIAL_OUT")
    return Path(root) if root else tmp_path_factory.mktemp("differential")


def evaluate(cases, output):
    try:
        result = run(cases, output, os.environ.get("LCM_DIFFERENTIAL_BASELINE", "v0.26.2"),
                     os.environ.get("LCM_DIFFERENTIAL_CANDIDATE", "worktree"), int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227")))
    except PrerequisiteUnavailable as exc:
        skip_or_fail(str(exc))
    assert not result["unexplained"], f"unexplained differential reds: {output / 'summary.json'}"
    return result


@pytest.mark.parametrize("case", [case for case in exact_cases() if not case.get("absolute_xfail")], ids=lambda case: case["id"])
def test_exact_shape(case, outputs):
    evaluate([case], outputs / case["id"])


def test_generated_sample(outputs):
    evaluate(generated_cases(int(os.environ.get("LCM_DIFFERENTIAL_CASES", "60")),
                             int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227"))), outputs / "generated")


def test_projection(outputs):
    evaluate(projection_cases(int(os.environ.get("LCM_DIFFERENTIAL_PROJECTION_CASES", "1000")),
                              int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227"))), outputs / "projection")


def test_i1225_objective_exceeds_room_32k(outputs):
    case = next(case for case in exact_cases() if case.get("absolute_xfail"))
    evaluate([case], outputs / case["id"])


def test_i1225_shape_fits(outputs):
    """This 22k-objective shape projects to 22,579 tokens and fits. It does not reproduce #1225's absolute
    failure, so it is pinned as a fit, not as a strict xfail; a reproducing shape needs the issue's host input."""
    import json
    path = outputs / "i1225_objective_exceeds_room_32k" / "summary.json"
    if not path.exists():
        case = next(case for case in exact_cases() if case.get("absolute_xfail"))
        evaluate([case], path.parent)
    absolute_fit(json.loads(path.read_text()))


def absolute_fit(result):
    assert not result["absolute_over_window"], result["absolute_over_window"]


from copy import deepcopy

from .driver import pair_errors
from .oracles import compare


def test_projection_delta_is_only_fit_with_ga_user_marks():
    case = dict(layer="projection", limit=100)
    ga = dict(out_tokens=90, user_marks=[0], input_mutated=False)
    cand = dict(out_tokens=110, user_marks=[], input_mutated=False)
    assert compare(case, ga, cand)[0]["delta"] == "D916_GA_MARKED_USER"
    ga["user_marks"] = []
    assert compare(case, ga, cand)[0]["delta"] is None
    cand.update(user_marks=[0], input_mutated=True)
    assert {row["oracle"] for row in compare(case, ga, cand)} == {"R1", "P_USER", "P_INPUT"}


def test_public_oracles_and_delta_do_not_hide_absolute_failures():
    case = dict(layer="public", window=100, steps=[[]])
    call = dict(tokens=90, sentinels=["⟦objective⟧", "⟦newest⟧"], user_marks=[0], pair_errors=[],
                compress_runs=0, summarizer_calls=0)
    ga = dict(calls=[call], produced=["⟦r0.0⟧"], stored=["⟦r0.0⟧"])
    cand = deepcopy(ga)
    cand["calls"][0].update(tokens=110, sentinels=[], user_marks=[], pair_errors=["missing-result:c"],
                            compress_runs=3, summarizer_calls=3)
    cand["stored"] = []
    failures = compare(case, ga, cand)
    assert {row["oracle"] for row in failures} == {"R1", "R2", "R3", "A1", "A2"}
    assert all(row["delta"] is None for row in failures if row["oracle"] != "R1")


def test_tool_results_must_follow_calls_and_every_call_needs_a_result():
    assistant = {"role": "assistant", "tool_calls": [{"id": "c"}]}
    tool = {"role": "tool", "tool_call_id": "c", "content": "stub"}
    assert pair_errors([assistant, tool]) == []
    assert pair_errors([tool, assistant]) == ["result-before-call:c"]
    assert pair_errors([assistant]) == ["missing-result:c"]
