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


@pytest.mark.parametrize("case", [case for case in exact_cases()
                                 if not case.get("absolute_xfail")
                                 and case["id"] != "i1225_objective_exceeds_room_32k"], ids=lambda case: case["id"])
def test_exact_shape(case, outputs):
    evaluate([case], outputs / case["id"])


def test_generated_sample(outputs):
    seed = int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227"))
    cases = generated_cases(int(os.environ.get("LCM_DIFFERENTIAL_CASES", "60")), seed)
    evaluate([case for case in cases if seed != 1227 or case["id"] != "g0"], outputs / "generated")


class I1281Retention(Exception):
    """The documented g0 user mark loses retention without reaching the budget."""


@pytest.mark.xfail(strict=True, raises=I1281Retention, reason="#1281: g0 marks newest user without reaching survival budget")
def test_i1281_g0_marks_newest_user_without_budget(outputs):
    case = generated_cases(1, 1227)[0]
    assert case["id"] == "g0"
    try:
        result = run([case], outputs / "i1281_g0", os.environ.get("LCM_DIFFERENTIAL_BASELINE", "v0.26.2"),
                     os.environ.get("LCM_DIFFERENTIAL_CANDIDATE", "worktree"), int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227")))
    except PrerequisiteUnavailable as exc:
        skip_or_fail(str(exc))
    events = [(event["oracle"], event["detail"]) for event in result["unexplained"]]
    if events == [("R2", dict(call=0, sentinel="⟦objective⟧")), ("R2", dict(call=0, sentinel="⟦newest⟧"))]:
        raise I1281Retention("#1281: g0 loses objective and newest at call 0 without reaching the survival budget")
    assert not result["unexplained"], result["unexplained"]


def test_projection(outputs):
    evaluate(projection_cases(int(os.environ.get("LCM_DIFFERENTIAL_PROJECTION_CASES", "1000")),
                              int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227"))), outputs / "projection")


def test_i1225_objective_exceeds_room_32k(outputs):
    case = next(case for case in exact_cases() if case["id"] == "i1225_objective_exceeds_room_32k")
    evaluate([case], outputs / case["id"])


def test_i1225_shape_fits(outputs):
    """This 22k-objective shape projects to 22,579 tokens and fits. It does not reproduce #1225's absolute
    failure, so it is pinned as a fit, not as a strict xfail; a reproducing shape needs the issue's host input."""
    import json
    path = outputs / "i1225_objective_exceeds_room_32k" / "summary.json"
    if not path.exists():
        case = next(case for case in exact_cases() if case["id"] == "i1225_objective_exceeds_room_32k")
        evaluate([case], path.parent)
    absolute_fit(json.loads(path.read_text()))


def absolute_fit(result):
    assert not result["absolute_over_window"], result["absolute_over_window"]


class ObjectiveOverflow(AssertionError):
    """The preserved objective leaves insufficient room for the newest turn."""


@pytest.mark.xfail(strict=True, raises=ObjectiveOverflow, reason="#1225: objective exceeds available window room")
def test_i1225_historical_objective_overflow(outputs):
    import json
    case = next(case for case in exact_cases() if case.get("absolute_xfail"))
    result = evaluate([case], outputs / case["id"])
    if result["absolute_over_window"]:
        ga = [json.loads(line) for line in (outputs / case["id"] / "public-ga.jsonl").read_text().splitlines()]
        assert ga[0]["calls"][0]["tokens"] > case["window"], "#1225 must also overflow on GA"
        raise ObjectiveOverflow(str(result["absolute_over_window"]))


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
    ga = dict(uncovered_rows=0, calls=[call], produced=["⟦r0.0⟧"], stored=["⟦r0.0⟧"])
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


class SummaryCoverageGap(AssertionError):
    """Stored history is outside both the live context and summary coverage."""


@pytest.mark.xfail(strict=True, raises=SummaryCoverageGap, reason="#1247: 32K history lacks summary coverage")
def test_i1247_32k_history_is_live_or_summarized(outputs):
    import json
    cid = "i1139_small_window_32k"
    path = outputs / cid / "summary.json"
    if not path.exists():
        evaluate([next(case for case in exact_cases() if case["id"] == cid)], path.parent)
    count = json.loads(path.read_text())["coverage"][cid]["candidate"]
    if count:
        raise SummaryCoverageGap(f"{count} stored rows neither live nor summarized")


def test_objective_is_checked_only_on_newest_user_row():
    from .driver import visible_sentinels
    first = dict(role="user", content="⟦r0⟧⟦objective⟧")
    later = dict(role="user", content="⟦r1⟧⟦newest⟧")
    assert "⟦objective⟧" in visible_sentinels([first], [first])
    assert "⟦objective⟧" not in visible_sentinels([later], [first])
    assert "⟦objective⟧" not in visible_sentinels([first, later], [first, later])
    later["content"] += "⟦objective⟧"
    assert "⟦objective⟧" in visible_sentinels([later], [first, later])


def test_managed_baseline_and_case_override_precedence():
    from .driver import MANAGED_BASELINE, config_overrides
    case = dict(overrides={})
    assert config_overrides(case, "bare") == {}
    assert config_overrides(case, "managed") == MANAGED_BASELINE
    assert MANAGED_BASELINE == dict(context_threshold=0.75, fresh_tail_count=24, fresh_tail_max_tokens=24000,
                                   leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
                                   large_output_externalization_enabled=True, large_output_active_replay_stubbing_enabled=True)
    case["overrides"] = dict(leaf_chunk_tokens=123)
    assert config_overrides(case, "managed")["leaf_chunk_tokens"] == 123
    assert config_overrides(case, "bare") == case["overrides"]


def test_summary_coverage_counts_live_and_all_dag_message_sources():
    from types import SimpleNamespace
    from .driver import uncovered_rows
    rows = [dict(role="assistant", content=f"⟦r{i}⟧", store_id=i) for i in range(4)]
    nodes = [SimpleNamespace(source_type="messages", source_ids=[1, 2]),
             SimpleNamespace(source_type="summaries", source_ids=[3])]
    assert uncovered_rows(rows, [rows[0]], nodes) == 1
    assert uncovered_rows(rows, [rows[0], rows[3]], nodes) == 0
    assert uncovered_rows(rows, [], []) == 4


def test_c1_tolerance_and_direction():
    case = dict(id="coverage", layer="public", window=100, steps=[])
    ga = dict(calls=[], produced=[], stored=[], uncovered_rows=0)
    cand = dict(ga, uncovered_rows=2)
    assert compare(case, ga, cand) == []
    cand["uncovered_rows"] = 3
    assert compare(case, ga, cand) == [dict(oracle="C1", detail=dict(ga=0, candidate=3), delta=None)]
    assert compare(case, cand, ga) == []


def test_i1123_delta_does_not_absorb_other_cells():
    from .ACCEPTED_DELTAS import i1123_leaf_trade
    case = dict(id="i1139_small_window_64k")
    failure = dict(detail=dict(metric="summarizer_calls", ga=46, candidate=65))
    assert i1123_leaf_trade(case, {}, {}, failure)
    assert not i1123_leaf_trade(dict(id="i1139_small_window_48k"), {}, {}, failure)
    for detail in (dict(metric="compress_runs", ga=46, candidate=65),
                   dict(metric="summarizer_calls", ga=45, candidate=65)):
        assert not i1123_leaf_trade(case, {}, {}, dict(detail=detail))


def test_i1123_40k_delta_is_a_separate_exact_cell():
    from .ACCEPTED_DELTAS import i1123_40k_compress_trade
    case = dict(id="i1139_small_window_40k")
    failure = dict(detail=dict(metric="compress_runs", ga=93, candidate=113))
    assert i1123_40k_compress_trade(case, {}, {}, failure)
    assert not i1123_40k_compress_trade(dict(id="i1139_small_window_64k"), {}, {}, failure)
    for detail in (dict(metric="summarizer_calls", ga=93, candidate=113),
                   dict(metric="compress_runs", ga=92, candidate=113)):
        assert not i1123_40k_compress_trade(case, {}, {}, dict(detail=detail))


def test_i1233_delta_is_only_projection_user_marks_already_in_ga():
    from .ACCEPTED_DELTAS import last_resort_user_mark
    case = dict(layer="projection", limit=100)
    ga = dict(out_tokens=90, user_marks=[0, 2], input_mutated=False)
    cand = dict(out_tokens=90, user_marks=[2, 0], input_mutated=False)
    assert compare(case, ga, cand) == [dict(oracle="P_USER", detail=[2, 0], delta="D1233_LAST_RESORT_USER_MARK")]
    cand = dict(out_tokens=90, user_marks=[2], input_mutated=False)
    assert compare(case, ga, cand)[0]["delta"] == "D1233_LAST_RESORT_USER_MARK"
    cand["out_tokens"] = 101
    assert compare(case, ga, cand) == [
        dict(oracle="R1", detail=dict(ga=90, candidate=101, limit=100), delta="D916_GA_MARKED_USER"),
        dict(oracle="P_USER", detail=[2], delta=None),
    ]
    cand.update(out_tokens=100, user_marks=[0, 2])
    assert compare(case, ga, cand)[0]["delta"] == "D1233_LAST_RESORT_USER_MARK"
    cand["out_tokens"] = 101
    failures = compare(case, ga, cand)
    assert failures == [dict(oracle="R1", detail=dict(ga=90, candidate=101, limit=100), delta="D916_GA_MARKED_USER"),
                        dict(oracle="P_USER", detail=[0, 2], delta=None)]
    ga["out_tokens"] = 101
    assert compare(case, ga, cand)[0]["delta"] == "D1233_LAST_RESORT_USER_MARK"
    cand["user_marks"] = [1]
    assert compare(case, ga, cand)[0]["delta"] is None
    cand["user_marks"] = []
    assert not last_resort_user_mark(case, ga, cand)
    assert compare(case, ga, cand) == []
