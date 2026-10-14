"""#1162: distinguish no-call survival fitting from a recorded summary failure."""
import json

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.tools import lcm_doctor, lcm_grep, lcm_load_session, lcm_status
from tests import test_issue_581_582_store_complete_survival as fit
from tests import test_issue_1162_retained_escape as retained

FITTED = ("To keep this conversation within the model's context window, {n} older messages left live context "
          "before LCM could summarise them; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
          "/lcm doctor reports why.")
FAILED = ("LCM could not summarise part of this conversation in time. To keep the session alive, {n} older "
          "messages left live context; they stay stored verbatim and searchable (lcm_grep, lcm_load_session). "
          "/lcm doctor reports it.")
summaries = retained.summaries
host_estimator = fit.host_estimator


@pytest.fixture
def no_call(tmp_path, monkeypatch, summaries):
    engine, view = retained._reorder(tmp_path, False)
    # Reach the retained stop with the existing repro, before applying a fit budget.
    for _ in range(3):
        view = engine.compress(view, current_tokens=10_000)
        engine.note_turn_complete()
        engine._no_progress_hold = None
        if engine._last_compression_noop_reason.startswith(retained.RETAINED):
            break
    assert engine._last_compression_noop_reason.startswith(retained.RETAINED)
    summaries.clear()
    monkeypatch.setattr(engine, "_survival_fit_budget", lambda *args, **kwargs: 850)
    anchored = []
    original = engine._identity_anchor_summary_input

    def anchor(*args, **kwargs):
        result = original(*args, **kwargs)
        anchored.append(result)
        return result

    monkeypatch.setattr(engine, "_identity_anchor_summary_input", anchor)
    records = []
    record = engine._survival_record

    def capture(reason, count, ids, *args, **kwargs):
        records.append(list(ids))
        return record(reason, count, ids, *args, **kwargs)

    monkeypatch.setattr(engine, "_survival_record", capture)
    try:
        stored = retained._rows(engine)
        result = engine.compress(view, current_tokens=engine._survival_measure(view))
        assert anchored == [[]] and summaries == []
        assert engine._last_compression_noop_reason.startswith(retained.RETAINED)
        assert len(result) < len(view) and len(records) == 1 and records[0]
        assert engine._last_survival_fit["uncovered_rows"] == len(records[0])
        yield engine, records[0], stored
    finally:
        engine.shutdown()


def test_regression_no_call_warning(no_call):
    engine, ids, _ = no_call
    warning = engine._survival_fit_pending_warning[1]
    assert warning == FITTED.format(n=len(ids)) and "in time" not in warning


def test_regression_no_call_cause(no_call):
    engine, _, _ = no_call
    cause = engine._last_compression_noop_reason
    assert engine._last_survival_fit["cause"] == cause
    assert engine._store.read_metadata_json(survival.SURVIVAL_FIT_COUNTER_KEY)["last_cause"] == cause


def test_regression_status_cause(no_call):
    engine, _, _ = no_call
    assert json.loads(lcm_status({}, engine=engine))["last_survival_fit"]["cause"] == (
        engine._last_compression_noop_reason)


def test_regression_command_doctor_cause(no_call):
    engine, _, _ = no_call
    observation = next(line for line in handle_lcm_command("doctor", engine).splitlines()
                       if "survival_fit: " in line)
    assert f"last cause {engine._last_compression_noop_reason}" in observation


def test_regression_tool_doctor_cause(no_call):
    engine, _, _ = no_call
    report = json.loads(lcm_doctor({}, engine=engine))
    check = next(item for item in report["checks"] if item["check"] == "survival_fit")
    assert check["detail"]["last_cause"] == engine._last_compression_noop_reason
    guidance = next(item for item in report["guidance"] if item["check"] == "survival_fit")
    assert engine._last_compression_noop_reason in guidance["rationale"]


def test_control_fitted_rows_remain_retrievable(no_call):
    engine, ids, stored = no_call
    loaded = json.loads(lcm_load_session({"session_id": "S", "limit": 100}, engine=engine))
    loaded_by_id = {row["store_id"]: row for row in loaded["messages"]}
    for row in stored:
        if row["store_id"] not in ids:
            continue
        assert loaded_by_id[row["store_id"]]["content"] == row["content"]
        hits = json.loads(lcm_grep({"query": row["content"], "content_scope": "history",
                                   "limit": 100}, engine=engine))
        assert row["store_id"] in {hit["store_id"] for hit in hits["results"]}
    assert retained._rows(engine) == stored


def test_control_summary_timeout_keeps_warning(tmp_path, monkeypatch, host_estimator):
    engine = fit._engine(tmp_path, context_length=fit.WINDOW)
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("synthetic summary timeout")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", timeout)
    try:
        view = fit._long_view()
        engine.ingest(view)
        engine.compress(view, current_tokens=host_estimator(view))
        assert calls
        assert engine._survival_fit_pending_warning[1] == FAILED.format(
            n=engine._last_survival_fit["dropped_rows"])
    finally:
        engine.shutdown()


@pytest.fixture
def publication(tmp_path, monkeypatch, summaries, host_estimator):
    engine = fit._engine(tmp_path, context_length=fit.WINDOW)

    def conflict(*args, **kwargs):
        raise LifecyclePublicationConflictError("synthetic publication conflict")

    monkeypatch.setattr(engine._lifecycle, "stage_compaction_publication", conflict)
    try:
        view = fit._long_view()
        engine.ingest(view)
        engine.compress(view, current_tokens=host_estimator(view))
        assert summaries and engine._last_survival_fit["reason"] == "publication_invariant_conflict"
        yield engine
    finally:
        engine.shutdown()


def test_regression_publication_warning(publication):
    assert publication._survival_fit_pending_warning[1] == FITTED.format(
        n=publication._last_survival_fit["dropped_rows"])


def test_control_publication_reason_is_unchanged(publication):
    counter = publication._store.read_metadata_json(survival.SURVIVAL_FIT_COUNTER_KEY)
    assert publication._last_survival_fit["reason"] == counter["last_reason"] == "publication_invariant_conflict"


@pytest.mark.parametrize("reason", ["sanitized", "compacted", "unknown", "sqlite_publication_locked",
                                  "lifecycle_binding_changed", "exception:TimeoutError"])
def test_regression_unproven_reason_is_neutral(tmp_path, reason):
    engine = retained._engine(tmp_path, False)
    try:
        engine._survival_record(reason, 1, [], 900, 500, 600, False, "")
        assert engine._survival_fit_pending_warning[1] == FITTED.format(n=1)
    finally:
        engine.shutdown()
