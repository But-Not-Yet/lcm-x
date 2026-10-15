"""Deterministic occurrence seeds and an unsuppressed machine probe for #1179."""
from copy import deepcopy
import sys
import types

import pytest

pytest.importorskip("hypothesis")
from hypothesis import settings, strategies as st
from hypothesis.stateful import invariant, rule, run_state_machine_as_test

from tests.invariants.test_exactly_once_machine import ExactlyOnceMachine, assert_occurrences, identity


class RetryCollapse(AssertionError):
    """Only the precise #1179 missing-prompt result, not an arbitrary failure."""


class RetryDiscoveryMachine(ExactlyOnceMachine):
    include_known_open = True

    @invariant()
    def exactly_once(self):
        try:
            super().exactly_once()
        except AssertionError as error:
            index = self.retry_occurrence_index
            wanted = list(map(identity, self.occurrences))
            actual = list(map(identity, self.rows()))
            if index is not None and actual == wanted[:index] + wanted[index + 1:]:
                raise RetryCollapse(str(error)) from error
            raise


@pytest.mark.xfail(strict=True, reason="#1179: retry loses a new equal-text user occurrence", raises=RetryCollapse)
def test_machine_discovers_retry_1179():
    run_state_machine_as_test(RetryDiscoveryMachine)


@pytest.mark.xfail(strict=True, reason="#1179: retry loses a new equal-text user occurrence", raises=RetryCollapse)
def test_retry_1179():
    machine = RetryDiscoveryMachine()
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.new_turn("same text", False, 0, False, "lookup")
        machine.exactly_once()
        machine.new_turn("same text", False, 0, True, "lookup")
        machine.exactly_once()
    finally:
        machine.teardown()


def test_oracle_reports_duplicates_and_collapses_separately():
    row = {"role": "user", "content": "same text"}
    with pytest.raises(AssertionError, match=r"duplicates=\{.+collapses=\{\}"):
        assert_occurrences([row], [row, row])
    with pytest.raises(AssertionError, match=r"duplicates=\{\}; collapses=\{.+"):
        assert_occurrences([row, row], [row])


def test_oracle_checks_order_and_assistant_tool_metadata():
    user = {"role": "user", "content": "same text"}
    assistant = {"role": "assistant", "content": "same text"}
    with pytest.raises(AssertionError, match="order_mismatch=True"):
        assert_occurrences([user, assistant], [assistant, user])
    for changed in ({"id": "different", "function": {"name": "lookup"}},
                    {"id": "call", "function": {"name": "different"}}):
        expected = dict(assistant, tool_calls=[{"id": "call", "function": {"name": "lookup"}}])
        with pytest.raises(AssertionError, match="collapses="):
            assert_occurrences([expected], [dict(assistant, tool_calls=[changed])])


def test_survival_carrier_1000_is_already_fixed(tmp_path, monkeypatch):
    """#1000 remains open for GA, but #1001 already fixed this rc's carrier.

    Survival fit is outside the append-only machine. Reuse the established
    held-fit construction and apply the same complete-row oracle to its replay.
    """
    from hermes_lcm import escalation, tokens
    from tests.test_issue_650_survival_fit_keeps_summary import _held_fit_after_inplace_boundary, _rough

    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = _rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *args, **kwargs:
                        "Earlier turns." + " summary" * 150 + "\nExpand for details about: turns")
    engine, fitted, _remainder = _held_fit_after_inplace_boundary(tmp_path, _rough, monkeypatch)
    try:
        expected = deepcopy(engine._store.get_session_messages("S"))
        reply = {"role": "assistant", "content": "reply CUR", "timestamp": 30_001.0}
        fitted[0]["timestamp"] = 40_000.0
        engine.ingest([*fitted, reply])
        assert_occurrences([*expected, reply], engine._store.get_session_messages("S"))
    finally:
        engine.shutdown()
        tokens._count_tokens_cached.cache_clear()


class HistoricalSteerMachine(ExactlyOnceMachine):
    """#633 sensitivity probe, outside the main append-only input domain.

    A known host prefix insertion supplies the historical seed. The same ordered
    oracle checks one transition; ordinary machine operations stay unchanged.
    """

    def __init__(self):
        super().__init__()
        self.steer = {"role": "user", "content": "repeat this instruction", "display_kind": "steer"}
        self.host = [self.stamp({"role": "user", "content": "plan"}), deepcopy(self.steer),
                     self.stamp({"role": "assistant", "content": "planned"}),
                     self.stamp({"role": "user", "content": "inspect"}),
                     self.stamp({"role": "assistant", "content": "", "tool_calls": [
                         {"id": "historical-call", "type": "function",
                          "function": {"name": "lookup", "arguments": "{}"}}]}),
                     self.stamp({"role": "tool", "content": "done", "tool_call_id": "historical-call"}),
                     self.stamp({"role": "assistant", "content": "inspected"})]
        self.occurrences = deepcopy(self.host)
        self.engine.ingest(self.host)
        self.exactly_once()

    @rule(text=st.sampled_from(("continue", "verify")))
    def new_turn(self, text):
        steer = deepcopy(self.steer)
        batch = [self.stamp({"role": "user", "content": text}),
                 self.stamp({"role": "assistant", "content": "completed " + text})]
        self.host.insert(6, steer)
        self.host.extend(batch)
        self.occurrences.extend(deepcopy([steer, *batch]))
        self.engine.ingest(self.host)


def test_historical_633_machine():
    run_state_machine_as_test(HistoricalSteerMachine, settings=settings(stateful_step_count=1))
