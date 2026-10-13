"""#1122: best-effort extraction reserves the next foreground summary call."""

import logging
import time

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from tests.test_issue_605_foreground_budget import (
    _advance_on,
    _engine,
    _leaves,
    _pin_lcm_char_counter as _pin_lcm_char_counter,
    _provider,
    _view,
)


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        offset = 0.0

        def __call__(self):
            return 1000.0 + self.offset

    fake = Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


@pytest.mark.parametrize("sweep_enabled", [True, False])
@pytest.mark.parametrize("route_estimate", [30.0, 45.0])
def test_extraction_timeout_reserves_next_summary_and_writes_leaf(
    tmp_path, monkeypatch, clock, sweep_enabled, route_estimate
):
    engine = _engine(
        tmp_path,
        extraction_enabled=True,
        summary_model="primary",
        summary_timeout_ms=LCMConfig().summary_timeout_ms,
        foreground_soft_seconds=60,
        foreground_hard_seconds=120,
        threshold_full_sweep_enabled=sweep_enabled,
    )
    if route_estimate != 30.0:
        engine._foreground_estimates.record_call("primary", route_estimate)
    provider = _provider(monkeypatch, clock, default=0.25)
    extraction_calls = []

    def hanging_extraction(**kwargs):
        budget = engine._foreground_budget
        now = time.monotonic()
        estimate = budget.estimate(engine._primary_summary_route())
        extraction_calls.append(
            (kwargs["timeout"], budget.usable_deadline - now - estimate, estimate)
        )
        clock.offset += kwargs["timeout"]
        raise TimeoutError("extraction timed out")

    monkeypatch.setattr(lcm_engine, "extract_before_compaction", hanging_extraction)
    try:
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert LCMConfig().summary_timeout_ms == 120_000
        assert len(extraction_calls) == 1
        timeout, remaining_for_extraction, estimate = extraction_calls[0]
        assert estimate == route_estimate
        assert timeout <= remaining_for_extraction
        assert len(provider.calls) == 1
        assert provider.calls[0][1] >= route_estimate
        assert len(_leaves(engine)) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep_enabled", [True, False])
def test_disabled_extraction_is_not_called(tmp_path, monkeypatch, clock, sweep_enabled):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=sweep_enabled)
    provider = _provider(monkeypatch, clock, default=0.25)
    extraction_calls = []
    monkeypatch.setattr(
        lcm_engine,
        "extract_before_compaction",
        lambda **kw: extraction_calls.append(kw),
    )
    try:
        assert engine._config.extraction_enabled is False
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert extraction_calls == []
        assert provider.calls and _leaves(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep_enabled", [True, False])
@pytest.mark.parametrize("pre_work, expected_leaves", [(84.5, 1), (101.0, 0)])
def test_extraction_skipped_when_reservation_leaves_less_than_one_second(
    tmp_path, monkeypatch, clock, caplog, sweep_enabled, pre_work, expected_leaves
):
    engine = _engine(
        tmp_path,
        extraction_enabled=True,
        summary_timeout_ms=LCMConfig().summary_timeout_ms,
        foreground_soft_seconds=60,
        foreground_hard_seconds=120,
        threshold_full_sweep_enabled=sweep_enabled,
    )
    provider = _provider(monkeypatch, clock, default=0.25)
    extraction_calls = []
    monkeypatch.setattr(
        lcm_engine,
        "extract_before_compaction",
        lambda **kw: extraction_calls.append(kw),
    )
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, pre_work)
    try:
        view = _view()
        engine.ingest(view)
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
            engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert extraction_calls == []
        assert len(provider.calls) == expected_leaves
        assert len(_leaves(engine)) == expected_leaves
        skip_lines = [
            record
            for record in caplog.records
            if "Pre-compaction extraction skipped" in record.getMessage()
        ]
        assert len(skip_lines) == 1
        assert skip_lines[0].levelno == logging.DEBUG
        assert "next summary call" in skip_lines[0].getMessage()
        if not expected_leaves:
            assert "reason=time_budget_exhausted" in caplog.text
    finally:
        engine.shutdown()
