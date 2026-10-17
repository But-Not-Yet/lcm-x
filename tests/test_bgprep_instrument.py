"""Red-on-wrong scoring fixtures: no product code or provider calls."""
from copy import deepcopy

import pytest

from bench.instruments.compaction_probe.bgprep.scoring import (
    at_least, coverage, intact, promotion, waits,
)


@pytest.fixture
def probe():
    return dict(expected=["user A", "reply A", "user B"],
                rows=[dict(store_id=i, identity=v) for i, v in enumerate(
                    ["user A", "reply A", "user B"], 1)],
                leaf_sources=[[1, 2]], frontier=2, active_view=["summary", "user B"])


def test_active_unsummarized_tail_is_covered(probe):
    score = coverage(probe)
    assert intact(score) and score["active_tail_ids"] == [3] and score["lost"] == 0


@pytest.mark.parametrize("corrupt,field", [
    ("missing_tail", "lost"), ("missing_stored", "missing_rows"),
    ("duplicate_leaf", "duplicate_coverage"), ("duplicate_stored", "duplicate_stored"),
    ("duplicate_id", "duplicate_ids"), ("dangling", "dangling_sources"),
])
def test_wrong_coverage_fails(probe, corrupt, field):
    if corrupt == "missing_tail":
        probe["active_view"] = ["summary"]
    elif corrupt == "missing_stored":
        probe["rows"].pop()
    elif corrupt == "duplicate_leaf":
        probe["leaf_sources"].append([2])
    elif corrupt == "duplicate_stored":
        probe["rows"].append(dict(store_id=4, identity="user A"))
    elif corrupt == "duplicate_id":
        probe["rows"].append(deepcopy(probe["rows"][0]))
    elif corrupt == "dangling":
        probe["leaf_sources"].append([99])
    score = coverage(probe)
    assert score[field] and not intact(score)


@pytest.mark.parametrize("mutation", ["reverse", "rewrite", "frontier"])
def test_identity_chronology_and_frontier_fail_closed(probe, mutation):
    if mutation == "reverse":
        probe["rows"].reverse()
    elif mutation == "rewrite":
        probe["rows"][0]["identity"] = "changed"
    else:
        probe["frontier"] = 3  # live raw row behind published frontier is not an active tail
    assert not intact(coverage(probe))


def test_promotion_counts_only_threshold_crossings():
    score = promotion([dict(crossing=True, promoted=2, leaf_compactions=3),
                       dict(crossing=True, promoted=1, leaf_compactions=2),
                       dict(crossing=False, promoted=9, leaf_compactions=9)])
    assert score == dict(numerator=3, denominator=5, rate=.6)
    assert promotion([])["rate"] is None
    assert not at_least(dict(numerator=0, denominator=0), score)
    with pytest.raises(ValueError):
        promotion([dict(crossing=True, promoted=3, leaf_compactions=2)])


def test_failed_foreground_leaf_still_counts_as_ran():
    score = promotion([dict(crossing=True, promoted=1, leaves=1, leaf_compactions=2)])
    assert score["rate"] == .5 and score["denominator"] == 2


def test_pooled_rate_is_not_seed_rate_mean_and_equality_passes():
    # Seeds 1/1 and 0/9 pool to 1/10, not the misleading mean 1/2.
    pooled = dict(numerator=1, denominator=10)
    reference = dict(numerator=1, denominator=5)
    assert not at_least(pooled, reference)
    assert at_least(reference, reference)
    assert at_least(dict(numerator=3, denominator=10), reference)


def test_wait_nearest_rank_zeros_and_denominators():
    values = [99, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
    score = waits([dict(turn=i, crossing=i == 1, visible_wait_s=v) for i, v in enumerate(values)])
    assert score["turns"] == 10 and score["waited_turns"] == 2
    assert score["p90_s"] == 1 and score["waited_turn_share"] == .2
