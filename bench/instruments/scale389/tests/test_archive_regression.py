from __future__ import annotations

import os

import pytest

from bench.instruments.scale389.archive_regression import (
    DEFAULT_DATASET,
    DEFAULT_QEVAL,
    DEFAULT_RESULTS,
    DEFAULT_SIDECAR,
    run_regression,
)
from bench.tools.benchpaths import bench_path


@pytest.mark.skipif(
    not os.environ.get("LCMX_BENCH_ROOT") or not all(
        bench_path(path).exists()
        for path in (DEFAULT_RESULTS, DEFAULT_QEVAL, DEFAULT_DATASET, DEFAULT_SIDECAR)
    ),
    reason="the machine-local read-only F34 archive is unavailable",
)
def test_archived_f34_outputs_reproduce_and_turn_join_is_non_degenerate():
    result = run_regression(
        bench_path(DEFAULT_RESULTS),
        bench_path(DEFAULT_QEVAL),
        bench_path(DEFAULT_DATASET),
        bench_path(DEFAULT_SIDECAR),
    )

    assert result["archive_read_only"] is True
    assert result["session_regression"]["status"] == "exact"
    assert result["session_regression"]["published_table"]["A3"] == {
        "500": 0.82,
        "2000": 0.60,
        "8000": 0.34,
        "19829": 0.233,
    }
    assert sorted(result["turn_join_check"]["values"]) == [0, 1]
