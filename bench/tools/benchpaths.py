"""Resolve optional benchmark data only when an instrument needs it."""

import os
from pathlib import Path


def bench_path(relative: str | Path) -> Path:
    root = os.environ.get("LCMX_BENCH_ROOT")
    if not root:
        raise SystemExit("set LCMX_BENCH_ROOT to the bench data root")
    return Path(root) / relative
