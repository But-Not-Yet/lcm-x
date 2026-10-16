"""Resolve optional benchmark data only when an instrument needs it."""

from __future__ import annotations

import os
from pathlib import Path


def bench_path(relative: str | Path, *, override: str | None = None) -> Path:
    root = os.environ.get("LCMX_BENCH_ROOT")
    if not root:
        hint = f" (or {override})" if override else ""
        raise SystemExit(f"set LCMX_BENCH_ROOT{hint} to resolve {relative}")
    return Path(root) / relative
