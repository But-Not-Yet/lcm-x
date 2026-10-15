"""Pinned, isolated git archives; no checkout or plugin registration."""
import importlib
import io
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {"v0.26.2": "6800cf45ad05280e6cf6fa9e1ad96c62b7132646"}


class PrerequisiteUnavailable(RuntimeError):
    pass


def skip_or_fail(reason):
    import pytest
    if os.environ.get("CI"):
        pytest.fail(f"{reason}; CI must provide it")
    pytest.skip(reason)


def check_host():
    try:
        importlib.import_module("agent.context_engine")
    except ImportError as exc:
        raise PrerequisiteUnavailable("agent.context_engine host stub is not importable") from exc


def resolve(ref):
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify",
                             f"{BASELINES.get(ref, ref)}^{{commit}}"], capture_output=True, text=True)
    if result.returncode:
        raise PrerequisiteUnavailable(f"commit {ref!r} is unavailable; fetch it explicitly")
    return result.stdout.strip()


def materialize(ref, cache_dir):
    sha = resolve(ref)
    destination = Path(cache_dir) / sha
    if not destination.exists():
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", sha], check=True, capture_output=True).stdout
        destination.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(destination, filter="data")
    return destination
