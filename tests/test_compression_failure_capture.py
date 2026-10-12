"""Safety and private benign-failure checks for the temporary PR1113 proposal."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from tests import compression_failure_capture as capture


@pytest.fixture
def paths(tmp_path):
    base = tmp_path / "pytest-base"
    fixture = base / "synthetic"
    fixture.mkdir(parents=True)
    root = tmp_path / "checkout"
    root.mkdir()
    return fixture, base, root


def collect(paths, nodeid="private"):
    fixture, base, root = paths
    return capture.capture_failure(fixture, base=base, root=root, nodeid=nodeid)


def test_exact_allowlist_and_hashes(paths):
    fixture, _, root = paths
    expected = {name: ("synthetic " + name).encode() for name in capture.FILES}
    for name, data in expected.items():
        (fixture / name).write_bytes(data)
    (fixture / ".env").write_text("PRIVATE_SENTINEL")
    (fixture / "home").mkdir()
    (fixture / "home" / "history.db").write_text("PRIVATE_SENTINEL")
    assert collect(paths, "SECRET_NODEID" * 100)
    output = root / capture.OUTPUT_NAME
    assert {p.name for p in output.iterdir()} == {*capture.FILES, "metadata.json"}
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["nodeid"] == "private-synthetic-probe"
    for name, data in expected.items():
        assert (output / name).read_bytes() == data
        assert metadata["files"][name] == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert "PRIVATE_SENTINEL" not in (output / "metadata.json").read_text()
    assert not collect(paths)  # First failure only, never overwrite evidence.


def test_missing_sidecars_are_explicit(paths):
    fixture, _, root = paths
    (fixture / "lcm.db").write_bytes(b"synthetic")
    assert collect(paths)
    metadata = json.loads((root / capture.OUTPUT_NAME / "metadata.json").read_text())
    assert metadata["absent"] == ["lcm.db-wal", "lcm.db-shm"]


@pytest.mark.parametrize("kind", ["outside", "traversal", "directory_link", "db_link", "sidecar_link", "hardlink", "fifo", "missing_db", "oversize", "total_size", "output_link"])
def test_fail_closed(paths, monkeypatch, kind):
    fixture, base, root = paths
    (fixture / "lcm.db").write_bytes(b"synthetic")
    secret = root / "unrelated.db"
    secret.write_text("PRIVATE_SENTINEL")
    if kind == "outside":
        fixture = root
    elif kind == "traversal":
        fixture = base / ".." / "pytest-base" / "synthetic"
    elif kind == "directory_link":
        (base / "link").symlink_to(fixture, target_is_directory=True)
        fixture = base / "link"
    elif kind in ("db_link", "sidecar_link"):
        name = "lcm.db" if kind == "db_link" else "lcm.db-wal"
        (fixture / name).unlink(missing_ok=True)
        (fixture / name).symlink_to(secret)
    elif kind == "hardlink":
        (fixture / "lcm.db").unlink()
        os.link(secret, fixture / "lcm.db")
    elif kind == "fifo":
        os.mkfifo(fixture / "lcm.db-wal")
    elif kind == "missing_db":
        (fixture / "lcm.db").unlink()
    elif kind == "oversize":
        monkeypatch.setattr(capture, "FILE_LIMIT", 2)
    elif kind == "total_size":
        monkeypatch.setattr(capture, "TOTAL_LIMIT", 2)
    elif kind == "output_link":
        (root / capture.OUTPUT_NAME).symlink_to(base, target_is_directory=True)
    assert not collect((fixture, base, root))
    assert not (root / capture.OUTPUT_NAME / "metadata.json").exists()
    assert secret.read_text() == "PRIVATE_SENTINEL"


def test_partial_output_is_not_uploadable(paths, monkeypatch):
    fixture, _, root = paths
    (fixture / "lcm.db").write_bytes(b"synthetic")
    original = capture.os.open

    def fail_write(path, flags, *args, **kwargs):
        if path == "metadata.pending":
            raise OSError("benign output error")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(capture.os, "open", fail_write)
    assert not collect(paths)
    assert not (root / capture.OUTPUT_NAME).exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_marker_is_atomic(paths, monkeypatch, interrupt):
    fixture, _, root = paths
    (fixture / "lcm.db").write_bytes(b"synthetic")

    def prevent_publication(*args, **kwargs):
        assert (root / capture.OUTPUT_NAME / "metadata.pending").exists()
        assert not (root / capture.OUTPUT_NAME / "metadata.json").exists()
        if interrupt:
            raise KeyboardInterrupt("benign private interruption")
        raise OSError("benign private publication failure")

    monkeypatch.setattr(capture.os, "rename", prevent_publication)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            collect(paths)
    else:
        assert not collect(paths)
    assert not (root / capture.OUTPUT_NAME / "metadata.json").exists()


def test_callback_error_preserves_original(paths, monkeypatch):
    from tests import test_compression_boundary as boundary

    def original(*args, **kwargs):
        raise AssertionError("original benign assertion")

    def callback():
        raise OSError("diagnostic callback failure")

    monkeypatch.setattr(boundary.LCMEngine, "ingest", original)
    with pytest.raises(AssertionError, match="original benign assertion"):
        boundary._run_host_commit_sequence(
            paths[0], monkeypatch, in_place=False, merge=True, tail=7,
            failure_capture=callback,
        )


@pytest.mark.parametrize("fail", [False, True])
def test_benign_pytest_exit_and_live_wal(tmp_path, fail):
    """A normal AssertionError exits 1; DB/WAL/SHM are copied before close."""
    probe = tmp_path / "probe"
    probe.mkdir()
    source = probe / "test_probe.py"
    source.write_text('''import hashlib, json, pathlib, sqlite3
from compression_failure_capture import capture_failure

def test_benign(tmp_path):
    conn = sqlite3.connect(tmp_path / "lcm.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE synthetic(value TEXT)")
    conn.execute("INSERT INTO synthetic VALUES ('ordinary synthetic fixture')")
    conn.commit()
    expected = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                for name in ("lcm.db", "lcm.db-wal", "lcm.db-shm")}
    pathlib.Path("expected.json").write_text(json.dumps(expected))
    try:
        assert FAIL is False, "ordinary benign synthetic assertion"
    except Exception:
        capture_failure(tmp_path, base=tmp_path.parent, root=pathlib.Path.cwd(), nodeid="private")
        raise
    finally:
        conn.close()
'''.replace("FAIL", repr(fail)))
    env = os.environ.copy()
    # Dependency path only; do not serialize or dump process environment.
    env["PYTHONPATH"] = str(Path(capture.__file__).parent)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--confcutdir", str(probe), str(source)],
                            cwd=probe, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (1 if fail else 0), result.stdout + result.stderr
    output = probe / capture.OUTPUT_NAME
    if not fail:
        assert not output.exists()
        return
    assert "ordinary benign synthetic assertion" in result.stdout
    expected = json.loads((probe / "expected.json").read_text())
    assert {p.name for p in output.iterdir()} == {*capture.FILES, "metadata.json"}
    for name, digest in expected.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    # Inspect the copied synthetic fixture privately; original files are untouched.
    with sqlite3.connect(output / "lcm.db") as db:
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert db.execute("SELECT value FROM synthetic").fetchall() == [("ordinary synthetic fixture",)]


def test_real_driver_captures_before_shutdown(paths, monkeypatch):
    from tests import test_compression_boundary as boundary

    fixture, _, root = paths

    def benign_failure(*args, **kwargs):
        raise AssertionError("ordinary driver assertion")

    monkeypatch.setattr(boundary.LCMEngine, "ingest", benign_failure)
    with pytest.raises(AssertionError, match="ordinary driver assertion"):
        boundary._run_host_commit_sequence(
            fixture, monkeypatch, in_place=False, merge=True, tail=7, restart_after=1,
            failure_capture=lambda: collect(paths),
        )
    metadata = json.loads((root / capture.OUTPUT_NAME / "metadata.json").read_text())
    assert set(metadata["files"]) == set(capture.FILES)
    assert metadata["exception_sites"]
    assert metadata["sqlite_database_error"] is False
