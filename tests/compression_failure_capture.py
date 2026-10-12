"""Opt-in PR1113 synthetic-fixture evidence; never a database repair."""

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys

FILES = ("lcm.db", "lcm.db-wal", "lcm.db-shm")
FILE_LIMIT = 8 * 1024 * 1024
TOTAL_LIMIT = 16 * 1024 * 1024
OUTPUT_NAME = ".ci-pr1113-failure"


def _open_directory(path):
    """Walk without following any directory symlinks (including ancestors)."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe directory")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def capture_failure(fixture, *, base, root, nodeid):
    """Copy only one failed synthetic fixture before shutdown; fail closed.

    Never queries SQLite, reads logs/environment, or serializes exception data.
    Reads may race background writes: hashes identify bytes copied, not a coherent
    SQLite snapshot. Capture failure must not replace the original test failure.
    """
    source_fd = root_fd = output_fd = None
    created = False
    try:
        fixture, base, root = map(Path, (fixture, base, root))
        if fixture.parent != base or fixture.name in ("", ".", ".."):
            raise ValueError("fixture outside pytest base")
        source_fd = _open_directory(fixture)
        root_fd = _open_directory(root)
        payloads = {}
        total = 0
        for name in FILES:
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd)
            except FileNotFoundError:
                if name == "lcm.db":
                    raise
                continue
            with os.fdopen(fd, "rb") as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > FILE_LIMIT:
                    raise ValueError("unsafe fixture file")
                data = source.read(FILE_LIMIT + 1)
                if len(data) > FILE_LIMIT:
                    raise ValueError("fixture grew beyond cap")
                total += len(data)
                if total > TOTAL_LIMIT:
                    raise ValueError("fixture exceeds total cap")
                payloads[name] = data
        # No recursive traversal; no partial artifact if input selection fails.
        os.mkdir(OUTPUT_NAME, mode=0o700, dir_fd=root_fd)
        created = True
        output_fd = os.open(OUTPUT_NAME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        sites = []
        traceback = sys.exc_info()[2]
        allowed_sources = {"test_compression_boundary.py", "engine.py", "reconcile.py", "store.py", "compaction.py"}
        while traceback is not None and len(sites) < 32:
            filename = Path(traceback.tb_frame.f_code.co_filename).name
            if filename in allowed_sources:
                sites.append({"file": filename, "line": traceback.tb_lineno})
            traceback = traceback.tb_next
        metadata = {
            "schema": 1,
            "diagnostic_only": True,
            "nodeid": nodeid if re.fullmatch(
                r"tests/test_compression_boundary\.py::test_resume_after_compaction_does_not_restore_tail"
                r"\[(inplace|rotation)-(True|False)-(6|7)-(1|2)\]", nodeid,
            ) else "private-synthetic-probe",
            "exception_sites": sites,
            "sqlite_database_error": isinstance(sys.exc_info()[1], sqlite3.DatabaseError),
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
            "capture_phase": "exception-before-engine-shutdown",
            "coherent_snapshot": False,
            "absent": [name for name in FILES if name not in payloads],
            "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()},
        }
        payloads["metadata.pending"] = (json.dumps(metadata, indent=2) + "\n").encode()
        for name, data in payloads.items():
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=output_fd)
            with os.fdopen(fd, "wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        # The final marker appears atomically only after all bounded files close.
        os.rename("metadata.pending", "metadata.json", src_dir_fd=output_fd, dst_dir_fd=output_fd)
        print("PR1113 diagnostic: captured bounded synthetic fixture (not a repair)", file=sys.stderr)
        return True
    except Exception:
        # Diagnostic failure never masks, passes, or xfails the original failure.
        if created and output_fd is not None:
            try:
                for name in ("metadata.json", "metadata.pending", *FILES):
                    try:
                        os.unlink(name, dir_fd=output_fd)
                    except FileNotFoundError:
                        pass
                os.rmdir(OUTPUT_NAME, dir_fd=root_fd)
            except OSError:
                # Suppress only diagnostic cleanup errors; no success marker.
                pass
        print("PR1113 diagnostic: capture refused/unavailable; original failure retained", file=sys.stderr)
        return False
    finally:
        for fd in (output_fd, root_fd, source_fd):
            if fd is not None:
                os.close(fd)
