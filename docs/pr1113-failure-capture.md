# PR1113 temporary synthetic-failure capture (diagnostic proposal)

This separate commit is evidence collection for PR #1113's unexplained CI failure,
not a SQLite repair or proof that #1090 is fixed in production. The previous run
38183103292 failed on Python 3.11 in
`test_resume_after_compaction_does_not_restore_tail[rotation-True-7-1]` with
`sqlite3.DatabaseError: database disk image is malformed`; Python 3.12–3.14 were
cancelled by the existing fail-fast policy. No failed database was retained.
Local successful repetitions did not establish a root cause or an infrastructure
flake. The original performance change and existing baseline exceptions remain
unchanged and disclosed in the PR.

## Scope and defaults

The existing low-FD `python -m pytest tests/ -q` command, assertions, test matrix,
scheduling, fail-fast policy, required check names, and failure exit are unchanged.
Only PR1113 from `merodahero/lcm-x` sets `LCM_PR1113_CAPTURE=1`. Unset/other values
leave the test seam inert. Only the resume-after-compaction parametrizations use
that seam; other tests do not capture. The test driver catches an ordinary
exception only to copy evidence before its existing `engine.shutdown()` finally,
then immediately re-raises. Successful calls create no artifact. Assertions made
after the driver returns, setup/teardown errors, and shutdown failures are outside
this narrowly targeted capture boundary. No engine/store/reconciliation product
code is modified, and no runtime host contract or new dependency is introduced.

## Privacy, bounds, and limitations

- Input is the test's own immediate child of pytest's basetemp, not a discovered
  runtime database. Only `lcm.db`, `lcm.db-wal`, `lcm.db-shm` are read. No directory
  recursion, externalized payloads, home/history, `.env`, logs or environment dumps.
- Directory ancestors and files are opened without following symlinks. Nonregular
  files, hard links, traversal, outside paths, missing DB, and oversized selection
  are refused. Read handles are closed; files are never queried or modified.
- Maximum 8 MiB per file and 16 MiB total payload, plus small bounded metadata.
  One capture per matrix job; later failures cannot overwrite it. Missing optional
  sidecars are recorded explicitly. Files are flushed/fsynced/closed before an
  atomic rename publishes the final metadata marker; interrupted writes have no
  upload marker. Partial output is cleaned up on ordinary write failure.
- Metadata contains only validated test ID, Python/linked SQLite versions,
  byte lengths/hashes, absent sidecars, a SQLite-error boolean and at most 32
  allowlisted source-basename/line sites. No exception text, locals or full paths.
  Existing GitHub job logs remain the test-failure log; they are not duplicated
  into the artifact.
- The source may still have background writers. These raw copies are NOT an atomic
  or guaranteed coherent SQLite snapshot. Do not mistake copy-time inconsistency
  for original corruption; preserve raw hashes and inspect a working copy only.
  Capture performs no checkpoint, backup API, integrity check or scheduler drain,
  and does not change failure semantics. File I/O happens only after a failure.
- Upload explicitly allows only those four files, uses the already pinned
  repository upload-artifact action, retains evidence for 3 days, and runs only
  for this failed fork PR job with a completed capture. Nothing from a live user
  store is eligible. The synthetic database contains only repository test data.
- Capture refusal prints a fixed notice and retains the original failure. A
  cancellation/crash or failure outside the seam may still yield no artifact.
  A passing CI run is not proof of repair, flakiness, or causation.

## Verification and removal

`tests/test_compression_failure_capture.py` exercises exact allowlisting/hashes,
missing sidecars, bounds, path/link/FIFO refusal, first-failure-only and partial
write behavior. Its private benign pytest assertion preserves exit 1 and exact
live DB/WAL/SHM hashes before close; its success counterpart preserves exit 0 and
creates no output. A real-engine driver probe verifies the pre-shutdown seam.
Run it and the original controls with:

```sh
pytest tests/test_compression_failure_capture.py tests/test_compression_boundary.py tests/test_issue_1090_replay_prefix_scaling.py -q
LCM_PR1113_CAPTURE=1 pytest tests/test_compression_boundary.py -q
ruff check .
actionlint
```

This is a narrowly scoped diagnostic proposal, not maintainer pre-approval of a
new project-wide policy. Maintainers may reject/remove it or ask for a separate
issue. Remove it with a normal revert of this separate commit, never a force push;
the #1090 performance commit remains independent. No merge/release/deployment or
LCM enablement is authorized by this capture experiment.
