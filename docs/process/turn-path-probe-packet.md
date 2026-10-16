# Turn-path probe packet

Every spec and review packet for a change on the turn path carries these sections. The turn path covers assembly, compaction,
survival fit, ingest/re-ingest, the summary route, and the host-facing `compress`/`should_compress` contract. The packet changes
*what is compared*, not how many reviewers look. The culprit PRs behind the v0.27.0 respins had 16–21 reviews each.

## 1. Consumer enumeration
For every changed function, attribute or config key, list its call sites (`rg -n "<name>"` output, production and tests) and
how each one is affected. A reviewer walks this list; a consumer missing from it is a finding.

## 2. Previous-GA differential row
Paste the `tests/differential` summary for the change: the exact shapes and generated families run, oracle counts, and any
accepted delta used, with its predicate and reason. A new accepted delta needs its own justification. Blanket deltas are not
accepted.

## 3. Sibling shapes
List the shape family the change touches (see [fix-scope.md](fix-scope.md)) and the cells actually exercised.

## 4. Invariants
List the [invariant](../invariants.md) IDs the change could affect. For each one, give the test that would fail if the change
broke it.

## 5. Risk probes for the reviewer
Name concrete probes, each with an expected result: inputs, windows, and the state to compare. A cross-model reviewer is given
the probes. "Review this diff" is not a probe.

## Rules
- No behaviour change mixed with code moves in one PR.
- No hard PR-size cap.
