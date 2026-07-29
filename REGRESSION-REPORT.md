# Regression report — stress CLI canary false negatives

## Affected test

`tests/test_stress_release_check.py::test_stress_cli_smoke_writes_results_summary_and_uses_output_sandbox`

## Verdict

GENUINE REGRESSION. The stress CLI exits 1 even though `lcm_grep` returns the
requested canary rows.

## Mechanism

The #168 retrieval redesign keeps compound canary queries on FTS. FTS snippets
insert `>>>` and `<<<` around matched terms, splitting a planted token such as
`CANARY_SCOPE_A_000` into
`>>>CANARY<<<_>>>SCOPE<<<_>>>A<<<_>>>000<<<`.

`benchmarking/stress.py` checks the serialized grep response for the original
contiguous token. The correct row is present, but the marker-split snippet makes
that string check false. The smoke run therefore reports:

- `grep_canary_recall_miss`
- `all_scope_missing_cross_session_hit`
- `explicit_session_scope_missing_hit`

This is a release-check CLI defect, not a retrieval miss and not a test
environment assumption. Per `SPEC.md`, product code is unchanged and the test
remains failing for the product-fix lane.

## Minimal repro

From worktree head `2edb8fc8e08cf533a4336e2d3d5c99d7772789b8`:

```sh
PYTHONPATH=/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/agent-stub \
/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/venv-ci-repro/bin/python \
scripts/lcm_stress_check.py \
  --output /Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-mono-pr-rounds/artifacts/laneA-logs/stress-cli-repro \
  --tier smoke \
  --json
```

Observed: exit 1, `failure_count: 3`, correct grep rows present with marker-split
canary snippets, and empty stderr.

## Evidence

- `laneA-logs/stress-cli-stdout.log`
- `laneA-logs/stress-cli-stderr.log`
- `laneA-logs/stress-cli-repro/results/stress-results.json`
