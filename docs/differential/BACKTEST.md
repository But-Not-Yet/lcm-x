# Previous-GA differential backtest

Measured 2026-10-15 against GA `6800cf45ad05280e6cf6fa9e1ad96c62b7132646`.
Product candidate: rc7 `432bc4567c7b6201f83421bb9e5402eee7980302`.
S1b starts at harness head `d0a27b259cd5d1d04b974aeb7dee70ceadb810e0`, whose production files match rc7.
The original historical receipts used checkout `3c9b34bc1165905869369c7d93f914aec95fdf5c`.
S1b corrects R2's stale objective probe, compares managed GA defaults by default, and adds C1 summary coverage.
The 32K #1247 shape has 734 uncovered rows on both managed GA and rc7; the bare-GA comparison exposes the gap.
The absolute zero-gap expectation is retained as a dedicated strict xfail. Two exact #1123 R3 cells are accepted
separately: 40K compress runs and 64K provider calls. No production code changed.
This is advisory synthetic evidence. It is not a release gate or proof of runtime readiness.

## S1b corrected public comparison

The newest-user objective rule replaces the static first-turn R2 probe. GA now defaults to the managed
configuration; select `--baseline-config bare|managed` or `LCM_DIFFERENTIAL_BASELINE_CONFIG` (default `managed`).
Only GA receives this selection; candidate defaults and explicit per-case overrides remain in force.
Managed values are `context_threshold=0.75`, `fresh_tail_count=24`, `fresh_tail_max_tokens=24000`,
`leaf_chunk_tokens=8000`, `threshold_full_sweep_enabled=True`, `large_output_externalization_enabled=True`,
and `large_output_active_replay_stubbing_enabled=True`. Explicit case overrides take precedence.

C1 counts non-system stored rows absent from the final live view and from all DAG message-source coverage.
Unique synthetic row sentinels identify live rows; summary-source IDs are not message IDs.
C1 is red only when candidate exceeds GA by more than two rows. This measures summary coverage, not store loss.
The absolute `i1247_*` strict xfail requires zero uncovered rows on the 32K i1139 shape and catches only
`SummaryCoverageGap`; a driver error or unrelated assertion cannot satisfy it.

Corrected receipts and the table below supersede the original public triage and acceptance results retained
under the historical sections. Historical rows used the bare GA defaults and the old static objective probe.

### rc7 public-layer results after correction

All 72 public cases (12 exact plus 60 generated), seed 1227. Counts are raw oracle events.
Managed receipts come from the final enabled pytest run; bare GA replays the identical saved inputs and
reuses those validated candidate records. No candidate result is regenerated or edited for the bare arm.

| GA config | Cases | R1 | R2 | R3 | C1 | A1 | A2 | Accepted | Unabsorbed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| managed | 72 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| bare | 72 | 0 | 0 | 3 | 15 | 0 | 0 | 0 | 18 |

The managed gate has zero unexplained reds. Both R3 events have separate exact-cell deltas:
`D1123_64K_LEAF_TRADE` names #1123 / PR #1129 and accepts only provider calls 65 versus 46 at 64K;
`D1123_40K_COMPRESS_TRADE` accepts only compress runs 113 versus 93 at 40K (provider calls fall 166 to 165).
The latter is an interpretation correction: the supplied triage numbers already exceed R3's 10% + 2 run
threshold despite its prose saying 40K passes. Neither exception absorbs another metric, case, or count pair.

| i1139 window | Bare GA / rc7 uncovered | Managed GA / rc7 uncovered | Managed GA / rc7 provider calls |
|---|---:|---:|---:|
| 32K | 0 / 734 | 734 / 734 | 3 / 3 |
| 40K | 8 / 29 | 206 / 29 | 166 / 165 |
| 48K | 0 / 0 | 0 / 0 | 246 / 198 |
| 64K | 0 / 0 | 26 / 0 | 46 / 65 |
| 128K | 130 / 25 | 125 / 25 | 12 / 7 |

Bare remains a diagnostic arm: its three R3 reds are provider calls at 40K (165/53), 48K (198/85),
and g51 (4/1), candidate/GA. C1 is red at 32K, 40K, and g9, g17, g23, g27, g28, g29, g30, g37,
g38, g54, g55, g57, g58. These events stay visible and unabsorbed; no broad default-change delta hides them.
They are comparison signals against bare settings, not evidence of regression against managed GA.
At 32K, C1 reproduces #1247 exactly: 734 versus bare GA's 0, and 734 versus managed GA's 734.
A1 has no missing produced sentinels; C1 does not claim stored context was deleted.

Final enabled acceptance: 24 passed, 1 strict xfailed (#1247), 630.78 seconds pytest / 632.21 seconds process wall.
The 1,000-case managed projection sample has 54 R1 reds, all accepted by D916; zero unexplained.
Opt-out: one module skip (expected exit 5), 1.85 seconds wall. Full collection: 7,698 tests, unchanged,
19.25 seconds pytest / 21.19 seconds wall. Enabled collection grows from 18 to 25 tests.
Bare GA public replay: 184.85 seconds process wall; candidate records reused. No wall-time assertion decides an oracle.
The initial development run (362.18 seconds) found the 40K run-count discrepancy and a missing-ID unit-fixture
error in a new delta predicate; both are corrected in the final run. Original S1 receipts below remain historical.

## Reproduce

With an interpreter providing pytest and the CI `agent.context_engine` host stub on `PYTHONPATH`:

```sh
HERMES_DISABLE_LAZY_INSTALLS=1 LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --baseline v0.26.2 --candidate 432bc4567c7b6201f83421bb9e5402eee7980302 --layers projection --cases 4000 --seed 1227 --out differential-results/projection
HERMES_DISABLE_LAZY_INSTALLS=1 LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --baseline v0.26.2 --candidate 432bc4567c7b6201f83421bb9e5402eee7980302 --baseline-config managed --layers public --cases 60 --seed 1227 --out differential-results/public-managed
HERMES_DISABLE_LAZY_INSTALLS=1 LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --baseline v0.26.2 --candidate 432bc4567c7b6201f83421bb9e5402eee7980302 --baseline-config bare --layers public --cases 60 --seed 1227 --out differential-results/public-bare
LCM_DIFFERENTIAL=1 nice -n 19 python -m pytest tests/differential -q -p no:randomly -p no:cacheprovider
```

For a historical row, substitute its full SHA and use `--layers public --cases 0 --only CASE_ID` (comma-separated IDs are accepted). Resolve merge parents with `git rev-parse SHA^1`. All required objects were already present; no fetch was needed.
`LCM_DIFFERENTIAL_BASELINE_CONFIG`, `LCM_DIFFERENTIAL_BASELINE`, `LCM_DIFFERENTIAL_CANDIDATE`, `LCM_DIFFERENTIAL_CASES`, `LCM_DIFFERENTIAL_SEED`, `LCM_DIFFERENTIAL_PROJECTION_CASES`, and `LCM_DIFFERENTIAL_OUT` control pytest. Defaults are managed, GA, worktree, 60, 1227, and 1000. The last variable selects the evidence directory.
Each run retains replayable `*-cases.jsonl`, per-tree `*.jsonl` and logs, `summary.json`, and `summary.md`. Replay a case file with `nice -n 19 python tests/differential/driver.py --tree TREE --layer public --cases CASES --out RESULTS` under the same opt-in environment.

## Historical matrix

All rows use the public layer, seed 1227, and only the named exact shapes. Counts are failing oracle events; omitted oracles have zero failures. Unexplained counts are events, not distinct cases. Wall time is GA plus candidate; means are GA/candidate seconds per case.

| Probe | Candidate SHA | Cases / exact shape | Oracle reds | Unexplained | Wall s | Mean s/case |
|---|---|---|---|---:|---:|---:|
| 917-parent | `bbcbf2d706706ddbb0f7a6aba63fdcebaf81447b` | 1: 1220 | 0 | 0 | 0.99 | 0.097/0.101 |
| 917-merge | `286e731800b95eb9bbf11d240c4ca6e2d7f3e8d4` | 1: 1220 | R1=1 | 1 | 1.06 | 0.109/0.169 |
| 936-parent | `7defa0b0fac8a4906d5e054302f6ef6d630544f4` | 1: 1221 | 0 | 0 | 1.02 | 0.071/0.055 |
| 936-merge | `112388f7f07665ec21a74a7b114c8ed2d07d6577` | 1: 1221 | 0 | 0 | 0.77 | 0.050/0.071 |
| 1129-parent | `86fc834ef0e995bc8e187b002db43268d033346a` | 1: 1139:32k | R2=2 | 2 | 16.41 | 11.988/3.457 |
| 1129-merge | `2c6dfef9c6425aca765113e769c0df7dd389d35e` | 1: 1139:32k | R2=2 | 2 | 15.43 | 2.955/11.676 |
| 1140-merge | `09e32580fdf95885474dbeb5ec4794e30136bb6a` | 1: 1139:32k | R2=2 | 2 | 6.33 | 2.322/2.914 |
| rc6 | `bd46fd9fed63bc95f81c464065400deff143f667` | 5: 1220, 1221, 1227:n90, 1227:n113, 1227:n128 | R1=2 | 2 | 2.13 | 0.130/0.161 |
| rc7 | `432bc4567c7b6201f83421bb9e5402eee7980302` | 5: 1220, 1221, 1227:n90, 1227:n113, 1227:n128 | 0 | 0 | 4.32 | 0.205/0.530 |

- #917: the parallel shape changes from green at the parent to R1 at the merge: GA sends 27,719 tokens, candidate 39,771, window 32,768. The short-text preservation branch is `survival_fit.py:349` at the merge.
- rc6: sequential outage n113 sends 33,641 tokens versus GA 27,716; n128 sends 38,111 versus GA 27,683. The tool-only second pass is `survival_fit.py:432`. Both shapes are green on rc7. n90, #1220, and #1221 are green on both rc6 and rc7.
- #936: both parent and merge are green. This public driver did not reproduce the assembly-only #1221 failure: its 24-row fresh tail still includes the objective, whereas the historical helper directly assembled the 20-row tool tail. This is a driver coverage gap, not evidence that #1221 was invalid.
- #1129 and #1140: the 32k shape has two R2 events at each parent/merge/fix; R3 stays green. GA has 119 compress runs / 52 provider calls; each compared tree has 49 / 3. This synthetic provider/cadence does not reproduce the historical progress regression, so it cannot establish that #1140 clears it. No oracle was tuned.

## Projection triage

4,000 instrumented p34 cases: 250 R1 events, all 250 absorbed by the sole `D916_GA_MARKED_USER` delta; zero unexplained, zero candidate user marks, zero input mutations. GA/candidate wall times were 51.52/74.08 seconds; mean case times 0.01085/0.01821 seconds. No missing seams.
The original uninstrumented p34 output artifacts independently contain 253 fit failures, all with GA mark-only user rows. Their SHA-256 receipts and all case IDs are in `historical-253.json` in the lane evidence packet. The new corpus replaces text prefixes with row sentinels, grows very short rows, and uses a fresh store per case (changing store-id widths). It is intentionally reported as 250, not presented as a byte-identical 253-case reproduction.

## Historical public triage and unexplained results

The rc7 run contains 12 exact shapes plus 60 generated cases, seed 1227. It reports 34 unexplained events in five cases: R2=31 and R3=3; R1=A1=A2=0. No delta absorbed a public case; D916 is zero-absorption on this layer but active in the full projection run.
- `i1139_small_window_32k`: objective missing versus GA at calls 10–11 (zero based).
- `i1139_small_window_40k`: objective missing at calls 13–36; provider calls 165 versus GA 53 (R3).
- `i1139_small_window_48k`: objective missing at calls 16–17; provider calls 198 versus GA 85 (R3).
- `i1139_small_window_64k`: objective missing at calls 21–23.
- `g51` (`mixed/1004/user`): provider calls 4 versus GA 1 (R3).
These are observed synthetic comparison failures, not newly confirmed product defects. In particular, R2 tracks the original tagged objective even after later scripted user turns. Every failure retains its full case JSON; no new delta suppresses it.
Absolute over-window outputs also occurred in g4, g10, g13, g15, g41, g42, and g44; GA did not fit either, so these are not R1 regressions. #1225 is separate: this 22k-objective/8k-tool-tail input is projected to 22,579 tokens and fits. Its strict absolute xfail XPASSes; the supplied shape does not reproduce the issue's absolute failure.

The rc6 enabled suite also reports 34 unexplained projection R1 cases among 1,000 samples (98 raw fit failures minus 64 D916 cases). These stay red; the rc7 projection sample clears them. The exact replay records live in `pytest-rc6/projection/summary.json`.

| rc6 projection case | GA tokens | Candidate tokens | Limit |
|---|---:|---:|---:|
| p10 | 1414 | 1977 | 1474 |
| p92 | 643 | 944 | 835 |
| p95 | 43 | 805 | 754 |
| p111 | 1488 | 1575 | 1523 |
| p221 | 2203 | 2785 | 2374 |
| p234 | 2160 | 2422 | 2198 |
| p241 | 1160 | 1461 | 1368 |
| p250 | 3555 | 4093 | 3775 |
| p252 | 3479 | 4648 | 3744 |
| p291 | 3582 | 3862 | 3618 |
| p300 | 1167 | 1468 | 1388 |
| p340 | 85 | 736 | 643 |
| p367 | 43 | 280 | 271 |
| p400 | 1362 | 1795 | 1587 |
| p433 | 2659 | 3149 | 3139 |
| p452 | 1003 | 1319 | 1185 |
| p492 | 1292 | 2160 | 2005 |
| p503 | 655 | 1515 | 711 |
| p574 | 2230 | 2359 | 2233 |
| p601 | 330 | 588 | 372 |
| p617 | 369 | 670 | 464 |
| p644 | 1173 | 2865 | 2677 |
| p649 | 915 | 1217 | 933 |
| p670 | 3645 | 4099 | 3877 |
| p691 | 406 | 1101 | 910 |
| p704 | 1507 | 2547 | 2386 |
| p715 | 3085 | 3902 | 3464 |
| p759 | 1510 | 2272 | 1888 |
| p813 | 1132 | 1316 | 1312 |
| p874 | 1471 | 2276 | 1615 |
| p894 | 1065 | 1305 | 1239 |
| p927 | 57 | 303 | 212 |
| p964 | 544 | 806 | 786 |
| p983 | 1257 | 2427 | 1311 |

## Method and interpretations

- Default config construction disables only temporal rollups; fleet overrides are the seven settings in the prior public driver. #1107 declares a 4,000-token first-sight threshold because its stated 5,000-token result is below the product default 10,000 threshold; host overhead is 1,000 tokens.
- Both trees use the deterministic character counter, no plugin registration, an offline provider, and a fresh engine/store per case. TCP connections are refused. Healthy replies are 80 tokens, or approximately 2,000 for #1139. The logical host advances 15 seconds per model call; a provider attempt advances one second. Product cooldown, circuit, and spend-guard logic still run.
- All loaded tree modules with a time-module binding share the scripted clock. An early matrix patched only engine/escalation and was invalid after compaction moved to its own module; `backtest-v2` supersedes it. No timing assertion decides product correctness.
- Objective cases seed a real DAG leaf through the guarded seams used by the existing regression test. They then use public ingest/should_compress/compress/retry/update calls; the driver does not force private assembly results into the sent view.
- A1 reads every stored message, follows externalized references found in stored content with that tree's `extract_externalized_refs` and `load_externalized_payload`, and includes retrieved payload content in the sentinel census. Unretrievable references are recorded. Sentinel coverage is not a byte-for-byte durability proof.
- Every public call records token/row counts, compression runs, provider calls, overflow status, sentinel census, and tool-pair errors. The newest alias is derived from the latest produced user row's unique sentinel. Projection fixtures retain p34's synthetic externalized-placeholder and unstored-row cases.
- Generated samples weight windows at or below 128k at 70% exactly for a 60-case run. Larger windows use larger rows to bound local row count; the sampler is not an exhaustive Cartesian matrix. A stopped initial 1M-window/8,311-row pilot is retained as incomplete evidence. The final generator pads any density-floor shortfall deterministically.
- Private seam absence is recorded and unsupported essential paths become DRIVER failures. No tested tree lacked a required seam. Missing commits/host stubs skip locally and fail under CI.
- The provided interpreter has no xdist: runs are serial under nice 19, cacheprovider disabled. This lane explicitly requires the local offline historical matrix; it does not run the full repository suite. A brief overlap during the pilot was suspended, then resumed serially.

## Historical acceptance evidence

Detailed command receipts and logs are in the S1 lane evidence packet. The default collection baseline is 7,698 tests. The disabled differential module reports one skip in 0.04 seconds (0.99 seconds process wall time); pytest exits 5 because the opt-out module collects no runnable tests.
Full collection after the additions is still 7,698 tests (8.94 seconds), with the new module skipped. `ruff check tests/differential` passes.

Enabled worktree pytest: 12 passed, 6 failed in 146.22 seconds. Enabled rc6 pytest: 9 passed, 9 failed in 164.54 seconds. Four small-window tests and the generated sample fail on the documented public reds; #1225 fails as strict XPASS. Rc6 additionally fails the two #1227 exact shapes and its projection test. Three cheap oracle checks pass. These are report-only failures; the opt-out suite remains unchanged.

After the density-floor correction, the focused 60-case generated test still has only g51's R3 failure; all inputs measure 0.80022–1.46751 times their window. Its final GA/candidate wall times are 73.30/37.77 seconds and mean case times 1.208/0.488 seconds. The exact-shape receipts are unchanged; `generated-final` supersedes only the generated-sample receipt.

No production source, existing test, workflow, or root configuration changed. There was no push, PR, merge, release, or runtime action.

## Historical remaining evidence gate (superseded by S1b)

The orchestrator should retain these unexplained cases and choose a bounded follow-up using the original historical host inputs for #1221, #1139, and #1225. The acceptance condition is the original failure on its culprit SHA and clearance on the fix while keeping the oracles unchanged. Treating a green blind spot or strict XPASS as proof of a fix would be misleading.
