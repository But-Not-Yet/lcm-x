# FrontierState: one owner for the ingest frontier (design note, C4)

**Status:** design only. This note changes no code. Code slices start only when the gates in section 7 hold.
**Line numbers** are at v0.27.0-rc8 (`4e603ed6`).

## 1. Why

The engine keeps an in-memory **ingest frontier**: how many host messages it has already stored, whether that count can be
trusted, which stored row the last published summary covers, and what the last `compress()` proved. These live in five instance
attributes written by 62 statements in four files. Nothing states what each event does to all five together, and nothing checks
them against their durable counterparts.

Divergence between this in-memory frontier and durable state, after a restart, rebind, rotation or reset, is the largest
defect family in the engine:

- Open: #1116 and #1286 (an adopted output whose durable proof is stale), #1211 (rebind reconcile), #1004 (fit proof not
  persisted), #503 (frontier-bounded matching), #806 and #871 (rotation handoff and crash window), #1037 (no durable per-row
  identity), #7 (positional cursor), #1184, #1162 and #1097 (downstream symptoms of a stale proof).
- Closed: #493 and #499 (host rewrites the cursor never saw), #517 (proof wire format), #534 and #644 (fit and recovery rows
  without proof), #752, #760 and #771 (frontier restored after a reset).

The state-coupling audit counted 27 commits to `_ingest_cursor`, 20 of them fixes. The goal of C4 is a single owner whose
transitions are named, unit-tested and visible in review. Behaviour does not change.

## 2. The state

| Attribute | Meaning | Durable counterpart | Writes |
|---|---|---|---|
| `_ingest_cursor` | Count of host messages already stored (positional). | None directly. It is rebuilt from stored rows and the durable commit proof. | 24 (1 in `__init__`) |
| `_ingest_cursor_needs_reconcile` | The cursor is untrusted; the next ingest rebuilds it from the store. | None. | 15 (1 in `__init__`) |
| `_last_compacted_store_id` | Store id of the last row a published summary covers (the frontier). | Lifecycle row `current_frontier_store_id`. | 9 (1 in `__init__`) |
| `_compress_commit_proof` | What the last `compress()` consumed and returned (identities), plus one-shot flags. | The durable commit proof, which is persisted only for some statuses (compaction.py:798-799). | 10, plus the binding filter loop at engine.py:3906 |
| `_last_emission_descriptors` | Rows LCM itself emitted (carriers, projections) in the last output. | Inside the durable proof when that proof is persisted. | 5, plus the same loop |

**Counting.** There are 63 attribute writes in 62 statements: survival_fit.py:229 assigns two attributes. Earlier plan text said
"12 writers"; that counted the writer methods of `_ingest_cursor`, not its statements. A regex count gives 22 for `_ingest_cursor`
because it misses the tuple at survival_fit.py:229 and the annotated assignment in `__init__`.

**Also part of the frontier, outside the five attributes:**
- In-place mutations of the proof dict:
  - `end_consumed` (engine.py:4500), set when the host commits by ending the session.
  - `consulted` (engine.py:5903), set on the first ingest after a compaction.
  - The child re-key at a rotation (engine.py:3847-3867).
- `_pending_reset_frontier_store_id` (engine.py:4618), a shadow copy of the frontier kept across a reset.
- `rotate_active_session` (engine.py:9025-9056), which advances the durable frontier and deliberately leaves the in-memory one
  behind.
- `_persist_frontier_marker` (engine.py:2533), which has no production caller; only tests call it.

## 3. Transitions observed today

Each row names a transition and the exact fields its sites write. A byte-identical refactor must keep each site's field subset
exactly. No site may gain or lose a write.

| Transition | Sites | Fields written |
|---|---|---|
| **init** | engine.py:581, 590-591, 595-596 | all five |
| **adopt** (cursor = length of the list `compress()` returns) | compaction.py:622 (hidden-only leaves), 1086 (stub-first exit), 1361 (cleanup-only), 1389 (hold, fit only), 2322 and 2332 (no-leaf reassembly), survival_fit.py:231 (fit, persisted) | cursor |
| **adopt and trust** | compaction.py:966-967 (fail-open after a publication conflict), 2453-2454 (leaf or condensation pass), engine.py:8458-8459 (forced overflow recovery) | cursor; needs_reconcile = False |
| **invalidate** | survival_fit.py:229 (fit shortened, not persisted), engine.py:3817-3818 (rotation with native digests and no native proof) | cursor = 0; needs_reconcile = True |
| **invalidate and drop proof** | engine.py:3512-3514 (in-place boundary without a consumed proof), 3875-3877 (rotation without a transferable proof) | proof = None; cursor = 0; needs_reconcile = True |
| **restart cursor** | engine.py:4121-4122 (boundary from a bypassed source), 4188-4189 (ordinary start or same-session rebind) | cursor = 0; frontier = 0 |
| **reset progress** | reset_state.py:44-46 | frontier = 0; cursor = 0; needs_reconcile = False |
| **reset proofs** | reset_state.py:60-61 | proof = None; descriptors = None |
| **bind frontier** | engine.py:2442 (every bind), 3458 (rebind after an unadopted commit), 3827 (rotation child) | frontier from the lifecycle row |
| **publish frontier** | compaction.py:2141 (leaf pass), engine.py:3498 (in-place boundary: publish the max, then read it back) | frontier |
| **schedule reconcile** | engine.py:5264, 5268, 5278 (after every non-carry-over bind; 5278 is the probe-failure fallback) | needs_reconcile |
| **consult proof** (first ingest after a compaction) | engine.py:5903-5933 | One of four outcomes. *Confirm* (5917): proof = None. *Host kept the input* (5921-5923): cursor = len(input); proof = None. *Remapped* (5929-5931): cursor = remapped; proof = None. *Unprovable* (5913, 5933): needs_reconcile = True. |
| **reconcile from store** | engine.py:5990, 5994 | cursor; needs_reconcile = False |
| **rewind** | engine.py:6053 (the identity anchor finds that the cursor overstates the stored prefix) | cursor |
| **advance** | engine.py:6337 (all new rows filtered), 6428 (new rows stored) | cursor |
| **record proof** | compaction.py:720 (clear), 792 and 797 (set), 801 (clear on exception) | proof; descriptors |
| **bind fit emission** | compaction.py:856 (fit carrier with no proof; never persisted) | descriptors |
| **filter by binding** | engine.py:3906 (after every `on_session_start`) | proof and/or descriptors = None |
| **transfer to child** | engine.py:3847-3867 (re-key in place), 3860 (descriptors = None on a binding mismatch) | proof; descriptors |

## 4. The owner

```python
@dataclass
class FrontierState:
    cursor: int = 0
    needs_reconcile: bool = False
    compacted_store_id: int = 0
    commit_proof: dict[str, Any] | None = None
    emission_descriptors: dict[str, Any] | None = None
    pending_reset_store_id: int | None = None
```

There is one method per transition in section 3, each writing exactly that row's fields. For example, `adopt(n)`,
`adopt_trusted(n)`, `invalidate(drop_proof=...)`, `restart_cursor()`, `reset_progress()`, `reset_proofs()`,
`bind_frontier(store_id)`, `publish_frontier(store_id)`, `consult(outcome)`, `reconcile(n)`, `rewind(n)` and `advance(n)`.

Rules:

1. **Every write goes through a method.** The last slice adds a static test that rejects assignment to the five legacy names
   outside `frontier_state.py`.
2. **No method reads durable state or computes identities.** Durable reads, identity comparisons and the remap stay where they
   are (engine, compaction, reconcile). They pass the result in, for example `consult(outcome)` with an outcome enum. This keeps
   FrontierState pure, so each transition gets a plain unit test.
3. **No logging or exception changes.** Log lines stay at their current call sites with the same logger names.
4. **Copies are explicit.**
   - `clone_for_agent` and `__deepcopy__` build a fresh engine (engine.py:858-917), so each clone already gets a fresh frontier from
     `__init__`.
   - The one shallow copy of an engine is `copy.copy(engine)` for the full-text grep deadline read (tools.py:3704). Today it
     copies the ints by value and shares the two dicts. With a FrontierState object it would share the whole object. The use is
     read-only, and B7 step 5 replaces that copy with a RetrievalContext. C4 slice 1 therefore lands after B7 step 5, or copies
     the frontier explicitly at that site.
   - A test pins that two clones never share one FrontierState object.

## 5. Slices

Each slice is its own refactor-only PR with byte-identical behaviour. Each runs the full suite at base and head with an identical
collected set, runs the exactly-once machines at the CI and nightly profiles with identical pass/xfail sets, and includes an AST
comparison showing that only the frontier statements changed.

| Slice | Content | Notes |
|---|---|---|
| S0 | This note. | |
| S1 | Add `FrontierState`; the engine holds `self._frontier`. The five old names become properties that read and write through it. No call site changes. | Tests and fixtures that write the old names keep working (for example tests/legacy_native_state.py:192-343). Engines built without `__init__` in tests (4 sites use `LCMEngine.__new__`: test_per_model_threshold.py:121, test_path_containment.py:62 and :97, test_issue_646_prompt_version.py:276) never get `_frontier`; S1 makes the properties tolerate that or gives those fixtures the field. The #1240 ratchet's attribute-write counts move; update its baseline in the same PR. |
| S2 | compaction.py call sites: adopt, adopt and trust, publish frontier, record proof, bind fit emission. | |
| S3 | engine.py boundary sites: invalidate, restart cursor, bind frontier, filter by binding, transfer to child. | Waits for the v0.27.1 boundary fixes (gate 1). |
| S4 | engine.py ingest sites: schedule reconcile, consult proof (typed outcome), reconcile from store, rewind, advance. | |
| S5 | reset_state.py and survival_fit.py. | B2 PR3a first moves survival_fit.py:229-231 into a caller-owned apply step; S5 edits that step. |
| S6 | The static guard. Remove the setters; getters stay while outside readers remain (tools.py:8552, identity_anchor.py, reconcile.py, store_complete.py). Fold the shadow `pending_reset_store_id` into the state, and delete the dead `_persist_frontier_marker`. | |

## 6. What the owner enables (fix-class, not part of C4)

These are behaviour changes. Each needs its own issue, a red test and the fix-scope rules. The owner makes each one a one-place
change.

- **Durable-or-invalidate on adopt.** The dominant defect: an output is adopted (cursor = its length) while the durable proof still
  describes an earlier compaction (#1116, #1286, #1004, #644; #806 at a rotation). The rule: "an adopted output that differs from
  its input either persists its proof or marks the cursor for reconcile". It belongs with #1117's fix, where #1286 also folds in.
- **Writes from a bypassed context.** `_record_compress_commit_proof` clears the proof (compaction.py:720) before its bypass
  check (:724). An in-process auxiliary thread that shares the engine can therefore clear the foreground session's in-memory
  proof. This comes from code reading and is being reproduced before anything is filed. Once the transitions take the binding,
  they can refuse writes from a bypassed context.
- **Probe-failure fallback.** When the reconcile probe fails, engine.py:5278 sets needs_reconcile = False with the cursor at 0, which
  trusts a zero cursor. This is worth its own look.
- **Debug check.** An opt-in assertion compares the frontier with the lifecycle row after each `compress()` and session end, for
  tests and the reliability cells only.

## 7. Gates before code (S1 and later)

1. **Product PRs first.** The open v0.27.1 PRs that edit writer functions have merged or closed. The ones within 3 lines of a write
   are #1147 and #1178. Others edit the same functions: #1117, #1142, #1143, #1146, #1148, #1174, #1182, #1195, #1205, #1212, #1218
   and #1277. Re-measure the hunk map before each slice.
2. **This note is accepted.**
3. **Characterization net, per slice.** Each write statement a slice touches must be executed by at least one test that fails when
   that write's value changes. A slice proves this with a mutation pass over its own statements.
   - **Today's data.** The exactly-once machines (compress, recovery fit) execute 30 of the 62 statements, at both the CI and the
     nightly profile; more examples reach none of the rest. The full test suite's coverage of the same statements is recorded with
     the C4 tracking issue.
   - **What reaches the gaps.** Grouped by the host operation that would reach them:
     - rotation (engine.py:3817-3877, 4121-4122): the rotation rule now in review;
     - an unadopted or refused commit, and the host's consecutive-user merge (engine.py:3458, 5913-5933);
     - reset (reset_state.py:44-61);
     - forced overflow (engine.py:8458-8459);
     - stub-first exit at the managed fleet profile policy (compaction.py:1086);
     - cleanup-only and no-leaf reassembly (compaction.py:1361, 2332);
     - publication and proof faults (compaction.py:801, 966-967; engine.py:5278).
   - **Ordering.** These are the next exactly-once operations, or targeted seeds where an operation is not host-shaped.

## 8. Not in scope

- **SessionBinding** (binding fields, holds, reset lists): after C4, one attribute cluster per PR.
- **Replacing the positional cursor with durable per-row identity** (#7, #493, #1037): a product design question for a later
  release, not a refactor.
