# LCM-X roadmap

Status: reconciled 2026-10-13. v0.26.2 is the current stable release. v0.27.0-rc4 is the GA candidate; its last gate is the reference-agent soak. The [vision](VISION.md) says what we are building and why. Issue [#658](https://github.com/electricsheephq/lcm-x/issues/658) is the tracker; GitHub milestones hold the issues for each release. The current stable release is recorded in [docs/project-status.md](docs/project-status.md).

## How releases work

- **One themed minor about every two weeks, by gate.** Each minor has one theme and a written "GA accepts when" line. Work runs in parallel lanes; a feature ships in the first release after it passes its own gate, so dates are forecasts.
- **Proven quality ships on.** A feature becomes the default when a pre-declared, paired measurement on held-out questions, read by a frontier model, shows a significant gain in evidence or answer quality, and it holds the lossless and wait bars. Token cost is not a gate ([vision](VISION.md)). Scope freezes at the first release candidate. After the third candidate, a minor ships only with known, non-blocking items; otherwise the failing change is taken out.
- **Patches only for release-blocking defects reachable in real deployments:** data loss, duplicate rows, a wedged or reset session, a request over the context window, or a security defect such as a secret exposure or an access-control bypass. At most one patch a week unless an incident is live.
- **Behaviour and default changes ship only in minors.** v0.24.9 was scoped before this policy was written; it is the last release under the old rule.
- **Every release candidate runs the gauntlet** in [bench/specs/RELEASE-READINESS-V1.md](bench/specs/RELEASE-READINESS-V1.md). A candidate that fails is fixed and respun; only a candidate that passes can become GA, and the GA tree differs from it only by its release notes.
- **Every release also carries a recall scorecard** ([#898](https://github.com/electricsheephq/lcm-x/issues/898)), first applied at v0.26.0:
  - **Every release candidate:** the default configuration (embeddings off) is scored on LongMemEval-M retrieval against the previous GA. It passes when no question's `lcm_recall` result changes. Otherwise each change must be traced to a named commit and judged intended. Net R@10 may not fall more than 0.005, and no category may fall more than 0.02.
  - **Once per minor:** two embeddings-on configurations are recorded, a cached Voyage tracker and local fastembed. Each starts gating once a repeat run in the same session agrees with itself on every question.
  - **Once per minor, at the first release candidate:** facts kept after compaction (Track S) are compared with the previous GA's row by a paired per-fact test on identical material. Answer accuracy (LongMemEval-S, LoCoMo) uses its own paired rule.
  - Comparisons with other compaction systems are reported as H1 evidence, not per-release gates.

## H1 — compaction you don't notice

Every compaction LCM-X starts runs on the turn thread, so the user waits for it. v0.24.9 bounded that wait. v0.25.0 met its bar: on the gauntlet's live soak, compaction ran at p90 55.5 s with the longest at 72.1 s. Compaction breaks the prompt cache; additional breaks from stub swaps are tracked in #788 and have not been isolated. H1 removes the wait in steps.

### Compaction architecture: four layers

The leading systems no longer rely on one lossy summary. They stack four layers, and this is LCM-X's compaction architecture (decided 2026-10-10, tracked in [#658](https://github.com/electricsheephq/lcm-x/issues/658)):

| Layer | What it does | LCM-X | Work |
|---|---|---|---|
| 1. Clear old tool output first | Replace large or old tool results with durable references before summarising | Large-output externalization and active-replay stubbing (first sight above 10k tokens; aged tier above 2k tokens outside the fresh tail) | On by default in v0.27.0 (#1013; the default flip is #1070) |
| 2. Summary | Summarise what is compacted | The summary DAG; the summariser sees the middle of long messages (#611) | v0.27.0; the prompt-v2 flip rule in #660 |
| 3. Deterministic re-insertion | Put back what must not be lost: the user's own words, instructions, plan | The objective survives assembly overflow (#1048); the LCM note reaches the model (#900); the user carry packet (#659); an identifier index (#1066). The host rebuilds the system prompt and re-inserts the todo list. | #659: Eval-2 was inconclusive (#1064), so the decision moves to v0.28.0 at a larger sample; #1066 |
| 4. Lossless, searchable history | Let the agent recover anything exactly | `lcm_recall`, `lcm_grep`, `lcm_expand` and `lcm_describe` over the lossless store; stemmed full-text recall (#1035); local semantic recall for managed deployments (#1014, #1063) | v0.27.0 |

**Why layer 3 comes first.** On Track S (checkpoint 304, LCM-X v0.24.8 against Codex native compaction), Codex kept 132/132 facts stated by the user against LCM-X's 75/132, and 27/72 facts from the middle of long tool outputs against 0/72. LCM-X was ahead on assistant facts (+20) and on the head and tail of tool outputs (+21). The Codex arm was read by a different reader model in that run.
- The tool-middle gap starts at the summariser input clip, which #611 addresses: 42/72 middle facts now reach leaf summaries, up from 0/72. That is a leaf-level result; whether those facts survive rollups and assembly for the same reader is measured end to end by Eval-2.
- The user-fact gap is the missing verbatim user carry; #611 does not change it. In an offline screen with the same reader, the carry packet raised user-row facts by 50.8 points with no losses (#659). Eval-2 decides ([#1064](https://github.com/electricsheephq/lcm-x/issues/1064)).

**Never below Hermes built-in.**
- Hermes's built-in compressor keeps user messages verbatim, an identifier index, a recovery footer and stubs for old tool results.
- When LCM-X is the context engine it replaces that compressor. The floor is measured, not assumed: LCM-X must score at least as well as Hermes built-in on every fact class, read by the same reader in Eval-2's Hermes built-in arm (#1064).
- v0.27.0 records that comparison; v0.28.0 gates on it. Layer 1 (#1070) and the objective and LCM-note parts of layer 3 (#1048, #900) ship in v0.27.0. The user carry packet (#659) ships when Eval-2 keeps it. The identifier index (#1066) ships only if Eval-2 measures a gain.

**Deferred, with revisit triggers.**
- **Summary-free compaction**, where the model writes notes and a fresh window starts. Revisit if a provider enables it for a generally available model; LCM-X's layer 4 already provides the history tool that design depends on.
- **Delegating the active-window summary to provider-native compaction.** Revisit after coexistence with Hermes native compaction is defined (#1068) and an Eval-2 arm shows a gain.
- **Turn-relevant evidence injection.** Revisit when the pre-answer arm passes its pre-declared bar.

| Release | Theme | GA accepts when |
|---|---|---|
| **v0.24.9** | Drain: bounded foreground compaction, capped hidden leaves, exit fit | Shipped 2026-10-01 |
| **v0.25.0** | Long sessions: drain speed and correctness (#597, #750, removal of native recovery) | Shipped 2026-10-04 |
| **v0.26.0** | Host message identity: record the host's stable message id beside LCM-X's own row matching (shadow mode; `on` mode is deferred, #643) | Shipped 2026-10-07 |
| **v0.27.0** | Summary quality. rc4 (2026-10-13) is the GA candidate. Scope: <ul><li>the summariser sees the middle of long messages (#611: each leaf's messages share the leaf token budget; on synthetic material, middle-placed facts reaching leaf summaries rise from 0/72 to 42/72; this also closes #899);</li><li>the LCM note and the retained-user anchor reach the model on Hermes (#900);</li><li>fixes for long single-turn tool loops (#921, #922, #923), compaction that grows the context on heartbeat sessions (#904, #930, #931), survival fits (#798, #916) and pre-leaf condensation (#909);</li><li>instruction-continuity probes (#659), recorded in this release;</li><li>product defaults set to the measured best configuration (#1013): compaction at 75%, fresh tail 24 messages / 24k tokens, leaf chunks of 24k tokens with a 2,000-token summary cap (#1123, #1128; on the per-message-wait replay, 20 compactions per 100 messages instead of 87), full sweep, rollups, and large-output externalization and replay stubbing on (the last two flip in #1070);</li><li>the objective survives assembly overflow (#1048); the inline floor scales with the context window (#1055);</li><li>recall: a stemmed full-text index (#1035), working `lcm_grep` OR/NOT/prefix operators (#1015), incremental embedding (#1014) and local-profile auto-registration (#1063), so managed deployments can turn on local semantic recall;</li><li>dormant tools are not advertised while their features are off (#1017);</li><li>not shipped: the user carry packet (#659). Eval-2 was inconclusive; the decision moves to v0.28.0.</li></ul>The summary prompt default stays v1 (#660). v2 has not passed the flip rule: at least as good as v1 on every paired axis, better on one, within the latency bars, and with a level-3 rate within band of zero; changing the default remains an owner decision. | <ul><li>No system-level axis below the v0.26.0 row: paired per-fact exact McNemar on identical v3 material, α 0.05, minimum effect 5 points; add runs when run1/run2 spread exceeds 0.10. QA rows use their own paired rule.</li><li>The theme gain is present: middle-placed facts reach summaries, and constraint continuity is at or above the floor declared from the v0.26.0 baseline.</li><li>Compaction p90 ≤ 60 s and longest ≤ 120 s on the ≥ 24 h soak; gauntlet cells ≤ 145 s.</li><li>Recall (#898): zero embeddings-off per-question flips versus v0.26.0, or each flip attributed and classified intended within the net −0.005 / category −0.02 floors.</li><li>Recall with local embeddings on, the managed default (#898): on the GA candidate, net R@10 at or above the embeddings-off row on the same questions (paired), with the local arm's own A/A′ placebo at 0 discordant; every per-question loss against embeddings-off is listed and classified.</li><li>Eval-2 recorded per fact class against the Hermes built-in arm with the same reader (#1064).</li><li>No release-blocking defects.</li></ul> |
| **v0.27.1** | Repeated compaction and slow summarisers. Fixes, background leaf preparation slice 1 shipped off, and the first proven defaults. Scope: <ul><li>local semantic recall on by default and built in: the fastembed dependency is declared so the host installs and keeps it, the model is prepared outside the user's turn, existing stores catch up in the background, and full-text recall answers, visibly, until the index is ready;</li><li>sender provenance on message rows and recall hits (#995; event time shipped in #1005);</li><li>background leaf preparation, slice 1, off by default (#787): the next leaf is summarised in the background after a turn and promoted at the threshold when the stored rows still match;</li><li>slow summarisers: extraction leaves the summary call its time (#1122), a hung route opens the circuit (#1137), a diagnostic for long pre-leaf phases (#1127), and #1096 and #1149;</li><li>correctness: duplicate re-ingest copies (#1124), leaves with no durable source (#1141), the stub-first commit proof (#1116), replay prefix hashing cost (#1090);</li><li>follow-ups to the 24k default (#1126, #1132, #1133, #1134, #1138).</li></ul> | <ul><li>Full suite with 0 failures on every supported host; gauntlet cells ≤ 145 s.</li><li>Recall fast tier: zero embeddings-off per-question flips versus v0.27.0; Track S not below the v0.27.0 row.</li><li>Background preparation off: no worker activity and no promotion on the replay probes.</li><li>Background preparation on: the deterministic replay probe promotes at least at the slice-1 rate, and the think-time cell shows 0 dropped, uncovered or duplicate rows; its visible wait, promoted share and facts kept are recorded against the off arm for the v0.28.0 decision.</li><li>Local semantic recall: a fresh install, a host update, an offline cold start and a large-store upgrade all reach a ready index with no foreground wait.</li><li>#995: a paired LoCoMo row with and without sender provenance shows no loss.</li><li>No release-blocking defects.</li></ul> |
| **v0.28.0** | Agentic memory. Scope: <ul><li>agentic recall as the default delivery: the model drives `lcm_recall`, `lcm_grep` and `lcm_expand` and stops when the evidence is sufficient;</li><li>session breadth: more distinct sessions per question, and expand-to-session for questions that aggregate across sessions;</li><li>reranking selectable independently of the embedding provider, with a built-in local reranker if it passes its screen;</li><li>the first frontier-protocol rows, and a LongMemEval-M answer row on all 500 questions.</li></ul> | <ul><li>Each default change passes the proven-quality rule.</li><li>Recall stays within its latency budget on the smallest supported host while compaction runs.</li><li>No system-level axis below v0.27.1; the recall scorecard as before.</li><li>No release-blocking defects.</li></ul> |
| **v0.29.0** | Active memory and a second harness. Scope: <ul><li>source-linked derived memory (facts, preferences, entities, events and their times), on by default if it passes the proven-quality rule;</li><li>a memory service with an MCP endpoint over the same store, one writer per store;</li><li>a Claude Code adapter.</li></ul> | <ul><li>Hermes and Claude Code share one memory through restarts and replays with stable identities and no duplicate rows.</li><li>Every derived memory resolves to its source rows; no stored row lost.</li><li>No release-blocking defects.</li></ul> |
| **v0.30.0** | Invisible compaction (moved from v0.28.0; ships earlier if its gate passes first): background preparation on by default, plus its second slice (#787; slice 1 ships off in v0.27.1), with #610, #626, #749, #775, #788 and #926. Also the default changes held from v0.27.1: pre-answer evidence on by default (#1111), and the carry packet decision (#659). Pre-answer evidence turns on only after it passes the proven-quality rule. | <ul><li>Default-on prerequisites closed (#1145), the think-time cell's on-arm bars met (visible-wait p90 ≤ 5 s, waited share ≤ 5%, facts no more than 0.10 below the off arm, 0 dropped, uncovered or duplicate rows), and promotion no lower than in the previous release.</li><li>Visible-wait p90 ≤ 5 s on the reference-agent soak.</li><li>No compaction slower than v0.24.9.</li><li>The continuity suite (#659) passes its per-class bars, declared from the v0.27.0 run (100% for host instructions).</li><li>No system-level axis below v0.27.0 under the same paired rule as v0.27.0.</li><li>Recall (#898): zero embeddings-off per-question flips versus v0.27.0, or each flip attributed and classified intended within the net −0.005 / category −0.02 floors.</li><li>At or above the Hermes built-in arm on every Eval-2 fact class (#1064).</li><li>No release-blocking defects.</li></ul> |
| **v0.31.0** | More harnesses: Codex, OpenClaw and Cursor adapters; service authentication; portable export | <ul><li>Each adapter passes the shared-memory identity and no-duplicate test.</li><li>No release-blocking defects.</li></ul> |

LCM-X is also compared at system level with Codex native compaction and lossless-claw ([#664](https://github.com/electricsheephq/lcm-x/issues/664)). That comparison is H1 exit evidence, not a release gate. The last run was on v0.24.8 ([#658](https://github.com/electricsheephq/lcm-x/issues/658)). The next is reported at the v0.27.0 release candidate.

Plan, todo and goal survival across a compaction is host behaviour. Fixes are proposed upstream in Hermes and tracked in [#901](https://github.com/electricsheephq/lcm-x/issues/901). LCM-X's own part is [#900](https://github.com/electricsheephq/lcm-x/issues/900).

## H2 — memory that wins (now)

Where LCM-X stands (2026-10-14):
- **LongMemEval-S.** 85.8% with local semantic recall (official judge, v0.26.1); 90.4% with the July Voyage configuration.
- **LoCoMo.** 69.2% on the 1,540-question set (common judge).
- **Not yet entered:** BEAM, LongMemEval-V2 and the neutral boards.
- **What others report.** Leading systems report 92–96% on LongMemEval-S. On boards that give every system the same reader, those numbers fall to roughly 56–89%, and at 10M tokens no system leads clearly.

The plan:
1. **Built in.** Local semantic recall on by default (v0.27.1); a local reranker if it passes its screen (v0.28.0).
2. **Evidence delivery.** Sender and time provenance (#995, v0.27.1); agentic recall and session breadth (v0.28.0). Measured offline and rejected against pre-declared bars: session-level scoring (#1039), a session-aware hit cap (#1040), time ranges read from the query (#1025), hit diversity (#1023), session score (#1022), entity and identifier postings, and returning neighbours with each hit.
3. **Derived memory**, linked to its sources (v0.29.0).
4. **Measurement.**
   - **Frontier protocol:**
     - frontier readers;
     - two frontier judges from different families on the same blinded answers;
     - published prompts, question ids and configurations;
     - a same-reader full-context baseline;
     - agentic and single-retrieval rows in separate tables.
   - **Neutral boards** that run every system the same way: the Agent Memory Leaderboard, an in-host action-graded board, OmniMemEval, LifeBench, LongMemEval-V2 and BEAM.
   - LongMemEval-S and LoCoMo stay as regression rows.
   - Results are published in the repository scoreboard with tokens per query and cost (#898, #1067).
5. **Outside designs** go into a research queue (#1150).

Milestone: **H2 — memory that measurably wins**.

## H3 — one memory, any harness

- **Architecture.** Context assembly and compaction stay inside the host process. A memory service serves recall and ingestion to other harnesses through an MCP endpoint. Each store has one writer, and events keep a stable identity across hosts.
- **First pair:** Hermes and Claude Code (v0.29.0).
- **Next:** Codex, OpenClaw and Cursor, and portable export (v0.31.0).
- **Also in H3:**
  - teams and multi-agent use;
  - the fact layer (blocked on #694, #987 and #1065);
  - query views (#694);
  - proactive recall (#962, #969, #1008, #865).

## H4 — memory for agents in the world (research)

Events with time, place and modality, for agents and robots that act over days. One measured use comes before any schema commitment.
