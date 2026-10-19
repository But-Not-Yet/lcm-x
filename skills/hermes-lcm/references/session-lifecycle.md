# Session lifecycle and rotate

Hermes `/new` starts a new host session. LCM-X binds that session to its own lifecycle row and may carry eligible higher-depth summaries into the new current-session context. Source eligibility and exact expansion still come from descendant raw messages; carried summaries do not rewrite source ownership.

Do not promise that `/new` deletes historical LCM data. Earlier rows remain in `lcm.db` unless an explicitly authorized cleanup removes them, and they remain available through bounded cross-session recall.

When the host's `pre_llm_call` history ends with two user rows, LCM-X ingests the
pending predecessor before the host repairs consecutive users. Normal ingest
still stores the current turn. This preserves unanswered turns across crash
recovery without reconstructing them from a later composite. Carried-stamp
replay still needs its exact store-time marker, in the same session or a verified
compression ancestor and the bound conversation; hosts without this history hook
keep the safe duplicate fallback.

## Compaction commits

Hermes commits a compaction by calling the session-end hook with the list it compressed, then the compression-start hook. LCM-X treats that call as a commit, not a session end. It does not re-store those rows. The session keeps its identity, published frontier and ingest position across the boundary, both when Hermes keeps the session id (in place) and when it rotates to a new one. A genuine session end (exit, `/new`, or an end after a cancelled compaction) still finalizes the session. It also pauses this engine's background embedding maintenance: queued passes are dropped and nothing new is scheduled until the next live bind re-arms it (a compaction commit leaves maintenance running; see `architecture.md` for the shutdown and process-exit waits). A session that resumes later continues from the frontier it finalized itself, never another session's.

`LCM_NATIVE_RECOVERY`: removed in v0.25.0; ignored if set. Native adoption proofs and replay-snapshot digests that older versions wrote are still read, so a host replay of an adopted native summary reconciles after a restart or rotation.

## `/lcm rotate`

`/lcm rotate` is a read-only preview of the tail-preserving frontier for the
active session. It keeps `session_id` and `conversation_id` and reports the
configured fresh tail, pre-tail message count, current and planned lifecycle
frontier, and rolling backup path. The preview refuses ignored or stateless sessions.

`/lcm rotate apply` is refused on Hermes (#1141): Hermes reloads its own
transcript after a restart, so advancing the frontier without summaries leaves
rows LCM-X cannot map. No backup is created and no lifecycle state is changed.
Use `/compress here N` to summarise everything except the most recent N messages.
If an earlier LCM-X already rotated this session and Hermes restarted since, compaction can stop on rows below that frontier and `/compress here N` will not repair it: start a new session with `/new` (raw rows stay recoverable via `lcm_load_session`; in-place fix tracked in #1184).
Raw source rows remain recoverable through `lcm_load_session` and `lcm_expand`.

Use a separate session when the user wants a new active conversational boundary.
