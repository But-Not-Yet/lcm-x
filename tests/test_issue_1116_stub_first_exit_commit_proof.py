"""A stub-first return the host adopts leaves the durable commit proof stale; a cold rebind then wedges.

After a real compaction writes ``compaction_commit_proof``, a later over-threshold pass can return through
``_stub_first_exit`` (#671, ``LCM compaction stop: reason=stub_first_exit leaves=0``). The host adopts that list
as its history, but no new proof is written. On the next cold bind (gateway restart, resume, engine recreate)
``_cursor_from_durable_commit_proof`` cannot prove the adopted list, reconcile takes "persisted ambiguous
delta" (cursor 0), the identity anchor stores the head as a new raw row at the end of the store, and the
in-order store-id mapper pins view[0] on that last row, so (nearly) no other row maps and every later pass
stops "leaf would end before an unresolved retained occurrence".

Two heads reach it:

* a pure-summary head that opens with the preserved objective (the fresh tail holds no user row, so the commit
  emits ``[Current user objective preserved ...]`` + summaries) -- the stub-first re-assembly emits the summaries
  without that block, so the adopted head is no longer the committed emission;
* a carrier head (summary + the glued real user row of a tail that starts on a user row) -- the stub-first
  exit keeps the host carrier AND assembles the summary prefix again, doubling it.

The control runs the same flow with no stub-first pass between the compaction and the rebind. The last test
starts from a store that already holds such a stale proof (the stub-first pass of a version that wrote none):
the upgrade restart is itself the cold bind, so it must reconcile past that proof.
"""
from __future__ import annotations

import json
import logging
import random
import re
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "stub-first-proof", "stub-first-proof-conv"
CONTEXT = 60_000  # threshold 0.75 x 60k = 45k tokens
OBJECTIVE = "[Current user objective preserved from compacted history]"
SUMMARY_HEADER = re.compile(r"Summary \(d\d, node \d+\)")
WEDGE = "unresolved retained occurrence"
STUB_FIRST = "reason=stub_first_exit"
# 4-row turns; the fresh tail (fresh_tail_count rows) decides the head kind: 7 rows leave the tail on an
# assistant row (a pure-summary head, the objective preserved because the tail holds no user row), 8 rows start
# it on a user row (a carrier head).
TAIL_COUNT = {"summary": 7, "carrier": 8}


class Host:
    """What Hermes does around the engine: stamps every row, adopts the list compress() returns, rebinds cold."""

    def __init__(self, tmp_path, tail_count):
        self.tmp_path, self.tail_count = tmp_path, tail_count
        self.rng, self.clock, self.n_turns = random.Random(1), 1_760_000_000.0, 0
        self.live: list = []
        self.engine = self._bind()

    def _config(self):
        return LCMConfig(
            database_path=str(self.tmp_path / "lcm.db"),
            large_output_externalization_path=str(self.tmp_path / "externalized"),
            fresh_tail_count=self.tail_count, fresh_tail_max_tokens=0, leaf_chunk_tokens=8_000,
            context_threshold=0.75,
        )

    def _bind(self, **start):
        engine = LCMEngine(config=self._config(), hermes_home=str(self.tmp_path / "home"))
        engine.on_session_start(SID, platform="cli", conversation_id=CID, context_length=CONTEXT, **start)
        return engine

    def words(self, n):
        return " ".join(f"t{self.rng.randrange(4000)}" for _ in range(n))

    def turn(self, steps=1, tool_words=450):
        """user, (assistant tool call, tool result) x steps, assistant reply -- each row stamped like a host row."""
        i, self.n_turns = self.n_turns, self.n_turns + 1
        rows = [{"role": "user", "content": f"turn {i}: " + self.words(25)}]
        for j in range(steps):
            call = f"toolu_{i:04d}_{j}"
            rows.append({"role": "assistant", "content": f"step {i}.{j} " + self.words(10), "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"n": j})}}]})
            rows.append({"role": "tool", "tool_call_id": call, "content": self.words(tool_words)})
        rows.append({"role": "assistant", "content": f"done {i}: " + self.words(40)})
        for row in rows:
            self.clock += 0.5
            row["timestamp"] = self.clock
        self.live += rows
        self.engine.ingest(self.live)

    def tokens(self):
        return count_messages_tokens(self.live)

    def compress(self, current_tokens=None, **kwargs):
        result = self.engine.compress(
            self.live, current_tokens=self.tokens() if current_tokens is None else current_tokens, **kwargs)
        return result, self.engine._last_compression_status

    def adopt(self, result):
        """The host swaps its history for the returned list and tells the engine about the boundary."""
        self.live = result
        self.engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, platform="cli",
                                     conversation_id=CID, context_length=CONTEXT)

    def cold_rebind(self):
        self.engine.shutdown()
        self.engine = self._bind()
        self.engine.ingest(deepcopy(self.live))

    def stored_rows(self):
        return self.engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (SID,)).fetchone()[0]

    def frontier(self):
        return int(self.engine._lifecycle.get_by_conversation(CID).current_frontier_store_id)

    def unmapped(self):
        """Indexes of the host rows (the head excepted) with no store row in the in-order store-id map."""
        probe = deepcopy(self.live)
        mapped = self.engine._get_store_id_map_for_messages(probe)
        return [i for i, row in enumerate(probe) if i > 0 and not mapped.get(id(row))]

    def drive(self, caplog, passes=3):
        """Further over-threshold passes (the host stays over the threshold), each result adopted.
        Returns the frontier before, and whether a pass stopped on an unresolved retained occurrence."""
        frontier, wedged = self.frontier(), False
        for _ in range(passes):
            caplog.clear()
            result, _status = self.compress(current_tokens=self.engine.threshold_tokens + 5_000)
            wedged = wedged or WEDGE in caplog.text
            if result != self.live:
                self.adopt(result)
        return frontier, wedged


@pytest.fixture
def host_for(tmp_path, monkeypatch):
    hosts = []
    calls = [0]

    def summarize(**kwargs):
        calls[0] += 1
        return f"Summary {calls[0]}: " + "lorem ipsum " * 200 + "\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)

    def build(head):
        host = Host(tmp_path, TAIL_COUNT[head])
        hosts.append(host)
        return host

    yield build
    for host in hosts:
        host.engine.shutdown()


def committed_session(host, head):
    """Ingest, then one real compaction that commits a durable proof; the host adopts its output."""
    while host.tokens() < host.engine.threshold_tokens + 8_000:
        host.turn()
    if head == "summary":
        host.turn(steps=8, tool_words=1_900)  # a tool loop: the fresh tail has no user row
    else:
        for _ in range(2):
            host.turn(tool_words=1_900)
    result, status = host.compress()
    assert status == "compacted"
    host.adopt(result)
    first = host.live[0]
    if head == "carrier":
        assert host.engine._generated_context_carrier_remainder(first) is not None
    else:
        assert host.engine._generated_context_carrier_remainder(first) is None
        assert str(first["content"]).lstrip().startswith(OBJECTIVE), "the commit emitted the objective block"
    assert host.engine._durable_commit_proof_payload() is not None


def grow_over_threshold(host):
    """Turns whose tool outputs stay under the externalization threshold (~10k chars) and that the aged stub
    tier shrinks, until the host is over the threshold."""
    while host.tokens() < host.engine.threshold_tokens + 1_000:
        host.turn(tool_words=1_900)


def stub_first_pass(host, caplog):
    """The over-threshold pass that returns via stub_first_exit; the host adopts the rewritten list."""
    before = deepcopy(host.live)
    caplog.clear()
    result, status = host.compress()
    assert status == "sanitized" and STUB_FIRST in caplog.text, "the over-threshold pass returns via stub_first_exit"
    assert result != before and any("Externalized tool output" in str(m.get("content")) for m in result)
    host.adopt(result)


def header_count(message):
    return len(SUMMARY_HEADER.findall(str(message.get("content"))))


def cold_rebind_problems(host, caplog):
    """Cold rebind on the same DB with the list the host holds, then further passes; what went wrong, if anything."""
    problems, rows = [], host.stored_rows()
    host.cold_rebind()
    if host.stored_rows() != rows:
        problems.append(f"the cold bind stored {host.stored_rows() - rows} new raw row(s) (the head, replayed as a new row)")
    unmapped = host.unmapped()
    if unmapped:
        problems.append(f"the store-id map misses {len(unmapped)} of the {len(host.live) - 1} rows after the head")
    frontier, wedged = host.drive(caplog)
    if wedged:
        problems.append("a pass stopped on 'leaf would end before an unresolved retained occurrence'")
    if host.frontier() <= frontier:
        problems.append(f"the compaction frontier did not advance after the rebind (stuck at {frontier})")
    return problems


def leave_proof_stale(host, caplog):
    """The stub-first pass of a version that wrote no proof for it: the store keeps the earlier proof, now stale."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LCMEngine, "_persist_compress_commit_proof", lambda self, proof: None)
        stub_first_pass(host, caplog)
    return host.engine._durable_commit_proof_payload()


@pytest.fixture
def stub_first_session(host_for, caplog):
    caplog.set_level(logging.INFO, logger="hermes_lcm")

    def build(head, *, stub_first):
        host = host_for(head)
        committed_session(host, head)
        grow_over_threshold(host)
        if stub_first:
            host.head_before = deepcopy(host.live[0])
            stub_first_pass(host, caplog)
        return host

    return build


# -- (a) pure-summary head ---------------------------------------------------------------------------------------


def test_pure_summary_head_cold_rebind_after_stub_first_pass_maps_and_progresses(stub_first_session, caplog):
    host = stub_first_session("summary", stub_first=True)
    assert host.engine._generated_context_carrier_remainder(host.live[0]) is None
    assert cold_rebind_problems(host, caplog) == []


# -- (b) carrier head (summary prefix + the glued real user row) -------------------------------------------------


def test_carrier_head_stub_first_pass_keeps_one_summary_prefix(stub_first_session):
    host = stub_first_session("carrier", stub_first=True)
    before, after = host.head_before, host.live[0]
    assert host.engine._generated_context_carrier_remainder(before) is not None
    assert header_count(after) == header_count(before), "the stub-first pass assembled the summary prefix again"
    assert len(after["content"]) == len(before["content"])


def test_carrier_head_cold_rebind_after_stub_first_pass_maps_and_progresses(stub_first_session, caplog):
    host = stub_first_session("carrier", stub_first=True)
    assert cold_rebind_problems(host, caplog) == []


# -- (c) control: the same flow without the stub-first pass --------------------------------------------------------


@pytest.mark.parametrize("head", ["summary", "carrier"])
def test_control_cold_rebind_without_stub_first_pass_maps_and_progresses(stub_first_session, caplog, head):
    host = stub_first_session(head, stub_first=False)
    assert cold_rebind_problems(host, caplog) == []


# -- (d) a store that already holds a stale proof (written before this fix) ---------------------------------------


@pytest.mark.parametrize("new_turn", [False, True], ids=["replay", "replay+new-turn"])
def test_stale_proof_from_an_earlier_version_is_retired_on_cold_rebind(stub_first_session, caplog, new_turn):
    """The upgrade restart is itself the cold bind: a proof an earlier version left stale must not wedge it."""
    host = stub_first_session("summary", stub_first=False)
    stale = leave_proof_stale(host, caplog)
    assert stale is not None and host.engine._cursor_from_durable_commit_proof(host.live) is None
    if new_turn:  # a gateway restart binds with the next user turn already appended
        host.clock += 0.5
        host.live.append({"role": "user", "content": "after the restart: " + host.words(25), "timestamp": host.clock})
    rows = host.stored_rows()
    host.cold_rebind()
    recon = host.engine._last_ingest_reconciliation
    assert recon["reason"] == "replayed durable tail past a stale commit proof", recon
    assert host.stored_rows() == rows + int(new_turn), "only the new turn is stored"
    assert host.unmapped() == []
    frontier, wedged = host.drive(caplog)
    assert not wedged and host.frontier() > frontier
    assert host.engine._durable_commit_proof_payload()["created_at"] != stale["created_at"], "a fresh proof replaced it"



# -- (e) the retirement changes a decision only on a replay of stored rows -----------------------------------------


def test_a_retry_that_only_skips_a_stale_snapshot_keeps_todays_decision(stub_first_session, caplog, monkeypatch):
    """Only a retry that replays stored rows confirms the retirement. One the no-proof reading takes for a stale
    no-overlap snapshot leaves today's decision (the batch persisted whole) and the durable proof in force."""
    host = stub_first_session("summary", stub_first=False)
    stale = leave_proof_stale(host, caplog)
    # the seam: only the retry (proof retired) finds no replayed tail and reads the list as a stale snapshot
    retired = lambda self: getattr(self, "_retired_durable_proof", None) is not None  # noqa: E731
    find = LCMEngine._find_reconciled_cursor_for_store_tail
    monkeypatch.setattr(LCMEngine, "_find_reconciled_cursor_for_store_tail",
                        lambda self, *args, **kwargs: None if retired(self) else find(self, *args, **kwargs))
    monkeypatch.setattr(LCMEngine, "_is_suspicious_stale_no_overlap_snapshot", lambda self, *_args: retired(self))
    host.cold_rebind()
    recon = host.engine._last_ingest_reconciliation
    assert (recon["action"], recon["reason"]) == ("persisted batch", "persisted ambiguous delta"), recon
    assert getattr(host.engine, "_retired_durable_proof", None) is None
    assert host.engine._active_emission_proof()["created_at"] == stale["created_at"]


def test_a_new_user_turn_equal_to_a_summary_is_stored_past_a_stale_proof(stub_first_session, caplog):
    """A genuinely new user row whose text equals one of this session's rendered summaries reads as an LCM head,
    and with the proof retired as scaffold. With no stored row replayed, the retry must not drop it."""
    host = stub_first_session("summary", stub_first=False)
    leave_proof_stale(host, caplog)
    summary = host.live[0]["content"]
    host.clock += 0.5
    host.live = [{"role": "user", "content": summary, "timestamp": host.clock}]
    rows = host.stored_rows()
    host.cold_rebind()
    recon = host.engine._last_ingest_reconciliation
    assert recon["reason"] == "persisted ambiguous delta", recon
    assert host.stored_rows() == rows + 1, "the new user row is stored"
    assert getattr(host.engine, "_retired_durable_proof", None) is None


def test_a_malformed_durable_proof_does_not_stop_the_ingest(stub_first_session):
    """A durable proof that does not parse is no proof: a new batch with no tail match is persisted."""
    host = stub_first_session("summary", stub_first=False)
    with host.engine._store._conn as conn:
        assert conn.execute("UPDATE metadata SET value = ? WHERE key LIKE ?",
                            ("{not json", "compaction_commit_proof%")).rowcount == 1
    host.clock += 0.5
    host.live = [{"role": "user", "content": "a new topic: " + host.words(25), "timestamp": host.clock},
                 {"role": "assistant", "content": "reply: " + host.words(30), "timestamp": host.clock + 0.5}]
    rows = host.stored_rows()
    host.cold_rebind()
    assert host.engine._last_ingest_reconciliation["reason"] == "persisted ambiguous delta"
    assert host.stored_rows() == rows + 2
