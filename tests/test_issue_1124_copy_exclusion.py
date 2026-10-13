"""#1124: copies need positive evidence and actual summary coverage, never a frontier guess."""

import subprocess
import types
from collections import Counter
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.tokens import count_messages_tokens

from tests.test_issue_581_582_store_complete_survival import (
    _assert_contiguous,
    _engine,
    _frontier,
    _rows,
)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    instance = _engine(tmp_path, temporal_rollups_enabled=False)
    monkeypatch.setattr(
        instance, "_host_uid_lineage_key", lambda: ("test:lineage", None)
    )
    yield instance
    instance.shutdown()


def _text(content="same text", role="assistant"):
    return {"role": role, "content": content}


def _segment():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_copy",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_copy",
            "content": "result",
            "name": "read",
        },
    ]


def _append(engine, messages, session="S", conversation="conv"):
    return [
        engine._store.append(session, message, conversation_id=conversation)
        for message in messages
    ]


def _bind(engine, store_id, uid="original", kind="canonical", lineage="test:lineage"):
    engine._store.add_host_uid_bindings(lineage, [(store_id, uid, kind, "agree_new")])


def _node(engine, ids, *, before_commit=None):
    return engine._dag.add_node(
        SummaryNode(
            session_id="S",
            summary="Earlier turns",
            token_count=2,
            source_token_count=20,
            source_ids=ids,
            source_type="messages",
        ),
        before_commit=before_commit,
    )


def _walk(
    engine, *, chunk=(), claims=None, mapping=None, view=(), carry=(), budget=10_000
):
    return engine._store_complete_input(
        list(chunk),
        claims or {},
        mapping or {},
        list(view),
        list(chunk),
        _frontier(engine),
        list(carry),
        budget,
    )


def _ids(out, mapping=None):
    mapping = mapping or {}
    return [
        sid
        for message, claims in out
        for sid in [
            *claims,
            *([mapping[id(message)]] if id(message) in mapping else []),
        ]
    ]


def _publish(engine, out):
    ids = _ids(out)
    frontier = _frontier(engine)
    proofs = engine._store_complete_copy_exclusion_proofs
    _node(
        engine,
        ids,
        before_commit=lambda conn, node_id: (
            engine._lifecycle.stage_compaction_publication(
                conn,
                "conv",
                "S",
                node_id,
                frontier,
                ids,
                excluded_store_ids=engine._store_complete_excluded,
                copy_exclusion_proofs=proofs,
            )
        ),
    )


def _control(engine):
    """A proven copy alongside each negative control makes classifier precision red on the base."""
    twin, copy = _append(engine, [_text("control twin"), _text("control twin")])
    _bind(engine, twin, "control")
    return twin, copy


def _assert_control(engine, out, control):
    twin, copy = control
    assert twin in _ids(out) and copy not in _ids(out)
    assert engine._store_complete_copy_exclusion_proofs[copy] == twin


@pytest.mark.parametrize("evidence", ["E1", "E2"])
def test_01_same_leaf_copy_is_excluded(engine, evidence):
    messages = _segment() if evidence == "E1" else [_text()]
    originals = _append(engine, messages)
    copies = _append(engine, [{**message, "timestamp": 500.0} for message in messages])
    if evidence == "E2":
        _bind(engine, originals[0])
    end = _append(engine, [_text("next turn", "user")])[0]
    out = _walk(engine)
    assert _ids(out) == [*originals, end]
    assert engine._store_complete_copy_exclusion_proofs == dict(zip(copies, originals))
    assert set(copies) <= set(engine._store_complete_excluded)


def test_02_node_covered_twin_is_excluded_next_leaf(engine):
    twin = _append(engine, _segment())[1]
    _node(engine, [twin])
    copy = _append(engine, [_segment()[1]])[0]
    end = _append(engine, [_text("next", "user")])[0]
    out = _walk(engine)
    assert copy not in _ids(out) and end in _ids(out)
    assert engine._store_complete_copy_exclusion_proofs == {copy: twin}


@pytest.mark.parametrize(
    "text", ["continue", "deliberate long repeated instruction " * 400]
)
def test_03_genuine_user_repeat_with_own_uid_stays(engine, text):
    originals = _append(engine, [_text(text, "user"), _text(text, "user")])
    for sid in originals:
        _bind(engine, sid, f"occurrence-{sid}")
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert set(originals) <= set(_ids(out))
    _assert_control(engine, out, control)


@pytest.mark.parametrize(
    "binding", ["absent", "version", "other-lineage", "unresolved"]
)
def test_04_text_without_canonical_lineage_evidence_stays(engine, monkeypatch, binding):
    originals = _append(engine, [_text(), _text()])
    if binding == "version":
        _bind(engine, originals[0], kind="version")
    elif binding == "other-lineage":
        _bind(engine, originals[0], lineage="other:lineage")
    elif binding == "unresolved":
        _bind(engine, originals[0])
        monkeypatch.setattr(
            engine, "_host_uid_lineage_key", lambda: (None, "unresolved")
        )
    # E1 still works without any uid, including an unresolved lineage.
    twin, copy = _append(engine, [_segment()[1], _segment()[1]])
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert set(originals) <= set(_ids(out))
    assert twin in _ids(out) and copy not in _ids(out)


def test_05_retained_unmapped_copy_stays_blocked(engine):
    control = _control(engine)
    _append(engine, [_text("before blocked", "user")])
    twin, copy = _append(engine, [_text(), _text()])
    _bind(engine, twin)
    _node(engine, [twin])
    _append(engine, [_text("later", "user")])
    out = _walk(engine, view=[_text()])
    _assert_control(engine, out, control)
    assert copy not in engine._store_complete_excluded and copy not in _ids(out)
    assert max(_ids(out)) < copy


@pytest.mark.parametrize("claimed", [False, True], ids=["mapped", "claimed"])
def test_06_inverted_mapping_and_claimed_rows_are_never_copies(engine, claimed):
    control = _control(engine)
    twin, copy = _append(engine, [_segment()[1], _segment()[1]])
    message = _segment()[1]
    mapping = {} if claimed else {id(message): copy}
    claims = {id(message): [copy]} if claimed else {}
    _append(engine, [_text("end", "user")])
    out = _walk(engine, chunk=[message], mapping=mapping, claims=claims, view=[message])
    assert {twin, copy} <= set(_ids(out, mapping))
    _assert_control(engine, out, control)
    assert copy not in engine._store_complete_excluded


def test_07_frontier_is_not_summary_coverage(engine):
    twin = _append(engine, [_segment()[1]])[0]
    engine._lifecycle.stage_frontier_advance(
        engine._store.connection, "conv", "S", twin
    )
    copy = _append(engine, [_segment()[1]])[0]
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert copy in _ids(out) and copy not in engine._store_complete_excluded
    _assert_control(engine, out, control)


def test_08_publication_advances_contiguously_over_non_copies(engine, monkeypatch):
    control = _control(engine)
    end = _append(engine, [_text("end", "user")])[0]
    tail = [_text("fresh", "user"), _text("fresh reply")]
    tail_ids = _append(engine, tail)
    mapping = dict(zip(map(id, tail), tail_ids))
    before = _rows(engine)
    out = _walk(engine, view=tail, mapping=mapping)
    _assert_control(engine, out, control)
    monkeypatch.setattr(
        lcm_engine, "summarize_with_escalation", lambda **kwargs: ("Earlier turns", 1)
    )
    engine._ingest_cursor = len(tail)
    engine.compress(tail)
    assert engine._last_compression_status == "compacted", (
        engine._last_compression_noop_reason
    )
    assert _frontier(engine) == end and _rows(engine) == before
    coverage = Counter(
        sid for node in engine._dag.get_session_nodes("S") for sid in node.source_ids
    )
    assert all(coverage[sid] == 1 for sid in (control[0], end))
    assert coverage[control[1]] == 0 and coverage[control[0]] == 1
    _assert_contiguous(engine, {control[1]: control[0]})


def test_09_copy_only_backlog_is_drained_without_hidden_only_hold(engine, monkeypatch):
    twin = _append(engine, [_text()])[0]
    _bind(engine, twin)
    _node(engine, [twin])
    _append(engine, [_text(), _text()])
    # Fresh tail is mapped and retained. No summary call or hidden-only hold is warranted.
    tail = [_text("fresh", "user"), _text("fresh reply")]
    tail_ids = _append(engine, tail)
    mapping = dict(zip(map(id, tail), tail_ids))
    monkeypatch.setattr(
        engine, "_get_store_id_map_for_messages", lambda _messages: mapping
    )
    backlog = engine._store_complete_backlog(tail, 0)
    assert not backlog and backlog.rows == 0 and not backlog.truncated
    assert engine._hidden_backlog_status() == {"hidden_rows": 0}
    engine._ingest_cursor = len(tail)
    result = engine.compress(tail)
    assert result == tail and engine._last_compression_status != "compacted"
    assert engine._no_progress_hold is None


def test_10_exit_fit_and_uncovered_counter_use_covered_view_rows(engine):
    control = _control(engine)
    turns = [
        _text(f"turn {i} " * 30, role)
        for i in range(3)
        for role in ("user", "assistant")
    ]
    ids = _append(engine, turns)
    out = _walk(engine)
    _assert_control(engine, out, control)
    _publish(engine, out)
    mapping = dict(zip(map(id, turns), ids))
    budget = engine._survival_measure(turns[4:]) + 10
    cut = engine._survival_cut(
        turns, 0, budget, True, "exit_fit:compacted", mapping, True, keep_from=4
    )
    assert cut is not None
    fitted, count, dropped, projected, notice = cut
    assert dropped and control[1] not in dropped
    assert set(dropped) <= set(ids)
    engine._survival_record(
        "exit_fit:compacted",
        count,
        dropped,
        1000,
        engine._survival_measure(fitted),
        budget,
        projected,
        notice,
        warn_user=False,
    )
    assert engine._last_survival_fit["uncovered_rows"] == 0
    assert (
        engine._store.read_metadata_json("survival_fit:counter")["last_uncovered_rows"]
        == 0
    )
    # A view that explicitly maps to the excluded copy must still fail the #738 gate.
    assert (
        engine._survival_cut(
            turns,
            0,
            budget,
            True,
            "exit_fit:compacted",
            {**mapping, id(turns[0]): control[1]},
            True,
            keep_from=4,
        )
        is None
    )


def test_11_copy_injection_preserves_freed_per_leaf(tmp_path, monkeypatch):
    outcomes = []
    for copies in (False, True):
        instance = _engine(tmp_path / str(copies), temporal_rollups_enabled=False)
        monkeypatch.setattr(
            instance, "_host_uid_lineage_key", lambda: ("test:lineage", None)
        )
        try:
            first = _text("leaf text " * 35)
            twin = _append(instance, [first])[0]
            _bind(instance, twin)
            if copies:
                _append(instance, [first] * 4)
            _append(instance, [_text("next turn " * 35, "user"), _text("later " * 35)])
            out = _walk(instance, budget=260)
            messages = [m for m, _claims in out]
            outcomes.append((messages, count_messages_tokens(messages) - 2))
        finally:
            instance.shutdown()
    assert outcomes[0] == outcomes[1]


def test_12_v0262_reader_accepts_store_after_copy_publication(engine):
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    _assert_control(engine, out, control)
    _publish(engine, out)
    frontier = _frontier(engine)
    root = Path(__file__).resolve().parents[1]
    code = subprocess.check_output(
        ["git", "show", "v0.26.2:lifecycle_state.py"], cwd=root, text=True
    )
    module = types.ModuleType("hermes_lcm._rollback_1124")
    module.__package__ = "hermes_lcm"
    # Dataclasses require the module registered while loading the historical reader.
    import sys

    sys.modules[module.__name__] = module
    try:
        exec(compile(code, "v0.26.2:lifecycle_state.py", "exec"), module.__dict__)
        reader = module.LifecycleStateStore(engine._config.database_path)
        try:
            assert (
                reader.get_by_conversation("conv").current_frontier_store_id == frontier
            )
            later = _append(engine, [_text("after rollback", "user")])[0]
            _node(
                engine,
                [later],
                before_commit=lambda conn, node_id: reader.stage_compaction_publication(
                    conn,
                    "conv",
                    "S",
                    node_id,
                    frontier,
                    [later],
                    excluded_store_ids=[],
                ),
            )
            assert reader.get_by_conversation("conv").current_frontier_store_id == later
        finally:
            reader.close()
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize(
    "forgery",
    [
        "bytes",
        "uncovered",
        "consumed-not-summarized",
        "foreign-session",
        "foreign-conversation",
        "later",
        "not-excluded",
        "changed-before-commit",
    ],
)
def test_13_fence_rejects_forged_copy_proof(engine, forgery):
    original = _segment()[1]
    twin = _append(
        engine,
        [original],
        session="OTHER" if forgery == "foreign-session" else "S",
        conversation="other" if forgery == "foreign-conversation" else "conv",
    )[0]
    if forgery not in ("uncovered", "consumed-not-summarized"):
        _node(engine, [twin])
    copy = _append(engine, [_text("wrong bytes") if forgery == "bytes" else original])[
        0
    ]
    end = _append(engine, [_text("end", "user")])[0]
    proof = {copy: end if forgery == "later" else twin}
    before = _frontier(engine)

    def stage(conn, node_id):
        if forgery == "changed-before-commit":
            conn.execute(
                "UPDATE messages SET tool_name='changed' WHERE store_id=?", (twin,)
            )
        engine._lifecycle.stage_compaction_publication(
            conn,
            "conv",
            "S",
            node_id,
            before,
            [twin, end] if forgery == "consumed-not-summarized" else [end],
            excluded_store_ids=(
                [copy]
                if forgery == "consumed-not-summarized"
                else [twin, copy]
                if forgery != "not-excluded"
                else [twin]
            ),
            copy_exclusion_proofs=proof,
        )

    with pytest.raises(LifecyclePublicationConflictError):
        _node(engine, [end], before_commit=stage)
    assert _frontier(engine) == before
    assert not any(
        end in node.source_ids for node in engine._dag.get_session_nodes("S")
    )
    assert engine._store.get(twin).get("tool_name") != "changed"


@pytest.mark.parametrize(
    "risk", ["lossy", "bound-copy", "raw-json-bytes", "foreign-twin"]
)
def test_copy_evidence_is_fail_safe(engine, risk):
    original = _segment()[1]
    if risk == "lossy":
        original = {
            **original,
            "content": "[LCM sensitive redaction: name=password_assignment; chars=12]",
        }
    if risk == "raw-json-bytes":
        original = _segment()[0]
    twin = _append(
        engine, [original], session="OTHER" if risk == "foreign-twin" else "S"
    )[0]
    copy = _append(engine, [original])[0]
    if risk == "bound-copy":
        original = _text()
        engine._store.connection.execute(
            "UPDATE messages SET role='assistant', content=?, tool_call_id=NULL, "
            "tool_name=NULL WHERE store_id IN (?, ?)",
            ("text", twin, copy),
        )
        engine._store.connection.commit()
        _bind(engine, twin)
        _bind(engine, copy, "own-copy", kind="version")
    elif risk == "raw-json-bytes":
        engine._store.connection.execute(
            "UPDATE messages SET tool_calls=replace(tool_calls, ': ', ':') "
            "WHERE store_id=?",
            (copy,),
        )
        engine._store.connection.commit()
    control = _control(engine)
    _append(engine, [_text("end", "user")])
    out = _walk(engine)
    assert copy in _ids(out) and copy not in engine._store_complete_excluded
    _assert_control(engine, out, control)
