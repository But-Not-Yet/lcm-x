"""I1/I2: preserve a pending predecessor before the host merges it with the next turn."""
from copy import deepcopy
import logging

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1501_rotation_stamp import lineage, message
from tests.test_packaging_install import _load_plugin_entrypoint_module


def register(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('LCM_DATABASE_PATH', str(tmp_path / 'lcm.db'))
    module = _load_plugin_entrypoint_module('hermes_lcm_pending_predecessor')

    class Context:
        def __init__(self):
            self.hooks = {}

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            self.hooks[name] = callback

    ctx = Context()
    module.register(ctx)
    ctx.module = module
    return ctx


@pytest.mark.parametrize('rotation', [False, True], ids=['in-place', 'rotation'])
def test_pending_predecessor_composite_and_cold_replay(tmp_path, monkeypatch, rotation):
    """Crash before T33's reply; the next pre-hook precedes repair and stamp carry-over."""
    lineage(tmp_path)
    q, p, r = message('reply T32', 9., 'assistant'), message('[T33] pending', 10.), message('[T34] current', 20.)
    reply = message('reply T34', 21., 'assistant')
    session = 'child' if rotation else 'parent'
    ctx = register(tmp_path, monkeypatch)
    try:
        e = ctx.engine
        e.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        e.ingest([deepcopy(q)])
        if rotation:
            e.on_session_start(session, platform='cli', context_length=128000, conversation_id='conv',
                               boundary_reason='compression')
        ctx.hooks['pre_llm_call'](session_id=session, conversation_id='conv',
                                  conversation_history=[deepcopy(q), deepcopy(p), deepcopy(r)])
        # The current request must still be stored through normal ingest, at the host's carried stamp.
        assert [row['content'] for row in e._store.get_session_messages(session)] == (
            [p['content']] if rotation else [q['content'], p['content']])
        e.ingest([deepcopy(q), {**r, 'timestamp': 10.}, deepcopy(reply)])
        composite = [deepcopy(q), message(p['content'] + '\n\n' + r['content'], 10.), deepcopy(reply)]
        e.on_session_end(session, composite)
        before = [row for owner in ('parent', 'child') for row in e._store.get_session_messages(owner)]
        assert [row['content'] for row in before] == [q['content'], p['content'], r['content'], reply['content']]
        carried = next(row for row in before if row['content'] == r['content'])
        assert e._store.get_message_relations([carried['store_id']], 'carried_stamp')
    finally:
        ctx.engine.shutdown()
    with engine_context(tmp_path, window=128000, session_id=session, conversation_id='conv') as e:
        e.ingest(deepcopy([p, r, reply]))
        assert [row for owner in ('parent', 'child') for row in e._store.get_session_messages(owner)] == before
        e.ingest(deepcopy([p, r, reply, message(r['content'], 30.)]))
        copies = [row for owner in ('parent', 'child') for row in e._store.get_session_messages(owner)
                  if row['content'] == r['content']]
        assert [row['observed_at'] for row in copies] == [10., 30.]


@pytest.mark.parametrize('previous', [message('completed reply', 9., 'assistant'),
                                     message('Earlier turns have been compacted.\nsummary', 9.)],
                         ids=['completed-turn', 'scaffold'])
def test_premerge_capture_skips_other_predecessors(tmp_path, monkeypatch, previous):
    ctx = register(tmp_path, monkeypatch)
    try:
        ctx.engine.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        monkeypatch.setattr(ctx.engine, 'ingest', lambda _rows: pytest.fail('not a pending user pair'))
        assert ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                         conversation_history=[deepcopy(previous), message('current', 20.)]) is None
    finally:
        ctx.engine.shutdown()


def test_premerge_scaffold_failure_preserves_context(tmp_path, monkeypatch, caplog):
    ctx = register(tmp_path, monkeypatch)
    try:
        e = ctx.engine
        e.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        expected = object()
        context_calls, ingested = [], []
        payload = dict(session_id='parent', conversation_id='conv',
                       conversation_history=[message('pending', 10.), message('current', 20.)])

        def fail_scaffold(_row):
            raise RuntimeError('private message content must never be logged')

        def context(active_engine, hook_payload):
            context_calls.append((active_engine, hook_payload))
            return expected

        monkeypatch.setattr(e, '_identity_is_lcm_scaffold', fail_scaffold)
        monkeypatch.setattr(e, 'ingest', ingested.append)
        monkeypatch.setattr(ctx.module, '_pre_llm_context', context)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=ctx.module.logger.name):
            assert ctx.hooks['pre_llm_call'](**payload) is expected
        assert context_calls == [(e, payload)]
        assert ingested == []
        assert e._store.get_session_messages('parent') == []
        assert [(record.levelname, record.getMessage()) for record in caplog.records] == [
            ('WARNING', 'LCM pending-predecessor capture failed (RuntimeError); continuing pre-LLM context')]
        assert caplog.records[0].exc_info is None
    finally:
        ctx.engine.shutdown()


@pytest.mark.parametrize('history', [[None, message('current', 20.)], [message('pending', 10.), None]],
                         ids=['non-dict-predecessor', 'non-dict-current'])
def test_premerge_non_dict_history_preserves_context(tmp_path, monkeypatch, history):
    ctx = register(tmp_path, monkeypatch)
    try:
        e = ctx.engine
        e.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        expected = object()
        monkeypatch.setattr(e, 'ingest', lambda _rows: pytest.fail('malformed history must not be ingested'))
        monkeypatch.setattr(ctx.module, '_pre_llm_context', lambda _engine, _payload: expected)
        assert ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                         conversation_history=history) is expected
        assert e._store.get_session_messages('parent') == []
    finally:
        ctx.engine.shutdown()


def test_same_session_marker_does_not_cross_conversation(tmp_path):
    q, p, r = message('earlier reply', 9., 'assistant'), message('previous', 10.), message('current', 10.)
    with engine_context(tmp_path, window=128000, session_id='parent', conversation_id='A') as e:
        e.ingest([deepcopy(q), deepcopy(p)])
        e.ingest([deepcopy(q), deepcopy(r)])
        stored = e._store.get_session_messages('parent')
        assert e._store.get_message_relations([stored[-1]['store_id']], 'carried_stamp')
    with engine_context(tmp_path, window=128000, session_id='parent', conversation_id='B') as e:
        e.ingest([message('unrelated request', 31.), message('unrelated reply', 32., 'assistant')])
        e.ingest([deepcopy(p), message('current', 20.)])
        rows = e._store.get_session_messages('parent')
        assert any(row['content'] == 'current' and row['conversation_id'] == 'B' and row['observed_at'] == 20.
                   for row in rows)
