"""Child-only diagnostics and cross-thread native fallback lifecycle ownership."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest

from tests.harness import assistant, engine_context, pin_char_counter, user


def _view():
    return [user("request"), assistant("reply")]


@pytest.fixture
def native(monkeypatch):
    created, ended, resets = [], [], []
    events = []

    class NativeCompressor:
        compression_count = 0
        _last_compress_aborted = True

        def __init__(self, **kwargs):
            self.index = len(created)
            self.events = events
            created.append(self)
            events.append(("create", self.index))

        def compress(self, messages, **kwargs):
            self.compression_count += 1
            events.append(("compress", self.index))
            return messages

        def on_session_end(self, sid, messages):
            ended.append((sid, self))
            events.append(("end", sid, self.index))

        def on_session_reset(self):
            resets.append(self)
            events.append(("reset", self.index))

    module = ModuleType("agent.context_compressor")
    module.ContextCompressor = NativeCompressor
    monkeypatch.setitem(sys.modules, "agent.context_compressor", module)
    return created, ended, resets


def _config(kind):
    return {f"{kind}_session_patterns": ["cron*"]}


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
@pytest.mark.parametrize("case,status,reason", [
    ("empty", "noop", "empty message list"),
    ("below", "noop", "LCM bypassed below threshold"),
    ("forced", "host_fallback", "LCM bypassed:"),
])
def test_main_bypass_status_cross_thread(tmp_path, monkeypatch, kind, case, status, reason):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        view = [] if case == "empty" else _view()
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(engine.compress, view, force=case == "forced").result() == view
        assert engine.last_compression_status == status
        assert engine.last_compression_noop_reason.startswith(reason)
        public = engine.get_status()
        assert public["last_compression_status"] == status
        assert public["last_compression_noop_reason"] == engine.last_compression_noop_reason
        assert engine.last_compression_was_noop == (status == "noop")


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
def test_main_bypass_aborted_visible_on_loop_thread(tmp_path, monkeypatch, native, kind):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as pool:
            pool.submit(engine.compress, _view(), force=True).result()
        assert getattr(engine, "_last_compress_aborted", False) is True


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
def test_main_bypass_reuses_native_across_pool_threads(tmp_path, monkeypatch, native, kind):
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        engine.model = "test-model"
        # Both workers stay alive, so distinct pools guarantee distinct threads.
        with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
            first.submit(engine.compress, _view(), force=True).result()
            second.submit(engine.compress, _view(), force=True).result()
        assert len(created) == 1
        assert created[0].compression_count == 2
        engine.on_session_end("cron_1", [])
        assert ended == [("cron_1", created[0])]
        assert resets == created


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
def test_main_bypass_reset_recreates_native(tmp_path, monkeypatch, native, kind):
    pin_char_counter(monkeypatch)
    created, _, resets = native
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as pool:
            pool.submit(engine.compress, _view(), force=True).result()
            first = created[0]
            engine.on_session_reset()
            pool.submit(engine.compress, _view(), force=True).result()
        assert resets == [first]
        assert len(created) == 2
        assert created[1] is not first


@pytest.mark.parametrize("kind", ["ignore", "stateless", "child"])
@pytest.mark.parametrize("event", ["end", "rebind", "reset"])
def test_native_lifecycle_from_other_thread(tmp_path, monkeypatch, native, kind, event):
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    sid = "child" if kind == "child" else "cron_1"
    config = {} if kind == "child" else _config(kind)
    with engine_context(tmp_path, window=200_000, session_id="parent" if kind == "child" else sid,
                        **config) as engine:
        engine.model = "test-model"

        def compress():
            if kind == "child":
                engine._mark_thread_context_stateless(sid)
            try:
                engine.compress(_view(), force=True)
            finally:
                if kind == "child":
                    engine._clear_thread_context_stateless(sid)

        with ThreadPoolExecutor(1) as pool:
            pool.submit(compress).result()
            if event == "end":
                engine.on_session_end(sid, [])
            elif event == "reset":
                engine.on_session_reset()
            else:
                engine.on_session_start("next_parent", platform="cli", context_length=200_000)
        assert len(created) == 1
        assert ended == ([] if event == "reset" else [(sid, created[0])])
        assert resets == created


def test_next_child_starts_with_clean_diagnostics(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)

        def children():
            engine._mark_thread_context_stateless("first_child")
            engine.compress(_view(), force=True)
            engine._last_compress_aborted = True
            engine._clear_thread_context_stateless("first_child")
            engine._mark_thread_context_stateless("next_child")
            try:
                return (engine.last_compression_status, engine.last_compression_noop_reason,
                        getattr(engine, "_last_compress_aborted", None))
            finally:
                engine._clear_thread_context_stateless("next_child")

        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(children).result() == ("idle", "", None)


def test_late_reclassification_preserves_foreground_status(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="background") as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        monkeypatch.setattr(engine, "_in_process_auxiliary_caller_generation", lambda sid, **kwargs: 1)
        engine._last_compression_status = "compacted"
        proof = engine._compress_commit_proof = {"sentinel": True}

        def compress():
            result = engine.compress(_view(), force=True)
            return result, engine.last_compression_status

        with ThreadPoolExecutor(1) as pool:
            result, status = pool.submit(compress).result()
        assert result == _view()
        assert status == "host_fallback"
        assert engine._last_compression_status == "compacted"
        assert engine._compress_commit_proof is proof


def test_profile_rebind_discards_child_native_on_other_thread(tmp_path, monkeypatch, native):
    """Profile reconfiguration must not leave a worker holding the old native cache."""
    pin_char_counter(monkeypatch)
    created, _, _ = native
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"

        def child():
            engine._mark_thread_context_stateless("child")
            try:
                engine.compress(_view(), force=True)
            finally:
                engine._clear_thread_context_stateless("child")

        with ThreadPoolExecutor(1) as pool:
            pool.submit(child).result()
            engine.on_session_start("next_parent", hermes_home=str(tmp_path / "next_home"),
                                    platform="cli", context_length=200_000)
            pool.submit(child).result()
        assert len(created) == 2
        assert created[0] is not created[1]


def _compress_child(engine, sid="child"):
    engine._mark_thread_context_stateless(sid)
    try:
        assert engine.compress(_view(), force=True) == _view()
    finally:
        engine._clear_thread_context_stateless(sid)


@pytest.mark.parametrize("event", ["reset", "rebind", "reconfigure"])
@pytest.mark.xfail(strict=True, reason="R2-1: registry displaces another worker with the same child id")
def test_same_child_id_two_threads_lifecycle_then_end(tmp_path, monkeypatch, native, event):
    """Every worker cache must receive the base lifecycle sequence."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
            for pool in (first, second):
                pool.submit(_compress_child, engine).result()
            initial = list(created)
            assert len(initial) == 2
            if event == "reset":
                engine.on_session_reset()
            else:
                kwargs = {"hermes_home": str(tmp_path / "next_home")} if event == "reconfigure" else {}
                engine.on_session_start("next_parent", platform="cli", context_length=200_000, **kwargs)
            for pool in (first, second):
                pool.submit(_compress_child, engine).result()
            assert len(created) == 4, "each worker must create a fresh compressor after the boundary"
            engine.on_session_end("child", [])
        expected = [("create", 0), ("compress", 0), ("create", 1), ("compress", 1)]
        if event == "reset":
            expected += [("reset", 0), ("reset", 1)]
        elif event == "rebind":
            expected += [("end", "child", 0), ("reset", 0), ("end", "child", 1), ("reset", 1)]
        # Profile reconfiguration discards native caches without callbacks, as on base.
        expected += [("create", 2), ("compress", 2), ("create", 3), ("compress", 3),
                     ("end", "child", 2), ("reset", 2), ("end", "child", 3), ("reset", 3)]
        assert created[0].events == expected
        assert resets == (created[2:] if event == "reconfigure" else created)
        assert ended == [("child", item) for item in (created if event == "rebind" else created[2:])]
        with engine._auxiliary_session_lock:
            assert engine._child_host_fallback_states == {}


@pytest.mark.xfail(strict=True, reason="R2-1: registry displaces another worker with the same child id")
def test_same_child_id_two_threads_end_then_first_thread_reuses_id(tmp_path, monkeypatch, native):
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
            first.submit(_compress_child, engine).result()
            second.submit(_compress_child, engine).result()
            engine.on_session_end("child", [])
            assert ended == [("child", item) for item in created]
            first.submit(_compress_child, engine).result()
            assert len(created) == 3, "the first worker must not reuse an ended compressor"
            engine.on_session_end("child", [])
        assert created[0].events == [
            ("create", 0), ("compress", 0), ("create", 1), ("compress", 1),
            ("end", "child", 0), ("reset", 0), ("end", "child", 1), ("reset", 1),
            ("create", 2), ("compress", 2), ("end", "child", 2), ("reset", 2),
        ]
        assert resets == created


@pytest.mark.xfail(strict=True, reason="R2-3: inner child clear erases outer diagnostics")
def test_nested_child_clear_keeps_outer_diagnostics(tmp_path, monkeypatch, native):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"

        def diagnostics():
            return (engine.last_compression_status, engine.last_compression_noop_reason,
                    getattr(engine, "_last_compress_aborted", None))

        def nested():
            engine._mark_thread_context_stateless("outer")
            try:
                engine.compress(_view(), force=True)
                before = diagnostics()
                engine._mark_thread_context_stateless("inner")
                engine._clear_thread_context_stateless("inner")
                assert engine._thread_context_session_id() == "outer"
                assert diagnostics() == before == ("host_fallback", "LCM bypassed: auxiliary thread context", True)
            finally:
                engine._clear_thread_context_stateless()
            engine._mark_thread_context_stateless("next_child")
            try:
                assert diagnostics() == ("idle", "", None)
            finally:
                engine._clear_thread_context_stateless("next_child")

        with ThreadPoolExecutor(1) as pool:
            pool.submit(nested).result()


@pytest.mark.parametrize("event", ["reset", "end"])
@pytest.mark.xfail(strict=True, reason="R2-1: registry displaces another worker with the same child id")
def test_same_child_registry_barrier_interleavings(tmp_path, monkeypatch, native, event):
    """Exercise registry writes and lifecycle walks without taking a second lock."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    barrier = threading.Barrier(2, timeout=10)
    errors = []
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"

        def worker(parent):
            try:
                for _ in range(200):
                    # Seed both thread dictionaries for the same child before each race.
                    _compress_child(engine)
                    barrier.wait()
                    if parent:
                        with engine._auxiliary_session_lock:
                            if event == "reset":
                                engine.on_session_reset()
                            else:
                                engine.on_session_end("child", [])
                    else:
                        _compress_child(engine)
                    barrier.wait()
                    if parent:
                        # A compress admitted after the callback is ended at quiescence.
                        engine.on_session_end("child", [])
                    barrier.wait()
            except BaseException as exc:
                errors.append(exc)
                barrier.abort()

        threads = [threading.Thread(target=worker, args=(parent,), daemon=True) for parent in (False, True)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert not any(thread.is_alive() for thread in threads), "registry/lifecycle deadlock"
        assert not errors, errors
        assert len(created) >= 400
        assert sorted(item.index for item in resets) == list(range(len(created))), "lost or duplicate reset"
        ended_indices = [item.index for sid, item in ended if sid == "child"]
        assert len(ended_indices) == len(set(ended_indices)), "duplicate end"
        if event == "end":
            assert sorted(ended_indices) == list(range(len(created))), "lost end"
        with engine._auxiliary_session_lock:
            assert engine._child_host_fallback_states == {}
