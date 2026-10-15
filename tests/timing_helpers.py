"""Ordering proofs for calls that must return before a blocker is released."""

import threading


def returns_while_blocked(call, release):
    results, errors = [], []

    def run():
        try:
            results.append(call())
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=run, daemon=True)
    caller.start()
    try:
        caller.join(timeout=10)
        assert not caller.is_alive(), "call waited for the blocker to be released"
        assert not release.is_set(), "blocker was released before the call returned"
        if errors:
            raise errors[0]
        return results[0]
    finally:
        release.set()
        caller.join(timeout=10)


class ContentionProbe:
    """Expose an actual failed lock acquisition instead of sleeping for a race."""

    def __init__(self, lock):
        self.lock = lock
        self.contended = threading.Event()
        self.attempts = []
        self.acquired = threading.Event()
        self.release_guard = None
        self.returned_before_release = False

    def acquire(self, blocking=True, timeout=-1):
        if self.lock.acquire(blocking=False):
            return True
        self.attempts.append((blocking, timeout))
        self.contended.set()
        if not blocking:
            return False
        acquired = self.lock.acquire(timeout=timeout)
        if self.release_guard is not None and not self.release_guard.is_set():
            self.returned_before_release = True
        self.acquired.set()
        return acquired

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()
