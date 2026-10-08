"""Bounded byte-lock retries and native Windows contention regressions."""
from concurrent.futures import ThreadPoolExecutor
import errno
import os
import sys
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from persistent_cognition import operator_state, private_storage


@pytest.fixture
def lock_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(delay):
        clock.sleeps.append(delay)
        clock.now += delay

    monkeypatch.setattr(private_storage, "time", SimpleNamespace(
        monotonic=lambda: clock.now, sleep=sleep,
    ))
    return clock


@pytest.mark.parametrize("busy_errno", [errno.EACCES, errno.EAGAIN, errno.EDEADLK])
def test_contended_lock_retries_nonblocking_and_preserves_byte_zero(
    tmp_path, monkeypatch, lock_clock, busy_errno,
):
    calls = []

    def locking(fd, mode, size):
        assert mode == 1 and size == 1
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        calls.append(mode)
        if len(calls) < 3:
            raise OSError(busy_errno, "lock busy")

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=1, locking=locking))
    with (tmp_path / "lock").open("w+b") as handle:
        private_storage.acquire_windows_lock(handle)
        assert handle.read() == b""  # Locking does not need a seed write.
    assert len(calls) == 3
    assert lock_clock.sleeps == [private_storage.WINDOWS_LOCK_RETRY_INTERVAL_SECONDS] * 2


def test_contended_lock_exhausts_deadline_without_running_policy_edit(
    tmp_path, monkeypatch, lock_clock,
):
    busy = OSError(errno.EACCES, "lock busy")
    modes = []

    def locking(fd, mode, size):
        modes.append(mode)
        raise busy

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2, locking=locking,
    ))
    monkeypatch.setattr(operator_state, "os", SimpleNamespace(**{**vars(os), "name": "nt"}))
    with pytest.raises(OSError) as caught:
        with operator_state.policy_lock(tmp_path / "policy.json"):
            pytest.fail("a policy edit must never run without its lock")
    assert caught.value is busy
    assert lock_clock.now == pytest.approx(private_storage.WINDOWS_LOCK_TIMEOUT_SECONDS)
    assert set(modes) == {1}  # Never unlock a lock that was not acquired.
    assert all(0 < delay <= private_storage.WINDOWS_LOCK_RETRY_INTERVAL_SECONDS
               for delay in lock_clock.sleeps)


def test_non_contention_errors_fail_immediately(tmp_path, monkeypatch, lock_clock):
    failure = OSError(errno.EBADF, "invalid descriptor")

    def locking(*args):
        raise failure

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=1, locking=locking))
    with (tmp_path / "lock").open("w+b") as handle:
        with pytest.raises(OSError) as caught:
            private_storage.acquire_windows_lock(handle)
    assert caught.value is failure
    assert lock_clock.sleeps == []


@pytest.mark.parametrize("native", [False, pytest.param(
    True, marks=pytest.mark.skipif(os.name != "nt", reason="requires native Windows byte-range locks"),
)])
def test_waiter_observes_release_without_losing_policy_updates(tmp_path, monkeypatch, native):
    if native:
        import msvcrt
    else:
        owned = Lock()

        def emulate_locking(fd, mode, size):
            assert os.lseek(fd, 0, os.SEEK_CUR) == 0 and size == 1
            if mode == 2:
                owned.release()
            elif not owned.acquire(blocking=False):
                raise OSError(errno.EACCES, "lock busy")

        msvcrt = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, LK_LOCK=3, locking=emulate_locking)
        monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
        monkeypatch.setattr(operator_state, "os", SimpleNamespace(**{**vars(os), "name": "nt"}))

    path = tmp_path / "policy.json"
    path.write_text("0", encoding="utf-8")
    attempted = Event()
    native_locking = msvcrt.locking

    def observed_locking(fd, mode, size):
        assert mode != msvcrt.LK_LOCK, "coarse CRT retries can starve concurrent edits"
        try:
            return native_locking(fd, mode, size)
        except OSError:
            attempted.set()
            raise

    monkeypatch.setattr(msvcrt, "locking", observed_locking)

    def edit():
        with operator_state.policy_lock(path):
            value = int(path.read_text(encoding="utf-8"))
            path.write_text(str(value + 1), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=13) as pool:
        with operator_state.policy_lock(path):
            futures = [pool.submit(edit) for _ in range(13)]
            assert attempted.wait(timeout=5), "waiters must encounter the owned lock"
            assert path.read_text(encoding="utf-8") == "0"
        for future in futures:
            future.result(timeout=20)
    assert path.read_text(encoding="utf-8") == "13"
