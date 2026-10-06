"""Byte locks must use the OS cursor, not a buffered stream's logical cursor."""
from contextlib import contextmanager
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from persistent_cognition import event_artifact_store, percept_journal
from persistent_cognition.private_storage import seek_lock_byte


@pytest.fixture
def byte_lock_emulator(monkeypatch):
    """Model msvcrt's documented file-descriptor offset semantics on any OS."""
    owned = {}
    positions = []

    def locking(fd, mode, size):
        position = os.lseek(fd, 0, os.SEEK_CUR)
        positions.append((mode, position))
        if mode == 1:
            assert position == 0
            owned[fd] = (position, size)
        else:
            if owned.get(fd) != (position, size):
                raise PermissionError("unlock does not match the owned byte range")
            del owned[fd]

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2, locking=locking,
    ))
    # Replace only these modules' OS views; mutating global os.name would break
    # pathlib and would not exercise a real Linux buffered descriptor.
    for module in (percept_journal, event_artifact_store):
        monkeypatch.setattr(module, "os", SimpleNamespace(**{**vars(os), "name": "nt"}))
    return owned, positions


@pytest.mark.parametrize("journal", ["percept", "event"])
@pytest.mark.parametrize("body_fails", [False, True])
def test_partial_buffered_read_releases_the_same_byte_range(
    tmp_path, byte_lock_emulator, journal, body_fails,
):
    path = tmp_path / "journal.jsonl"
    original = b'{"index":0}\n{"index":1}\n{"index":2}\n{"index":3}\n'
    path.write_bytes(original)

    @contextmanager
    def owned_file():
        if journal == "percept":
            with percept_journal.locked(path) as handle:
                yield handle
        else:
            with path.open("r+b") as handle:
                with event_artifact_store._locked_event_file(handle):
                    yield handle

    def operation():
        with owned_file() as handle:
            assert handle.readline() == b'{"index":0}\n'
            if body_fails:
                raise RuntimeError("original read failure")

    if body_fails:
        with pytest.raises(RuntimeError, match="original read failure"):
            operation()
    else:
        operation()
    owned, positions = byte_lock_emulator
    assert owned == {}
    assert positions == [(1, 0), (2, 0)]
    assert path.read_bytes() == original


def test_repeated_pagination_does_not_leak_or_misrelease_byte_locks(tmp_path, byte_lock_emulator):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"index":0}\n{"index":1}\n{"index":2}\n{"index":3}\n')
    for _ in range(3):
        assert percept_journal.read_page(path, offset=1, page_size=2) == [
            {"index": 1}, {"index": 2},
        ]
    owned, positions = byte_lock_emulator
    assert owned == {}
    assert positions == [(1, 0), (2, 0)] * 3


def test_lock_position_synchronization_preserves_pending_writes(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b"first\nsecond\n")
    with path.open("r+b") as handle:
        handle.readline()
        handle.seek(0, os.SEEK_END)
        handle.write(b"third\n")
        seek_lock_byte(handle)
        assert os.lseek(handle.fileno(), 0, os.SEEK_CUR) == 0
        assert handle.read() == b"first\nsecond\nthird\n"
    assert path.read_bytes() == b"first\nsecond\nthird\n"


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows byte-range locks")
def test_native_windows_lock_blocks_another_process_then_releases_after_partial_read(tmp_path):
    path = tmp_path / "journal.jsonl"
    original = b'{"index":0}\n{"index":1}\n{"index":2}\n'
    path.write_bytes(original)
    child = """
import msvcrt, sys
with open(sys.argv[1], 'r+b') as handle:
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print('BUSY')
    else:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        print('FREE')
"""

    def probe():
        result = subprocess.run([sys.executable, "-c", child, str(path)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    with percept_journal.locked(path) as handle:
        assert handle.readline() == b'{"index":0}\n'
        assert probe() == "BUSY"
    assert probe() == "FREE"
    assert path.read_bytes() == original
