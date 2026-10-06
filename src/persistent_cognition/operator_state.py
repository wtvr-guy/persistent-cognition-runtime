"""Serialize operator policy edits so granting one scope cannot undo revocation."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from persistent_cognition.private_storage import (
    atomic_private_write, private_directory, regular_file, seek_lock_byte,
)


def write_private_policy(path: Path, value: dict):
    """Publish a flushed owner-only file; replacements never inherit the umask."""
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_private_write(path, encoded)


@contextmanager
def policy_lock(path: Path):
    private_directory(path.parent)
    with regular_file(path.with_suffix(path.suffix + ".lock"), write=True, create=True) as handle:
        if os.name == "nt":
            import msvcrt
            if not handle.seek(0, os.SEEK_END):
                handle.write(b"\0")
                handle.flush()
            seek_lock_byte(handle)
            # Windows retries a contended byte lock using its native bounded
            # LK_LOCK behavior. Failure aborts the edit rather than losing it.
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                seek_lock_byte(handle)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
