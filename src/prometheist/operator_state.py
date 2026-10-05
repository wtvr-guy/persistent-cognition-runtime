"""Serialize operator policy edits so granting one scope cannot undo revocation."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

from prometheist.artifact_journal import _fsync_parent, _replace_with_retry


def write_private_policy(path: Path, value: dict):
    """Publish a flushed owner-only file; replacements never inherit the umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
        _fsync_parent(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def policy_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if not handle.seek(0, os.SEEK_END):
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            # Windows retries a contended byte lock using its native bounded
            # LK_LOCK behavior. Failure aborts the edit rather than losing it.
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
