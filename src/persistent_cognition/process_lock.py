"""Fail immediately if another app owns this private runtime."""
from contextlib import contextmanager
import os
from persistent_cognition.private_storage import private_directory, regular_file


@contextmanager
def instance_lock(root):
    path = root / "operator" / "app-instance.lock"
    private_directory(path.parent)
    with regular_file(path, write=True, create=True) as handle:
        try:
            if os.name == "nt":
                import msvcrt
                if not handle.seek(0, os.SEEK_END):
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another Persistent Cognition app is already using this profile") from None
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
