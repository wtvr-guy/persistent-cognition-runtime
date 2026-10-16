"""Owner-only storage, no-follow regular-file opens, and serialized disk admission."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import BinaryIO

from persistent_cognition.resource_limits import (
    DEFAULT_ARTIFACT_QUOTA_BYTES, DEFAULT_MIN_FREE_DISK_BYTES,
    ResourceLimitExceeded, limit,
)


def private_directory(path: Path) -> None:
    """Create private directories without chmod'ing unrelated existing parents."""
    if not path.exists():
        if not path.parent.exists():
            private_directory(path.parent)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    if path.is_symlink() or not path.is_dir():
        raise ValueError("private storage directory must not be a symlink")
    if os.name != "nt":
        path.chmod(0o700)


@contextmanager
def regular_file(path: Path, *, write: bool = False, create: bool = False):
    """POSIX walks directory descriptors with O_NOFOLLOW, closing path races.

    Windows checks links before opening; its ACL boundary must be provisioned by
    the operator. This helper is not an OS sandbox against a filesystem owner.
    """
    flags = os.O_RDWR if write else os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = None
    try:
        absolute = path.absolute()
        if os.name == "posix":
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            parent_fd = os.open(absolute.anchor, directory_flags)
            for component in absolute.parts[1:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = next_fd
            descriptor = os.open(absolute.name, flags, 0o600, dir_fd=parent_fd)
        else:
            if any(item.is_symlink() for item in (absolute, *absolute.parents)):
                raise ValueError("private storage must not traverse symlinks")
            descriptor = os.open(absolute, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("private storage requires a regular file")
            if write and os.name == "posix":
                os.fchmod(descriptor, 0o600)
            handle = os.fdopen(descriptor, "r+b" if write else "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            yield handle
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def seek_lock_byte(handle: BinaryIO) -> None:
    """Position the OS descriptor at byte zero, even after buffered read-ahead.

    A buffered seek can move only Python's logical cursor. Seeking from EOF
    first synchronizes and clears that buffer before the absolute seek used by
    Windows byte-range locking. Pending writes are flushed by the seek.
    """
    handle.seek(0, os.SEEK_END)
    handle.seek(0)


WINDOWS_LOCK_TIMEOUT_SECONDS = 10.0
WINDOWS_LOCK_RETRY_INTERVAL_SECONDS = 0.05


def acquire_windows_lock(handle: BinaryIO) -> None:
    """Acquire byte zero with bounded retries that promptly observe releases.

    CRT LK_LOCK retries only once per second and can exhaust its ten attempts
    when many short edits contend. Nonblocking attempts avoid that coarse wait;
    the monotonic deadline retains the bounded, fail-closed edit policy. Windows
    permits locking beyond EOF, so empty lock files need no unprotected write.
    """
    import msvcrt

    deadline = time.monotonic() + WINDOWS_LOCK_TIMEOUT_SECONDS
    while True:
        seek_lock_byte(handle)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(WINDOWS_LOCK_RETRY_INTERVAL_SECONDS, remaining))


def _store_root() -> Path:
    configured = os.environ.get("PCR_ARTIFACT_ROOT", "").strip()
    return Path(configured) if configured else Path(".pcr") / "artifacts"


@contextmanager
def storage_admission(additional_bytes: int):
    """Serialize root quota checks with writes across runtime processes.

    Counts filesystem bytes, including temporary files, without loading contents.
    External database/model storage needs its own OS/database quota.
    """
    root = _store_root()
    private_directory(root)
    with regular_file(root / ".storage.lock", write=True, create=True) as lock:
        if os.name == "nt":
            import msvcrt
            acquire_windows_lock(lock)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            quota = limit("PCR_ARTIFACT_QUOTA_BYTES", DEFAULT_ARTIFACT_QUOTA_BYTES)
            used = 0
            for directory, children, files in os.walk(root, followlinks=False):
                for name in (*children, *files):
                    path = Path(directory) / name
                    info = path.lstat()
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError("artifact root contains a symlink")
                    if stat.S_ISREG(info.st_mode):
                        used += info.st_size
                if used + additional_bytes > quota:
                    raise ResourceLimitExceeded("artifact root exceeds PCR_ARTIFACT_QUOTA_BYTES")
            if used + additional_bytes > quota:
                raise ResourceLimitExceeded("artifact root exceeds PCR_ARTIFACT_QUOTA_BYTES")
            reserve = limit("PCR_MIN_FREE_DISK_BYTES", DEFAULT_MIN_FREE_DISK_BYTES)
            if shutil.disk_usage(root).free < reserve + additional_bytes:
                raise ResourceLimitExceeded("disk admission would consume protected free space")
            yield
        finally:
            if os.name == "nt":
                seek_lock_byte(lock)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def atomic_private_write(path: Path, data: bytes) -> None:
    private_directory(path.parent)
    with storage_admission(len(data)):
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            from persistent_cognition.artifact_journal import _replace_with_retry, _fsync_parent
            _replace_with_retry(temporary, path)
            _fsync_parent(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def remediate_permissions(root: Path) -> dict[str, int]:
    """Explicit migration of existing stores; reject links rather than follow them."""
    if os.name == "nt":
        raise ValueError("Windows stores require an owner-only ACL; chmod cannot establish it")
    private_directory(root)
    directories = files = 0
    for directory, children, names in os.walk(root, followlinks=False):
        private_directory(Path(directory))
        directories += 1
        for name in children:
            if (Path(directory) / name).is_symlink():
                raise ValueError("cannot remediate a symlink in private storage")
        for name in names:
            with regular_file(Path(directory) / name, write=True):
                files += 1
    return {"directories": directories, "files": files}
