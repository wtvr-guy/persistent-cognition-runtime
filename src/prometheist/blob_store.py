"""Content-addressed blob store for large exact payloads.

Journal artifacts (see ``artifact_journal.py``) already hash-chain every field
they contain, but embedding large exact payloads (long generations, documents,
audio, ...) inline in every JSON artifact that references them would duplicate
those bytes across the interaction chain. This module stores exact bytes once,
addressed by their SHA-256 digest, and returns a small OCI-style descriptor

    {"mediaType": "...", "digest": "sha256:...", "size": N}

that a journal artifact can embed instead of the raw payload. The digest
identifies the bytes; nothing about storage location or filename participates
in identity.

Deduplication here is scoped to one local artifact root (one installation),
not globally across unrelated people/security domains: a shared global blob
store would let two otherwise-isolated domains learn that they hold identical
bytes merely by comparing digests. A future multi-tenant deployment should
partition ``blob_root()`` per security domain rather than widen this store.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from prometheist.artifact_journal import artifact_root

DIGEST_ALGORITHM = "sha256"


@dataclass(frozen=True, slots=True)
class BlobDescriptor:
    """An OCI-style content descriptor: exact bytes identified by digest and size."""

    media_type: str
    digest: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"mediaType": self.media_type, "digest": self.digest, "size": self.size}

    @staticmethod
    def from_dict(value: dict[str, Any]) -> "BlobDescriptor":
        return BlobDescriptor(
            media_type=str(value["mediaType"]),
            digest=str(value["digest"]),
            size=int(value["size"]),
        )


class BlobIntegrityError(ValueError):
    """Stored bytes for a digest no longer match that digest."""


def blob_root() -> Path:
    """Return the local content-addressed blob root, beside the artifact journal."""

    return artifact_root() / "blobs"


def _digest_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_ref(hex_digest: str) -> str:
    return f"{DIGEST_ALGORITHM}:{hex_digest}"


def _split_digest(digest: str) -> str:
    algorithm, separator, hex_digest = digest.partition(":")
    if not separator or algorithm != DIGEST_ALGORITHM or len(hex_digest) != 64:
        raise ValueError(f"unsupported or malformed digest: {digest}")
    return hex_digest


def blob_path(digest: str) -> Path:
    """Return the on-disk path for a ``sha256:<hex>`` digest.

    Layout mirrors the OCI/registry convention: ``blobs/sha256/<aa>/<hex>``.
    """

    hex_digest = _split_digest(digest)
    return blob_root() / DIGEST_ALGORITHM / hex_digest[:2] / hex_digest


def _fsync_parent(path: Path) -> None:
    """Best-effort directory fsync; Windows does not expose POSIX directory fsync."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def put_blob(data: bytes, *, media_type: str) -> BlobDescriptor:
    """Durably store exact bytes once, content-addressed by SHA-256.

    Idempotent: writing identical bytes twice is a no-op the second time. A
    colliding path with a different size is a corruption/attack signal, since
    two different byte strings must not legitimately share one SHA-256 digest.
    """

    hex_digest = _digest_hex(data)
    digest = _digest_ref(hex_digest)
    descriptor = BlobDescriptor(media_type=media_type, digest=digest, size=len(data))
    path = blob_path(digest)
    if path.exists():
        if path.stat().st_size != len(data):
            raise BlobIntegrityError(
                f"blob path exists with mismatched size for digest: {digest}"
            )
        return descriptor
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path.parent)
    return descriptor


def blob_exists(digest: str) -> bool:
    try:
        return blob_path(digest).exists()
    except ValueError:
        return False


def get_blob(digest: str) -> bytes:
    """Read and verify exact bytes for a digest.

    Raises ``FileNotFoundError`` when the blob is missing and
    ``BlobIntegrityError`` when stored bytes no longer match their digest.
    """

    path = blob_path(digest)
    if not path.exists():
        raise FileNotFoundError(f"blob not found: {digest}")
    data = path.read_bytes()
    if _digest_ref(_digest_hex(data)) != digest:
        raise BlobIntegrityError(f"stored blob no longer matches its digest: {digest}")
    return data


def verify_blob(digest: str) -> bool:
    """Return whether the stored bytes for ``digest`` still match that digest."""

    try:
        get_blob(digest)
    except (FileNotFoundError, BlobIntegrityError, ValueError):
        return False
    return True


def verify_descriptor(descriptor: BlobDescriptor) -> bool:
    """Return whether a full descriptor (digest *and* size) still matches storage."""

    try:
        data = get_blob(descriptor.digest)
    except (FileNotFoundError, BlobIntegrityError, ValueError):
        return False
    return len(data) == descriptor.size


def _looks_like_descriptor(value: dict[str, Any]) -> bool:
    digest = value.get("digest")
    return (
        isinstance(value.get("mediaType"), str)
        and isinstance(digest, str)
        and digest.startswith(f"{DIGEST_ALGORITHM}:")
        and isinstance(value.get("size"), int)
        and not isinstance(value.get("size"), bool)
    )


def iter_blob_descriptors(value: Any) -> Iterator[BlobDescriptor]:
    """Recursively find OCI-style blob descriptors embedded in a JSON structure.

    Any current or future artifact payload can embed a descriptor produced by
    ``put_blob`` anywhere in its structure; this walk lets generic tooling
    (the audit renderer, integrity checks, ...) discover and verify those
    references without each artifact producer registering them separately.
    """

    if isinstance(value, dict):
        if _looks_like_descriptor(value):
            try:
                yield BlobDescriptor.from_dict(value)
            except (KeyError, TypeError, ValueError):
                pass
            return
        for item in value.values():
            yield from iter_blob_descriptors(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_blob_descriptors(item)
