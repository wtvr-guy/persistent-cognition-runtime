"""Local adapters preserve raw content and emit typed, provenance-bearing percepts."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import Field

from prometheist.artifact_journal import artifact_root, _fsync_parent
from prometheist.percept_context import FrozenRecord

MAX_MEDIA_BYTES = 67_108_864
MEDIA_CHUNK_BYTES = 65_536
MAX_EVENT_STREAM_RECORDS = 64


class MediaReference(FrozenRecord):
    storage: Literal["LOCAL_CONTENT_HASH"] = "LOCAL_CONTENT_HASH"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=1, le=MAX_MEDIA_BYTES)
    mime_type: str = Field(min_length=1, max_length=128)
    # Descriptions are untrusted adapter metadata, not extracted media semantics.
    description: str = Field(default="", max_length=1024)


def media_path(reference: MediaReference) -> Path:
    return artifact_root() / "media" / reference.sha256


def verify_media(reference: MediaReference) -> Path:
    if (artifact_root() / "quarantine" / f"{reference.sha256}.json").exists():
        raise ValueError("media is quarantined")
    path = media_path(reference)
    if path.stat().st_size != reference.byte_length:
        raise ValueError("media length does not match its immutable reference")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MEDIA_CHUNK_BYTES), b""):
            digest.update(chunk)
    if digest.hexdigest() != reference.sha256:
        raise ValueError("media checksum does not match its immutable reference")
    return path


def preserve_media(path: Path, *, mime_type: str, description: str = "") -> MediaReference:
    directory = artifact_root() / "media"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    descriptor, temporary_name = tempfile.mkstemp(dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target, path.open("rb") as source:
            for chunk in iter(lambda: source.read(MEDIA_CHUNK_BYTES), b""):
                size += len(chunk)
                if size > MAX_MEDIA_BYTES:
                    raise ValueError("media exceeds the configured intake bound; split at the adapter")
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        reference = MediaReference(sha256=digest.hexdigest(), byte_length=size,
                                   mime_type=mime_type, description=description)
        destination = media_path(reference)
        try:
            os.link(temporary, destination)  # Atomic create; never overwrite canonical media.
        except FileExistsError:
            verify_media(reference)
        _fsync_parent(destination)
        return reference
    finally:
        temporary.unlink(missing_ok=True)
