"""One durable append-only work product for a percept and its workers.

Entries are complete JSON lines, independently hashed by their respective
event or interaction writers. Reads and appends take the same process lock so
observers never see an in-progress write. A torn final line fails closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Iterator
from uuid import UUID, uuid5
from persistent_cognition.private_storage import (
    acquire_windows_lock, private_directory, regular_file, seek_lock_byte, storage_admission,
)
from persistent_cognition.resource_limits import (
    DEFAULT_MAX_JOURNAL_BYTES, DEFAULT_MAX_JOURNAL_ENTRY_BYTES, DEFAULT_MAX_JOURNAL_RECORDS,
    ResourceLimitExceeded, limit,
)


def root() -> Path:
    configured = os.environ.get("PCR_ARTIFACT_ROOT", "").strip()
    return (Path(configured) if configured else Path(".pcr") / "artifacts") / "percepts"


def path_for(interaction_id: UUID) -> Path:
    return root() / f"{interaction_id}.jsonl"


def scope_for(conversation_id: UUID, correlation_id: UUID) -> UUID | None:
    """Resolve only an already-started percept; never invent one for seed events."""

    derived = uuid5(conversation_id, f"interaction:{correlation_id}")
    found = [candidate for candidate in (derived, correlation_id) if path_for(candidate).exists()]
    if len(found) > 1:
        raise RuntimeError(f"ambiguous percept journal correlation: {correlation_id}")
    return found[0] if found else None


@contextmanager
def locked(path: Path, *, create: bool = False) -> Iterator[BinaryIO]:
    if create:
        private_directory(path.parent)
    existed = path.exists()
    with regular_file(path, write=True, create=create) as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            acquire_windows_lock(handle)
            try:
                if create and not existed:
                    _fsync_directory(path.parent)
                yield handle
            finally:
                seek_lock_byte(handle)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                if create and not existed:
                    _fsync_directory(path.parent)
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def parse_unlocked(
    handle: BinaryIO, path: Path, *, allow_partial: bool = False
) -> tuple[list[dict[str, Any]], bytes]:
    handle.seek(0)
    maximum = limit("PCR_MAX_JOURNAL_BYTES", DEFAULT_MAX_JOURNAL_BYTES)
    entry_maximum = limit("PCR_MAX_JOURNAL_ENTRY_BYTES", DEFAULT_MAX_JOURNAL_ENTRY_BYTES)
    record_maximum = limit("PCR_MAX_JOURNAL_RECORDS", DEFAULT_MAX_JOURNAL_RECORDS)
    if os.fstat(handle.fileno()).st_size > maximum:
        raise ResourceLimitExceeded("journal exceeds PCR_MAX_JOURNAL_BYTES; archive or raise the explicit bound")
    records = []
    consumed = 0
    while line := handle.readline(entry_maximum + 1):
        consumed += len(line)
        if len(line) > entry_maximum or consumed > maximum or len(records) >= record_maximum:
            raise ResourceLimitExceeded("journal exceeds configured inspection bounds")
        if not line.endswith(b"\n"):
            if not allow_partial:
                raise RuntimeError(f"incomplete percept journal: {path}")
            return records, line
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid percept journal entry: {path}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"non-object percept journal entry: {path}")
        records.append(record)
    return records, b""


def entries_unlocked(handle: BinaryIO, path: Path) -> list[dict[str, Any]]:
    return parse_unlocked(handle, path)[0]


def read(path: Path) -> list[dict[str, Any]]:
    with locked(path) as handle:
        return entries_unlocked(handle, path)


def read_page(path: Path, *, offset: int = 0, page_size: int = 100) -> list[dict[str, Any]]:
    """Read one page under the process lock without buffering preceding entries."""
    if offset < 0 or page_size < 1:
        raise ValueError("journal page requires a nonnegative offset and positive size")
    maximum = limit("PCR_MAX_JOURNAL_ENTRY_BYTES", DEFAULT_MAX_JOURNAL_ENTRY_BYTES)
    record_maximum = limit("PCR_MAX_JOURNAL_RECORDS", DEFAULT_MAX_JOURNAL_RECORDS)
    if offset + page_size > record_maximum:
        raise ResourceLimitExceeded("journal page exceeds record budget")
    with locked(path) as handle:
        if os.fstat(handle.fileno()).st_size > limit("PCR_MAX_JOURNAL_BYTES", DEFAULT_MAX_JOURNAL_BYTES):
            raise ResourceLimitExceeded("journal exceeds PCR_MAX_JOURNAL_BYTES")
        records = []
        for index in range(offset + page_size):
            line = handle.readline(maximum + 1)
            if not line:
                break
            if len(line) > maximum:
                raise ResourceLimitExceeded("journal entry exceeds byte budget")
            if not line.endswith(b"\n"):
                raise RuntimeError(f"incomplete percept journal: {path}")
            if index >= offset:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise RuntimeError(f"non-object percept journal entry: {path}")
                records.append(record)
        return records


def append_unlocked(
    handle: BinaryIO, path: Path, record: dict[str, Any], *, partial_prefix: bytes = b""
) -> None:
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8") + b"\n"
    if len(encoded) > limit("PCR_MAX_JOURNAL_ENTRY_BYTES", DEFAULT_MAX_JOURNAL_ENTRY_BYTES):
        raise ResourceLimitExceeded("journal entry exceeds PCR_MAX_JOURNAL_ENTRY_BYTES")
    if partial_prefix:
        if not encoded.startswith(partial_prefix):
            raise RuntimeError(f"conflicting incomplete percept journal entry: {path}")
        encoded = encoded[len(partial_prefix):]
    handle.seek(0, os.SEEK_END)
    if handle.tell() + len(encoded) > limit("PCR_MAX_JOURNAL_BYTES", DEFAULT_MAX_JOURNAL_BYTES):
        raise ResourceLimitExceeded("journal append exceeds PCR_MAX_JOURNAL_BYTES")
    with storage_admission(len(encoded)):
        if handle.write(encoded) != len(encoded):
            raise OSError(f"short percept journal append: {path}")
        handle.flush()
        os.fsync(handle.fileno())


def inspect(path: Path) -> dict[str, Any]:
    """Verify every entry, not merely the last line or the file digest."""

    from persistent_cognition import artifact_journal, event_artifact_store

    entries = read(path)
    if not entries:
        raise RuntimeError(f"empty percept journal: {path}")
    events: dict[str, dict[str, Any]] = {}
    previous_id = previous_hash = None
    interaction_count = 0
    for entry in entries:
        kind = entry.get("artifact_type")
        if kind in {"EVENT_RECORD", "EVENT_DATABASE_COMMIT"}:
            event_id = entry.get("event_id")
            if not isinstance(event_id, str):
                raise RuntimeError(f"event entry without identity: {path}")
            if kind == "EVENT_RECORD":
                try:
                    conversation = UUID(entry["conversation_id"])
                    correlation = UUID(entry["correlation_id"])
                except (KeyError, ValueError, TypeError) as exc:
                    raise RuntimeError(f"event entry without valid scope: {path}") from exc
                if path.stem not in {
                    str(correlation), str(uuid5(conversation, f"interaction:{correlation}"))
                }:
                    raise RuntimeError(f"event entry outside its percept scope: {path}")
            key = "record" if kind == "EVENT_RECORD" else "commit"
            pair = events.setdefault(event_id, {})
            if key in pair:
                raise RuntimeError(f"duplicate event entry {event_id}: {path}")
            pair[key] = entry
        else:
            if entry.get("interaction_id") != path.stem:
                raise RuntimeError(f"unexpected interaction entry: {path}")
            interaction_count += 1
            if (
                entry.get("journal_sequence") != interaction_count
                or entry.get("previous_artifact_id") != previous_id
                or entry.get("previous_artifact_hash") != previous_hash
                or entry.get("payload_hash") != artifact_journal._sha256(entry.get("payload"))
                or entry.get("artifact_hash") != artifact_journal._sha256(
                    {key: value for key, value in entry.items() if key != "artifact_hash"}
                )
            ):
                raise RuntimeError(f"invalid interaction entry {interaction_count}: {path}")
            previous_id = entry["artifact_id"]
            previous_hash = entry["artifact_hash"]
    for event_id, pair in events.items():
        record, commit = pair.get("record"), pair.get("commit")
        if record is None or not event_artifact_store.verify_event_record(record):
            raise RuntimeError(f"invalid event record {event_id}: {path}")
        if commit is not None and (
            not event_artifact_store.verify_event_commit(commit)
            or commit.get("record_hash") != record["record_hash"]
            or commit.get("conversation_seq") != record["conversation_seq"]
        ):
            raise RuntimeError(f"invalid event commit {event_id}: {path}")
    return {
        "artifact_type": "PERCEPT_JOURNAL",
        "interaction_id": path.stem if interaction_count else None,
        "artifact_hash": previous_hash,
        "journal_sequence": interaction_count,
        "event_count": len(events),
        "uncommitted_events": sum("commit" not in pair for pair in events.values()),
        "record_count": len(entries),
    }
