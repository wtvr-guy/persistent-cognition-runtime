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


def root() -> Path:
    configured = os.environ.get("PROMETHEIST_ARTIFACT_ROOT", "").strip()
    return (Path(configured) if configured else Path(".prometheist") / "artifacts") / "percepts"


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
        path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    with path.open("a+b" if create else "r+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                if create and not existed:
                    _fsync_directory(path.parent)
                yield handle
            finally:
                handle.seek(0)
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
    raw = handle.read()
    tail = raw.rsplit(b"\n", 1)[-1] if raw and not raw.endswith(b"\n") else b""
    if tail and not allow_partial:
        raise RuntimeError(f"incomplete percept journal: {path}")
    records = []
    for line in raw.split(b"\n")[:-1] if raw else ():
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid percept journal entry: {path}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"non-object percept journal entry: {path}")
        records.append(record)
    return records, tail


def entries_unlocked(handle: BinaryIO, path: Path) -> list[dict[str, Any]]:
    return parse_unlocked(handle, path)[0]


def read(path: Path) -> list[dict[str, Any]]:
    with locked(path) as handle:
        return entries_unlocked(handle, path)


def append_unlocked(
    handle: BinaryIO, path: Path, record: dict[str, Any], *, partial_prefix: bytes = b""
) -> None:
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8") + b"\n"
    if partial_prefix:
        if not encoded.startswith(partial_prefix):
            raise RuntimeError(f"conflicting incomplete percept journal entry: {path}")
        encoded = encoded[len(partial_prefix):]
    handle.seek(0, os.SEEK_END)
    if handle.write(encoded) != len(encoded):
        raise OSError(f"short percept journal append: {path}")
    handle.flush()
    os.fsync(handle.fileno())


def inspect(path: Path) -> dict[str, Any]:
    """Verify every entry, not merely the last line or the file digest."""

    from prometheist import artifact_journal, event_artifact_store

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
