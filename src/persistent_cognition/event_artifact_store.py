"""Immutable filesystem mirror for canonical Persistent Cognition events."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Iterator
from uuid import UUID, uuid5

from persistent_cognition import percept_journal
from persistent_cognition.private_storage import seek_lock_byte

ARTIFACT_SCHEMA_VERSION = 1


def artifact_root() -> Path:
    configured = os.environ.get("PCR_ARTIFACT_ROOT", "").strip()
    return Path(configured) if configured else Path(".pcr") / "artifacts"


def _event_dir() -> Path:
    return artifact_root() / "events"


def _record_path(event_id: UUID) -> Path:
    return _event_dir() / f"{event_id}.json"


def _commit_path(event_id: UUID) -> Path:
    return _event_dir() / f"{event_id}.commit.json"


def _stream_path(event_id: UUID) -> Path:
    return _event_dir() / f"{event_id}.jsonl"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"event artifact is not an object: {path}")
    return value


def _line(value: dict[str, Any]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _read_stream(
    path: Path, *, allow_partial_commit: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None, bytes]:
    return _parse_stream(path.read_bytes(), path, allow_partial_commit=allow_partial_commit)


def _parse_stream(
    raw: bytes, path: Path, *, allow_partial_commit: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None, bytes]:
    first, delimiter, remainder = raw.partition(b"\n")
    if not delimiter or not first:
        raise RuntimeError(f"incomplete event record: {path}")
    try:
        record = json.loads(first)
        commit_line, second_delimiter, tail = remainder.partition(b"\n")
        commit = json.loads(commit_line) if second_delimiter else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid event stream: {path}") from exc
    if not isinstance(record, dict) or (commit is not None and not isinstance(commit, dict)):
        raise RuntimeError(f"invalid event stream entry: {path}")
    partial = b"" if second_delimiter else remainder
    if tail or (partial and not allow_partial_commit):
        raise RuntimeError(f"incomplete or extra event stream entry: {path}")
    if record.get("artifact_type") != "EVENT_RECORD" or not verify_event_record(record):
        raise RuntimeError(f"invalid event record hash: {path}")
    if commit is not None and (
        commit.get("artifact_type") != "EVENT_DATABASE_COMMIT"
        or not verify_event_commit(commit)
        or commit.get("record_hash") != record["record_hash"]
        or commit.get("event_id") != record["event_id"]
        or commit.get("conversation_seq") != record["conversation_seq"]
    ):
        raise RuntimeError(f"invalid event commit entry: {path}")
    return record, commit, partial


def read_event_file(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read either event format; reject torn frames and conflicting histories."""

    if path.suffix == ".jsonl":
        record, commit, _partial = _read_stream(path)
        if path.stem != record["event_id"]:
            raise RuntimeError(f"misnamed event stream: {path}")
        return record, commit
    record = _load(path)
    if not verify_event_record(record):
        raise RuntimeError(f"event artifact hash verification failed: {path}")
    if path.stem != record["event_id"]:
        raise RuntimeError(f"misnamed event artifact: {path}")
    commit_path = path.with_name(f"{path.stem}.commit.json")
    commit = _load(commit_path) if commit_path.exists() else None
    if commit is not None and (
        not verify_event_commit(commit)
        or commit.get("record_hash") != record["record_hash"]
        or commit.get("event_id") != record["event_id"]
        or commit.get("conversation_seq") != record["conversation_seq"]
    ):
        raise RuntimeError(f"event record/commit mismatch: {path}")
    return record, commit


@contextmanager
def _locked_event_file(handle: BinaryIO) -> Iterator[None]:
    """Serialize same-event commit retries across processes on Windows and POSIX."""

    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        seek_lock_byte(handle)
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


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, encoded)


def _atomic_write_bytes(path: Path, encoded: bytes) -> None:
    from persistent_cognition.private_storage import atomic_private_write
    atomic_private_write(path, encoded)


def _semantic_record(
    *,
    event_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    conversation_seq: int,
    event_type: str,
    source: str,
    payload: dict[str, Any],
    payload_text: str | None,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "EVENT_RECORD",
        "event_id": str(event_id),
        "conversation_id": str(conversation_id),
        "correlation_id": str(correlation_id),
        "conversation_seq": conversation_seq,
        "event_type": event_type,
        "source": source,
        "payload": payload,
        "payload_text": payload_text,
    }


def write_event_record(
    *,
    event_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    conversation_seq: int,
    event_type: str,
    source: str,
    payload: dict[str, Any],
    payload_text: str | None,
    journal_id: UUID | None = None,
) -> dict[str, Any]:
    semantic = _semantic_record(
        event_id=event_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        conversation_seq=conversation_seq,
        event_type=event_type,
        source=source,
        payload=payload,
        payload_text=payload_text,
    )
    record = dict(semantic)
    record["record_hash"] = _digest(semantic)
    record["journaled_at"] = datetime.now(timezone.utc).isoformat()
    if journal_id is not None and journal_id not in {
        correlation_id, uuid5(conversation_id, f"interaction:{correlation_id}")
    }:
        raise ValueError("percept journal identity must match its correlation")
    legacy_path = _record_path(event_id)
    stream_path = _stream_path(event_id)
    if legacy_path.exists() and stream_path.exists():
        raise RuntimeError(f"event has both legacy and stream artifacts: {event_id}")
    if legacy_path.exists() or stream_path.exists():
        existing = (
            _load(legacy_path)
            if legacy_path.exists()
            else _read_stream(stream_path, allow_partial_commit=True)[0]
        )
        existing_semantic = {
            key: existing.get(key)
            for key in semantic
        }
        if (
            existing_semantic != semantic
            or existing.get("record_hash") != record["record_hash"]
            or not verify_event_record(existing)
        ):
            raise ValueError(f"conflicting event artifact retry: {event_id}")
        return existing
    scope = journal_id or percept_journal.scope_for(conversation_id, correlation_id)
    if scope is not None:
        path = percept_journal.path_for(scope)
        with percept_journal.locked(path, create=True) as handle:
            entries, partial = percept_journal.parse_unlocked(handle, path, allow_partial=True)
            if any(
                item.get("conversation_id") != str(conversation_id)
                or item.get("correlation_id") != str(correlation_id)
                for item in entries
                if item.get("artifact_type") == "EVENT_RECORD"
                or item.get("interaction_id") == str(scope)
            ):
                raise RuntimeError(f"percept journal scope conflicts with event: {path}")
            existing_records = [
                item for item in entries
                if item.get("artifact_type") == "EVENT_RECORD" and item.get("event_id") == str(event_id)
            ]
            if len(existing_records) > 1:
                raise RuntimeError(f"duplicate event in percept journal: {event_id}")
            if existing_records:
                existing = existing_records[0]
                if not verify_event_record(existing) or any(
                    existing.get(key) != value for key, value in semantic.items()
                ):
                    raise ValueError(f"conflicting event artifact retry: {event_id}")
                return existing
            if partial:
                raise RuntimeError(f"incomplete percept journal before new event: {path}")
            if any(item.get("event_id") == str(event_id) for item in entries):
                raise RuntimeError(f"orphaned event commit in percept journal: {event_id}")
            percept_journal.append_unlocked(handle, path, record)
        return record
    _atomic_write_bytes(stream_path, _line(record))
    return record


def write_event_commit(
    *,
    event_id: UUID,
    global_seq: int,
    conversation_seq: int,
    created_at: datetime,
    schema_version: int,
    conversation_id: UUID | None = None,
    correlation_id: UUID | None = None,
) -> dict[str, Any]:
    legacy_path = _record_path(event_id)
    stream_path = _stream_path(event_id)
    if legacy_path.exists() and stream_path.exists():
        raise RuntimeError(f"event has both legacy and stream artifacts: {event_id}")
    if not legacy_path.exists() and not stream_path.exists():
        if conversation_id is None or correlation_id is None:
            raise RuntimeError(f"event commit has no semantic artifact: {event_id}")
        scope = percept_journal.scope_for(conversation_id, correlation_id)
        if scope is None:
            raise RuntimeError(f"event commit has no percept journal: {event_id}")
        path = percept_journal.path_for(scope)
        with percept_journal.locked(path) as handle:
            entries, partial = percept_journal.parse_unlocked(handle, path, allow_partial=True)
            records = [
                item for item in entries
                if item.get("artifact_type") == "EVENT_RECORD" and item.get("event_id") == str(event_id)
            ]
            if len(records) != 1:
                raise RuntimeError(f"event commit has no unique semantic artifact: {event_id}")
            record = records[0]
            if not verify_event_record(record) or record["conversation_seq"] != conversation_seq:
                raise RuntimeError(f"invalid semantic artifact: {event_id}")
            semantic = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "EVENT_DATABASE_COMMIT",
                "event_id": str(event_id),
                "record_hash": record["record_hash"],
                "global_seq": global_seq,
                "conversation_seq": conversation_seq,
                "created_at": created_at.isoformat(),
                "schema_version": schema_version,
            }
            commit = {**semantic, "commit_hash": _digest(semantic)}
            existing = [
                item for item in entries
                if item.get("artifact_type") == "EVENT_DATABASE_COMMIT" and item.get("event_id") == str(event_id)
            ]
            if existing:
                if len(existing) != 1 or existing[0] != commit:
                    raise ValueError(f"conflicting event commit retry: {event_id}")
                if partial:
                    raise RuntimeError(f"incomplete percept journal after event commit: {path}")
                return existing[0]
            percept_journal.append_unlocked(handle, path, commit, partial_prefix=partial)
        return commit
    record = (
        _load(legacy_path)
        if legacy_path.exists()
        else _read_stream(stream_path, allow_partial_commit=True)[0]
    )
    if not verify_event_record(record) or record.get("event_id") != str(event_id):
        raise RuntimeError(f"invalid semantic artifact: {event_id}")
    if record["conversation_seq"] != conversation_seq:
        raise ValueError(f"event commit sequence conflicts with semantic artifact: {event_id}")
    semantic = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "EVENT_DATABASE_COMMIT",
        "event_id": str(event_id),
        "record_hash": record["record_hash"],
        "global_seq": global_seq,
        "conversation_seq": conversation_seq,
        "created_at": created_at.isoformat(),
        "schema_version": schema_version,
    }
    commit = dict(semantic)
    commit["commit_hash"] = _digest(semantic)
    if legacy_path.exists():
        path = _commit_path(event_id)
        if path.exists():
            existing = _load(path)
            if existing != commit:
                raise ValueError(f"conflicting event commit artifact retry: {event_id}")
            return existing
        _atomic_write(path, commit)
        return commit

    with stream_path.open("r+b") as handle, _locked_event_file(handle):
        # Read under the lock: another process may have completed the same retry.
        handle.seek(0)
        current_record, existing, partial = _parse_stream(
            handle.read(), stream_path, allow_partial_commit=True
        )
        if current_record["record_hash"] != record["record_hash"]:
            raise RuntimeError(f"event changed during commit append: {event_id}")
        if existing is not None:
            if existing != commit:
                raise ValueError(f"conflicting event commit artifact retry: {event_id}")
            return existing
        encoded = _line(commit)
        if partial and not encoded.startswith(partial):
            raise RuntimeError(f"conflicting incomplete event commit: {event_id}")
        handle.seek(0, os.SEEK_END)
        remainder = encoded[len(partial):]
        if handle.write(remainder) != len(remainder):
            raise OSError(f"short event commit write: {event_id}")
        handle.flush()
        os.fsync(handle.fileno())
    return commit


def verify_event_record(record: dict[str, Any]) -> bool:
    semantic = {
        key: value
        for key, value in record.items()
        if key not in {"record_hash", "journaled_at"}
    }
    return record.get("record_hash") == _digest(semantic)


def verify_event_commit(commit: dict[str, Any]) -> bool:
    semantic = {key: value for key, value in commit.items() if key != "commit_hash"}
    return commit.get("commit_hash") == _digest(semantic)


def iter_event_artifacts() -> list[dict[str, Any]]:
    directory = _event_dir()
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    paths = sorted((*directory.glob("*.json"), *directory.glob("*.jsonl"))) if directory.exists() else []
    for path in paths:
        if path.name.endswith(".commit.json"):
            continue
        record, commit = read_event_file(path)
        event_id = UUID(record["event_id"])
        if path.stem != str(event_id) or str(event_id) in seen:
            raise RuntimeError(f"duplicate or misnamed event artifact: {path}")
        seen.add(str(event_id))
        pairs.append({"record": record, "commit": commit})
    for path in directory.glob("*.commit.json") if directory.exists() else ():
        if path.name.removesuffix(".commit.json") not in seen:
            raise RuntimeError(f"orphaned event commit artifact: {path}")
    for path in sorted(percept_journal.root().glob("*.jsonl")) if percept_journal.root().exists() else ():
        by_id: dict[str, dict[str, Any]] = {}
        for entry in percept_journal.read(path):
            kind = entry.get("artifact_type")
            if kind not in {"EVENT_RECORD", "EVENT_DATABASE_COMMIT"}:
                continue
            event_id = entry.get("event_id")
            if not isinstance(event_id, str):
                raise RuntimeError(f"event entry has no identity: {path}")
            pair = by_id.setdefault(event_id, {})
            key = "record" if kind == "EVENT_RECORD" else "commit"
            if key in pair:
                raise RuntimeError(f"duplicate event entry {event_id}: {path}")
            pair[key] = entry
        for event_id, pair in by_id.items():
            record, commit = pair.get("record"), pair.get("commit")
            if record is None or not verify_event_record(record):
                raise RuntimeError(f"invalid event record {event_id}: {path}")
            if commit is not None and (
                not verify_event_commit(commit)
                or commit.get("record_hash") != record["record_hash"]
                or commit.get("conversation_seq") != record["conversation_seq"]
            ):
                raise RuntimeError(f"invalid event receipt {event_id}: {path}")
            if event_id in seen or str(UUID(event_id)) != event_id:
                raise RuntimeError(f"duplicate or malformed event id {event_id}: {path}")
            seen.add(event_id)
            pairs.append({"record": record, "commit": commit})
    pairs.sort(
        key=lambda item: (
            int(item["commit"]["global_seq"]) if item["commit"] else 2**63 - 1,
            item["record"]["journaled_at"],
            item["record"]["event_id"],
        )
    )
    return pairs
