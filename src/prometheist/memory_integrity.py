"""Tamper-evident hashing for the immutable evidence ledger.

The event log remains authoritative.  Integrity records are derived metadata:
they can be destroyed and rebuilt from the event ledger without changing the
events themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from prometheist.memory_kernel import MemoryEvent

HASH_ALGORITHM = "sha256-chain-v1"
GENESIS_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class IntegrityRecord:
    event_id: str
    global_seq: int
    previous_hash: str
    content_hash: str
    algorithm: str = HASH_ALGORITHM


@dataclass(frozen=True, slots=True)
class IntegrityVerification:
    valid: bool
    checked_events: int
    first_invalid_event_id: str | None = None
    reason: str | None = None


def canonical_event_bytes(event: MemoryEvent) -> bytes:
    document = {
        "event_id": event.event_id,
        "global_seq": event.global_seq,
        "conversation_id": event.conversation_id,
        "conversation_seq": event.conversation_seq,
        "event_type": event.event_type,
        "source": event.source,
        "created_at": event.created_at.isoformat(),
        "text": event.text,
        "payload": event.payload,
    }
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def hash_event(event: MemoryEvent, previous_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(previous_hash.encode("ascii"))
    digest.update(b"\x00")
    digest.update(canonical_event_bytes(event))
    return digest.hexdigest()


def build_integrity_chain(events: Iterable[MemoryEvent]) -> tuple[IntegrityRecord, ...]:
    ordered = sorted(events, key=lambda event: (event.global_seq, event.event_id))
    previous = GENESIS_HASH
    records: list[IntegrityRecord] = []
    for event in ordered:
        content_hash = hash_event(event, previous)
        records.append(
            IntegrityRecord(
                event_id=event.event_id,
                global_seq=event.global_seq,
                previous_hash=previous,
                content_hash=content_hash,
            )
        )
        previous = content_hash
    return tuple(records)


def verify_integrity_chain(
    events: Iterable[MemoryEvent], records: Iterable[IntegrityRecord]
) -> IntegrityVerification:
    ordered_events = sorted(events, key=lambda event: (event.global_seq, event.event_id))
    ordered_records = sorted(records, key=lambda record: (record.global_seq, record.event_id))
    if len(ordered_events) != len(ordered_records):
        return IntegrityVerification(
            valid=False,
            checked_events=0,
            reason="event/integrity record count mismatch",
        )

    previous = GENESIS_HASH
    for index, (event, record) in enumerate(zip(ordered_events, ordered_records), start=1):
        if record.event_id != event.event_id or record.global_seq != event.global_seq:
            return IntegrityVerification(
                valid=False,
                checked_events=index - 1,
                first_invalid_event_id=event.event_id,
                reason="integrity record is attached to the wrong event",
            )
        if record.previous_hash != previous:
            return IntegrityVerification(
                valid=False,
                checked_events=index - 1,
                first_invalid_event_id=event.event_id,
                reason="previous hash does not match chain state",
            )
        expected = hash_event(event, previous)
        if record.content_hash != expected:
            return IntegrityVerification(
                valid=False,
                checked_events=index - 1,
                first_invalid_event_id=event.event_id,
                reason="event content hash mismatch",
            )
        previous = record.content_hash

    return IntegrityVerification(valid=True, checked_events=len(ordered_events))
