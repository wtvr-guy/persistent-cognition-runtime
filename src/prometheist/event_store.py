"""Event store: the sole database path for canonical persisted events.

Persistence is unconditional and automatic.  Every event is also mirrored into
the independent JSON artifact journal so PostgreSQL is an indexed operational
representation rather than the only surviving copy of canonical history.
"""
from __future__ import annotations

import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist import event_artifact_store
from prometheist.db import require_row
from prometheist.models import Event, EventType


def start_conversation(conn: psycopg.Connection, conversation_id: uuid.UUID | None = None) -> uuid.UUID:
    """Create a new conversation, or a no-op if `conversation_id` already exists."""
    conversation_id = conversation_id or uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations (conversation_id)
            VALUES (%s)
            ON CONFLICT (conversation_id) DO NOTHING
            """,
            (conversation_id,),
        )
    conn.commit()
    return conversation_id


def _journal_committed_event(event: Event, payload_text: str | None) -> None:
    """Ensure both the semantic record and DB-commit metadata exist on disk."""

    event_artifact_store.write_event_record(
        event_id=event.event_id,
        conversation_id=event.conversation_id,
        correlation_id=event.correlation_id,
        conversation_seq=event.conversation_seq,
        event_type=event.event_type.value,
        source=event.source,
        payload=event.payload,
        payload_text=payload_text,
    )
    event_artifact_store.write_event_commit(
        event_id=event.event_id,
        global_seq=event.global_seq,
        conversation_seq=event.conversation_seq,
        created_at=event.created_at,
        schema_version=event.schema_version,
        conversation_id=event.conversation_id,
        correlation_id=event.correlation_id,
    )


def record_event(
    conn: psycopg.Connection,
    *,
    conversation_id: uuid.UUID,
    correlation_id: uuid.UUID,
    event_type: EventType,
    source: str,
    payload: dict[str, Any],
    payload_text: str | None = None,
    event_id: uuid.UUID | None = None,
    journal_id: uuid.UUID | None = None,
) -> Event:
    """Append one canonical event and durably mirror it outside PostgreSQL.

    The semantic JSON record is fsync'd before the database insert is allowed to
    commit.  A second immutable commit artifact records PostgreSQL sequence/time
    metadata after commit.  Therefore a crash can leave an uncommitted semantic
    artifact, but it cannot leave a committed new database event with no semantic
    artifact.  Retrying the same deterministic event repairs a missing commit
    artifact without duplicating history.
    """

    event_id = event_id or uuid.uuid4()
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT event_id, conversation_id, correlation_id, global_seq,
                       conversation_seq, event_type, source, created_at,
                       payload, payload_text, schema_version
                FROM events WHERE event_id = %s
                """,
                (event_id,),
            )
            existing = cur.fetchone()
            if existing is not None:
                stored = _row_to_event(existing)
                if (
                    stored.conversation_id != conversation_id
                    or stored.correlation_id != correlation_id
                    or stored.event_type is not event_type
                    or stored.source != source
                    or stored.payload != payload
                    or existing["payload_text"] != payload_text
                ):
                    raise ValueError("conflicting deterministic event retry")
                conn.commit()
                _journal_committed_event(stored, existing["payload_text"])
                return stored

            cur.execute(
                "SELECT next_event_seq FROM conversations WHERE conversation_id = %s FOR UPDATE",
                (conversation_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Unknown conversation_id: {conversation_id}")
            conversation_seq = int(row["next_event_seq"])

            # Independent semantic durability precedes DB commit.
            event_artifact_store.write_event_record(
                event_id=event_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                conversation_seq=conversation_seq,
                event_type=event_type.value,
                source=source,
                payload=payload,
                payload_text=payload_text,
                journal_id=journal_id,
            )

            cur.execute(
                """
                INSERT INTO events (
                    event_id, conversation_id, correlation_id, conversation_seq,
                    event_type, source, payload, payload_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING event_id, conversation_id, correlation_id, global_seq,
                          conversation_seq, event_type, source, created_at,
                          payload, schema_version
                """,
                (
                    event_id,
                    conversation_id,
                    correlation_id,
                    conversation_seq,
                    event_type.value,
                    source,
                    Json(payload),
                    payload_text,
                ),
            )
            inserted = cur.fetchone()

            cur.execute(
                "UPDATE conversations SET next_event_seq = next_event_seq + 1 WHERE conversation_id = %s",
                (conversation_id,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    stored = _row_to_event(require_row(inserted, context="inserted event"))
    # If this write fails, the DB event is already safe and a deterministic retry
    # will repair the commit artifact.  We still surface the failure fail-closed.
    event_artifact_store.write_event_commit(
        event_id=stored.event_id,
        global_seq=stored.global_seq,
        conversation_seq=stored.conversation_seq,
        created_at=stored.created_at,
        schema_version=stored.schema_version,
        conversation_id=stored.conversation_id,
        correlation_id=stored.correlation_id,
    )
    return stored


def get_event_by_id(conn: psycopg.Connection, event_id: uuid.UUID) -> Event | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT event_id, conversation_id, correlation_id, global_seq,
                   conversation_seq, event_type, source, created_at,
                   payload, schema_version
            FROM events WHERE event_id = %s
            """,
            (event_id,),
        )
        row = cur.fetchone()
    return _row_to_event(row) if row else None


def get_events_by_conversation(conn: psycopg.Connection, conversation_id: uuid.UUID) -> list[Event]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT event_id, conversation_id, correlation_id, global_seq,
                   conversation_seq, event_type, source, created_at,
                   payload, schema_version
            FROM events WHERE conversation_id = %s
            ORDER BY conversation_seq ASC
            """,
            (conversation_id,),
        )
        rows = cur.fetchall()
    return [_row_to_event(r) for r in rows]


def get_first_event_in_conversation(conn: psycopg.Connection, conversation_id: uuid.UUID) -> Event | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT event_id, conversation_id, correlation_id, global_seq,
                   conversation_seq, event_type, source, created_at,
                   payload, schema_version
            FROM events WHERE conversation_id = %s
            ORDER BY conversation_seq ASC LIMIT 1
            """,
            (conversation_id,),
        )
        row = cur.fetchone()
    return _row_to_event(row) if row else None


def _row_to_event(row: dict[str, Any]) -> Event:
    return Event(
        event_id=row["event_id"],
        conversation_id=row["conversation_id"],
        correlation_id=row["correlation_id"],
        global_seq=row["global_seq"],
        conversation_seq=row["conversation_seq"],
        event_type=EventType(row["event_type"]),
        source=row["source"],
        created_at=row["created_at"],
        payload=row["payload"],
        schema_version=row["schema_version"],
    )
