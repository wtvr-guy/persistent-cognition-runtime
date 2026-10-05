"""Immutable cognitive records and an atomically maintained, rebuildable head index.

The PostgreSQL trigger updates heads in the canonical event transaction. All
records still pass through event_store's independent, fsync'd artifact journal.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from uuid import UUID, uuid5

import psycopg

from prometheist import event_store
from prometheist.models import EventType

COGNITIVE_NAMESPACE = UUID("9782441e-7c35-579b-a480-2b198c54a31b")
SYSTEM_CONVERSATION = uuid5(COGNITIVE_NAMESPACE, "system-provenance")
HEAD_PAGE_SIZE = 64


@contextmanager
def record_lock(conn: psycopg.Connection, key: str) -> Iterator[None]:
    # Session locks survive event_store's commits. Never change the user's
    # transaction boundaries to make a transaction-scoped lock appear durable.
    conn.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (f"cognition:{key}",))
    conn.commit()
    try:
        yield
    finally:
        conn.rollback()
        conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (f"cognition:{key}",))
        conn.commit()


def get_record(conn: psycopg.Connection, kind: str, key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT payload FROM cognitive_heads WHERE record_kind = %s AND record_key = %s",
        (kind, key),
    ).fetchone()
    return dict(row[0]) if row else None


def put_record(
    conn: psycopg.Connection, kind: str, key: str, data: dict[str, Any], *, revision: str,
    conversation_id: UUID = SYSTEM_CONVERSATION, correlation_id: UUID = SYSTEM_CONVERSATION,
) -> UUID:
    event_store.start_conversation(conn, conversation_id)
    event_id = uuid5(COGNITIVE_NAMESPACE, f"{kind}:{key}:{revision}")
    event_store.record_event(
        conn, conversation_id=conversation_id, correlation_id=correlation_id,
        event_type=EventType.DERIVED_REPRESENTATION, source="cognitive_runtime",
        event_id=event_id,
        payload={"kind": "COGNITIVE_RECORD", "record_kind": kind, "record_key": key,
                 "revision": revision, "data": data, "epistemic_status": "DERIVED"},
    )
    return event_id


def list_records(
    conn: psycopg.Connection, kind: str, *, after_key: str = "", limit: int = HEAD_PAGE_SIZE,
) -> list[tuple[str, dict[str, Any]]]:
    if not 1 <= limit <= HEAD_PAGE_SIZE:
        raise ValueError("head page exceeds bounded read policy")
    rows = conn.execute(
        """SELECT record_key, payload FROM cognitive_heads
           WHERE record_kind = %s AND record_key > %s ORDER BY record_key LIMIT %s""",
        (kind, after_key, limit),
    ).fetchall()
    return [(str(row[0]), dict(row[1])) for row in rows]


def list_records_with_prefix(
    conn: psycopg.Connection,
    kind: str,
    prefix: str,
    *,
    after_key: str = "",
    limit: int = HEAD_PAGE_SIZE,
) -> list[tuple[str, dict[str, Any]]]:
    """Read one bounded page of current records whose keys share an exact prefix.

    This is an indexing primitive, not an epistemic boundary. Callers that need
    the complete state for one tightly scoped semantic key must continue paging
    until exhaustion rather than interpreting a storage page as evidence scope.
    """

    if not 1 <= limit <= HEAD_PAGE_SIZE:
        raise ValueError("head page exceeds bounded read policy")
    rows = conn.execute(
        """SELECT record_key, payload FROM cognitive_heads
           WHERE record_kind = %s AND starts_with(record_key, %s)
             AND record_key > %s
           ORDER BY record_key LIMIT %s""",
        (kind, prefix, after_key, limit),
    ).fetchall()
    return [(str(row[0]), dict(row[1])) for row in rows]


def record_history(conn: psycopg.Connection, kind: str, key: str) -> list[dict[str, Any]]:
    """Every revision ever written for one exact (kind, key), oldest first.

    ``get_record``/``list_records`` read the rebuildable ``cognitive_heads``
    index, which exposes only the current revision. A superseded revision is
    never deleted; it remains reconstructable directly from canonical
    ``events``. This is an intentionally exceptional forensic/audit read
    scoped to one exact key, not the ordinary bounded per-percept recall path.
    """
    rows = conn.execute(
        """SELECT payload FROM events
           WHERE source = 'cognitive_runtime' AND payload->>'kind' = 'COGNITIVE_RECORD'
             AND payload->>'record_kind' = %s AND payload->>'record_key' = %s
           ORDER BY global_seq ASC""",
        (kind, key),
    ).fetchall()
    return [dict(row[0]["data"]) for row in rows]


def rebuild_heads(conn: psycopg.Connection) -> None:
    """Explicit offline recovery only; ordinary cognition never scans history."""
    with record_lock(conn, "rebuild-heads"):
        conn.execute("LOCK TABLE events IN SHARE MODE")
        conn.execute("DELETE FROM cognitive_heads")
        conn.execute("""
            INSERT INTO cognitive_heads(record_kind, record_key, event_id, global_seq, payload)
            SELECT DISTINCT ON (payload->>'record_kind', payload->>'record_key')
                   payload->>'record_kind', payload->>'record_key', event_id, global_seq, payload->'data'
            FROM events WHERE source = 'cognitive_runtime' AND payload->>'kind' = 'COGNITIVE_RECORD'
            ORDER BY payload->>'record_kind', payload->>'record_key', global_seq DESC
        """)
        conn.commit()
