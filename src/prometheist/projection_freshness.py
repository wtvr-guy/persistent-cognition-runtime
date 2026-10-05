"""Incremental freshness helpers for disposable memory projections.

Steady-state recall must not establish freshness by materializing the lifetime
canonical ledger. A pair of per-event projections acts as the committed
high-water mark: lexical evidence and association features are written in the
same transaction as any new association edges. If their high-water marks ever
disagree, callers must rebuild disposable state rather than guess across a gap.
"""
from __future__ import annotations

import json
from typing import Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist.association_feature_projection import AssociationFeatureProjection
from prometheist.association_projection import derive_associations
from prometheist.associative_memory import Association
from prometheist.db import require_row
from prometheist.memory_kernel import MemoryEvent
from prometheist.memory_projection import LexicalProjection


def _row_to_memory_event(row: dict) -> MemoryEvent:
    payload = row["payload"] or {}
    text = row.get("payload_text") or payload.get("text") or json.dumps(
        payload,
        sort_keys=True,
        default=str,
    )
    return MemoryEvent(
        event_id=str(row["event_id"]),
        global_seq=int(row["global_seq"]),
        conversation_id=str(row["conversation_id"]),
        conversation_seq=int(row["conversation_seq"]),
        event_type=str(row["event_type"]),
        source=str(row["source"]),
        created_at=row["created_at"],
        text=text,
        payload=payload,
    )


def canonical_high_water(
    conn: psycopg.Connection,
    *,
    before_global_seq: int | None,
) -> int:
    cutoff_sql = "WHERE global_seq < %s" if before_global_seq is not None else ""
    params: tuple[object, ...] = (before_global_seq,) if before_global_seq is not None else ()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COALESCE(MAX(global_seq), 0) FROM events {cutoff_sql}",
            params,
        )
        return int(require_row(cur.fetchone(), context="canonical high-water mark")[0])


def projection_high_water(
    conn: psycopg.Connection,
    *,
    projection_name: str,
    projection_version: str,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(source_global_seq), 0)
            FROM memory_projection_entries
            WHERE projection_name = %s AND projection_version = %s
            """,
            (projection_name, projection_version),
        )
        return int(require_row(cur.fetchone(), context="projection high-water mark")[0])


def committed_projection_high_waters(conn: psycopg.Connection) -> tuple[int, int]:
    lexical = LexicalProjection()
    features = AssociationFeatureProjection()
    return (
        projection_high_water(
            conn,
            projection_name=lexical.name,
            projection_version=lexical.version,
        ),
        projection_high_water(
            conn,
            projection_name=features.name,
            projection_version=features.version,
        ),
    )


def load_event_delta(
    conn: psycopg.Connection,
    *,
    after_global_seq: int,
    before_global_seq: int | None,
) -> tuple[MemoryEvent, ...]:
    cutoff_sql = "AND global_seq < %s" if before_global_seq is not None else ""
    params: tuple[object, ...] = (
        (after_global_seq, before_global_seq)
        if before_global_seq is not None
        else (after_global_seq,)
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT event_id, global_seq, conversation_id, conversation_seq,
                   event_type, source, created_at, payload, payload_text
            FROM events
            WHERE global_seq > %s
              {cutoff_sql}
            ORDER BY global_seq ASC, event_id ASC
            """,
            params,
        )
        return tuple(_row_to_memory_event(row) for row in cur.fetchall())


def _load_previous_state_event(
    conn: psycopg.Connection,
    *,
    current: MemoryEvent,
    concept: str,
) -> MemoryEvent | None:
    features = AssociationFeatureProjection()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT e.event_id, e.global_seq, e.conversation_id,
                   e.conversation_seq, e.event_type, e.source, e.created_at,
                   e.payload, e.payload_text
            FROM memory_projection_entries AS p
            JOIN events AS e ON e.event_id = p.source_event_id
            WHERE p.projection_name = %s
              AND p.projection_version = %s
              AND p.source_global_seq < %s
              AND p.data @> %s::jsonb
            ORDER BY p.source_global_seq DESC, e.event_id ASC
            LIMIT 1
            """,
            (
                features.name,
                features.version,
                current.global_seq,
                Json({"source": current.source, "concepts": [concept]}),
            ),
        )
        row = cur.fetchone()
    return _row_to_memory_event(row) if row is not None else None


def _load_previous_unresolved_event(
    conn: psycopg.Connection,
    *,
    current: MemoryEvent,
    entity_terms: tuple[str, ...],
) -> MemoryEvent | None:
    if not entity_terms:
        return None
    features = AssociationFeatureProjection()
    overlap_sql = " OR ".join("p.data @> %s::jsonb" for _ in entity_terms)
    overlap_params = [
        Json({"is_unresolved": True, "entity_terms": [term]})
        for term in entity_terms
    ]
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT e.event_id, e.global_seq, e.conversation_id,
                   e.conversation_seq, e.event_type, e.source, e.created_at,
                   e.payload, e.payload_text
            FROM memory_projection_entries AS p
            JOIN events AS e ON e.event_id = p.source_event_id
            WHERE p.projection_name = %s
              AND p.projection_version = %s
              AND p.source_global_seq < %s
              AND ({overlap_sql})
            ORDER BY p.source_global_seq DESC, e.event_id ASC
            LIMIT 1
            """,
            [
                features.name,
                features.version,
                current.global_seq,
                *overlap_params,
            ],
        )
        row = cur.fetchone()
    return _row_to_memory_event(row) if row is not None else None


def derive_incremental_associations(
    conn: psycopg.Connection,
    events: Iterable[MemoryEvent],
) -> tuple[Association, ...]:
    """Derive only edges whose later/source event is in the supplied delta."""

    feature_projection = AssociationFeatureProjection()
    derived: dict[str, Association] = {}

    for event in sorted(events, key=lambda item: (item.global_seq, item.event_id)):
        features = feature_projection.project(event)

        for association in derive_associations((event,)):
            derived[association.association_id] = association

        if bool(features["is_change"]):
            for concept in features["concepts"]:
                previous = _load_previous_state_event(
                    conn,
                    current=event,
                    concept=str(concept),
                )
                if previous is None:
                    continue
                for association in derive_associations((previous, event)):
                    if (
                        association.relationship == "PREVIOUS_STATE"
                        and association.source == event.event_id
                    ):
                        derived[association.association_id] = association

        if bool(features["is_resolved"]):
            entity_terms = tuple(str(value) for value in features["entity_terms"])
            previous = _load_previous_unresolved_event(
                conn,
                current=event,
                entity_terms=entity_terms,
            )
            if previous is not None:
                for association in derive_associations((previous, event)):
                    if (
                        association.relationship == "RESOLVED_BY"
                        and association.target == event.event_id
                    ):
                        derived[association.association_id] = association

    return tuple(sorted(derived.values(), key=lambda item: item.association_id))
