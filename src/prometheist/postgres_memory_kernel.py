"""PostgreSQL adapter for the deterministic Prometheist memory kernel.

This module is additive to the MVP Retrieval Service. It intentionally does
not replace ``prometheist.retrieval`` yet; the existing proof-of-concept remains
usable while the Memory Kernel evolves beside it. A later MAS milestone can
place the verified kernel behind the shared JIT-memory boundary.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, LiteralString, cast

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist.association_projection import (
    ASSOCIATION_PROJECTION_NAME,
    ASSOCIATION_PROJECTION_VERSION,
    association_projection_digest,
)
from prometheist.associative_memory import AssociativeMemoryPacket, associative_recall
from prometheist.memory_integrity import (
    IntegrityRecord,
    build_integrity_chain,
    verify_integrity_chain,
)
from prometheist.memory_kernel import (
    CueState,
    MemoryEvent,
    MemoryPacket,
    normalize_text,
    recall,
    score_event,
    tokenize,
)
from prometheist.memory_projection import LexicalProjection, build_projection, projection_digest
from prometheist.postgres_association_projection import (
    load_reachable_associations,
    rebuild_associations,
)

CANDIDATE_ROUTER_VERSION = "specificity-routes-v4"
LEXICAL_PROJECTION_NAME = LexicalProjection.name
LEXICAL_PROJECTION_VERSION = LexicalProjection.version


def _row_to_memory_event(row: dict) -> MemoryEvent:
    payload = row["payload"] or {}
    text = row.get("payload_text") or payload.get("text") or json.dumps(payload, sort_keys=True, default=str)
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


def load_events(conn: psycopg.Connection, *, before_global_seq: int | None = None) -> list[MemoryEvent]:
    where = "WHERE global_seq < %s" if before_global_seq is not None else ""
    params: tuple[object, ...] = (before_global_seq,) if before_global_seq is not None else ()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT event_id, global_seq, conversation_id, conversation_seq,
                   event_type, source, created_at, payload, payload_text
            FROM events
            {where}
            ORDER BY global_seq ASC
            """,
            params,
        )
        return [_row_to_memory_event(row) for row in cur.fetchall()]


def _load_events_by_ids(
    conn: psycopg.Connection,
    event_ids: list[uuid.UUID],
    *,
    before_global_seq: int | None = None,
) -> list[MemoryEvent]:
    if not event_ids:
        return []
    cutoff_sql = "AND global_seq < %s" if before_global_seq is not None else ""
    params: tuple[object, ...] = (
        (event_ids, before_global_seq)
        if before_global_seq is not None
        else (event_ids,)
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT event_id, global_seq, conversation_id, conversation_seq,
                   event_type, source, created_at, payload, payload_text
            FROM events
            WHERE event_id = ANY(%s)
              {cutoff_sql}
            """,
            params,
        )
        return [_row_to_memory_event(row) for row in cur.fetchall()]


def rebuild(conn: psycopg.Connection) -> dict[str, Any]:
    """Rebuild every disposable Memory Kernel structure from authoritative events."""
    events = load_events(conn)
    integrity = build_integrity_chain(events)
    projection = LexicalProjection()
    entries = build_projection(events, projection)
    digest = projection_digest(entries)
    associations = rebuild_associations(conn, events)
    association_digest = association_projection_digest(associations)
    run_id = uuid.uuid4()
    association_run_id = uuid.uuid4()

    with conn.cursor() as cur:
        # Derived state is explicitly disposable. Authoritative ``events`` is
        # never modified by this operation.
        cur.execute("DELETE FROM event_integrity")
        cur.execute(
            "DELETE FROM memory_projection_entries WHERE projection_name = %s AND projection_version = %s",
            (projection.name, projection.version),
        )

        if integrity:
            cur.executemany(
                """
                INSERT INTO event_integrity (
                    event_id, global_seq, previous_hash, content_hash, hash_algorithm
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        uuid.UUID(record.event_id),
                        record.global_seq,
                        record.previous_hash,
                        record.content_hash,
                        record.algorithm,
                    )
                    for record in integrity
                ],
            )

        if entries:
            cur.executemany(
                """
                INSERT INTO memory_projection_entries (
                    projection_name, projection_version, source_event_id,
                    source_global_seq, source_hash, data
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        entry.projection_name,
                        entry.projection_version,
                        uuid.UUID(entry.source_event_id),
                        entry.source_global_seq,
                        entry.source_hash,
                        Json(dict(entry.data)),
                    )
                    for entry in entries
                ],
            )

        cur.execute(
            """
            INSERT INTO memory_projection_runs (
                run_id, projection_name, projection_version, source_event_count,
                entry_count, projection_digest, status, completed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, 'COMPLETED', now())
            """,
            (run_id, projection.name, projection.version, len(events), len(entries), digest),
        )
        cur.execute(
            """
            INSERT INTO memory_projection_runs (
                run_id, projection_name, projection_version, source_event_count,
                entry_count, projection_digest, status, completed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, 'COMPLETED', now())
            """,
            (
                association_run_id,
                ASSOCIATION_PROJECTION_NAME,
                ASSOCIATION_PROJECTION_VERSION,
                len(events),
                len(associations),
                association_digest,
            ),
        )
    conn.commit()
    return {
        "run_id": str(run_id),
        "association_run_id": str(association_run_id),
        "source_event_count": len(events),
        "integrity_record_count": len(integrity),
        "projection_entry_count": len(entries),
        "association_entry_count": len(associations),
        "projection_digest": digest,
        "association_projection_digest": association_digest,
    }


def verify(conn: psycopg.Connection):
    events = load_events(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT event_id, global_seq, previous_hash, content_hash, hash_algorithm
            FROM event_integrity
            ORDER BY global_seq ASC
            """
        )
        records = tuple(
            IntegrityRecord(
                event_id=str(row["event_id"]),
                global_seq=int(row["global_seq"]),
                previous_hash=row["previous_hash"],
                content_hash=row["content_hash"],
                algorithm=row["hash_algorithm"],
            )
            for row in cur.fetchall()
        )
    return verify_integrity_chain(events, records)


def _candidate_event_ids(
    conn: psycopg.Connection,
    cue: CueState,
    *,
    before_global_seq: int | None,
    candidate_limit: int,
) -> list[uuid.UUID]:
    """Build a bounded candidate union from independent deterministic routes.

    The router deliberately separates exact-entity, lexical-specificity,
    historical-anchor, and recency routes. Newest-only candidate composition
    can erase equally relevant old evidence before the pure kernel has a chance
    to preserve temporal alternatives, so content-routed recall reserves up to
    one output packet's worth of bounded historical anchors. The total candidate
    bound remains fixed; only its composition changes.

    Source-type restrictions are applied inside each route rather than after
    truncation so disallowed event types cannot consume the bounded window.
    """
    ignored_terms = {
        token
        for value in cue.ignored_terms
        for token in tokenize(value)
    }
    terms = sorted(
        {
            token
            for token in tokenize(cue.query_text or "")
            if token not in ignored_terms
        }
    )
    entity_terms = sorted(
        {
            normalized
            for entity in cue.entities
            if (normalized := normalize_text(entity))
        }
    )
    source_types = sorted(set(cue.source_types))
    cutoff_sql = "AND e.global_seq < %s" if before_global_seq is not None else ""
    cutoff_params: list[object] = [before_global_seq] if before_global_seq is not None else []
    source_type_sql = "AND e.event_type = ANY(%s::text[])" if source_types else ""
    source_type_params: list[object] = [source_types] if source_types else []

    # Preserve a small independent recency route, but never let it consume the
    # minimum space needed for the requested output when content cues exist.
    has_content_routes = bool(terms or entity_terms)
    if has_content_routes:
        recency_budget = min(
            100,
            candidate_limit // 10,
            max(0, candidate_limit - cue.limit),
        )
    else:
        recency_budget = candidate_limit
    content_budget = candidate_limit - recency_budget

    historical_budget = (
        min(cue.limit, max(0, content_budget - cue.limit)) if has_content_routes else 0
    )
    primary_content_budget = content_budget - historical_budget

    if terms and entity_terms:
        entity_budget = min(
            primary_content_budget,
            max(cue.limit, primary_content_budget // 4),
        )
        lexical_budget = primary_content_budget - entity_budget
    elif entity_terms:
        entity_budget = primary_content_budget
        lexical_budget = 0
    elif terms:
        entity_budget = 0
        lexical_budget = primary_content_budget
    else:
        entity_budget = 0
        lexical_budget = 0

    ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()

    def append_rows(rows, budget: int) -> None:
        added = 0
        for row in rows:
            event_id = row[0]
            if event_id in seen:
                continue
            seen.add(event_id)
            ids.append(event_id)
            added += 1
            if added >= budget:
                break

    with conn.cursor() as cur:
        if entity_budget and entity_terms:
            cur.execute(
                f"""
                SELECT e.event_id,
                       (
                           SELECT count(*)
                           FROM unnest(%s::text[]) AS q(entity)
                           WHERE (p.data -> 'entities') ? q.entity
                       ) AS entity_matches,
                       (
                           SELECT count(*)
                           FROM unnest(%s::text[]) AS q(term)
                           WHERE (p.data -> 'terms') ? q.term
                       ) AS term_matches
                FROM memory_projection_entries p
                JOIN events e ON e.event_id = p.source_event_id
                WHERE p.projection_name = %s
                  AND p.projection_version = %s
                  {cutoff_sql}
                  {source_type_sql}
                  AND (p.data -> 'entities') ?| %s::text[]
                ORDER BY entity_matches DESC, term_matches DESC,
                         e.global_seq DESC, e.event_id ASC
                LIMIT %s
                """,
                [entity_terms, terms, LEXICAL_PROJECTION_NAME, LEXICAL_PROJECTION_VERSION]
                + cutoff_params
                + source_type_params
                + [entity_terms, entity_budget],
            )
            append_rows(cur.fetchall(), entity_budget)

        if lexical_budget and terms:
            # Fetch enough lexical rows to compensate for overlap with the
            # entity route while keeping SQL work bounded by candidate_limit.
            lexical_fetch_limit = min(
                candidate_limit,
                lexical_budget + len(seen),
            )
            cur.execute(
                f"""
                SELECT e.event_id,
                       (
                           SELECT count(*)
                           FROM unnest(%s::text[]) AS q(term)
                           WHERE (p.data -> 'terms') ? q.term
                       ) AS term_matches,
                       (
                           SELECT count(*)
                           FROM unnest(%s::text[]) AS q(entity)
                           WHERE (p.data -> 'entities') ? q.entity
                       ) AS entity_matches
                FROM memory_projection_entries p
                JOIN events e ON e.event_id = p.source_event_id
                WHERE p.projection_name = %s
                  AND p.projection_version = %s
                  {cutoff_sql}
                  {source_type_sql}
                  AND (p.data -> 'terms') ?| %s::text[]
                ORDER BY term_matches DESC, entity_matches DESC,
                         e.global_seq DESC, e.event_id ASC
                LIMIT %s
                """,
                [terms, entity_terms, LEXICAL_PROJECTION_NAME, LEXICAL_PROJECTION_VERSION]
                + cutoff_params
                + source_type_params
                + [terms, lexical_fetch_limit],
            )
            append_rows(cur.fetchall(), lexical_budget)

        if historical_budget and terms:
            historical_fetch_limit = min(
                candidate_limit,
                historical_budget + len(seen),
            )
            cur.execute(
                f"""
                SELECT e.event_id,
                       (
                           SELECT count(*)
                           FROM unnest(%s::text[]) AS q(term)
                           WHERE (p.data -> 'terms') ? q.term
                       ) AS term_matches,
                       (
                           SELECT count(*)
                           FROM unnest(%s::text[]) AS q(entity)
                           WHERE (p.data -> 'entities') ? q.entity
                       ) AS entity_matches
                FROM memory_projection_entries p
                JOIN events e ON e.event_id = p.source_event_id
                WHERE p.projection_name = %s
                  AND p.projection_version = %s
                  {cutoff_sql}
                  {source_type_sql}
                  AND (p.data -> 'terms') ?| %s::text[]
                ORDER BY term_matches DESC, entity_matches DESC,
                         e.global_seq ASC, e.event_id ASC
                LIMIT %s
                """,
                [terms, entity_terms, LEXICAL_PROJECTION_NAME, LEXICAL_PROJECTION_VERSION]
                + cutoff_params
                + source_type_params
                + [terms, historical_fetch_limit],
            )
            append_rows(cur.fetchall(), historical_budget)
        elif historical_budget and entity_terms:
            historical_fetch_limit = min(
                candidate_limit,
                historical_budget + len(seen),
            )
            cur.execute(
                f"""
                SELECT e.event_id,
                       (
                           SELECT count(*)
                           FROM unnest(%s::text[]) AS q(entity)
                           WHERE (p.data -> 'entities') ? q.entity
                       ) AS entity_matches
                FROM memory_projection_entries p
                JOIN events e ON e.event_id = p.source_event_id
                WHERE p.projection_name = %s
                  AND p.projection_version = %s
                  {cutoff_sql}
                  {source_type_sql}
                  AND (p.data -> 'entities') ?| %s::text[]
                ORDER BY entity_matches DESC,
                         e.global_seq ASC, e.event_id ASC
                LIMIT %s
                """,
                [entity_terms, LEXICAL_PROJECTION_NAME, LEXICAL_PROJECTION_VERSION]
                + cutoff_params
                + source_type_params
                + [entity_terms, historical_fetch_limit],
            )
            append_rows(cur.fetchall(), historical_budget)

        if recency_budget:
            conditions: list[str] = []
            params: list[object] = []
            if before_global_seq is not None:
                conditions.append("global_seq < %s")
                params.append(before_global_seq)
            if source_types:
                conditions.append("event_type = ANY(%s::text[])")
                params.append(source_types)
            where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""
            recency_fetch_limit = min(
                candidate_limit,
                recency_budget + len(seen),
            )
            params.append(recency_fetch_limit)
            # ``where_sql`` is composed only of module-local literals; every caller
            # value stays a bound parameter.
            cur.execute(
                cast(
                    LiteralString,
                    f"""
                SELECT event_id FROM events
                {where_sql}
                ORDER BY global_seq DESC, event_id ASC
                LIMIT %s
                """,
                ),
                params,
            )
            append_rows(cur.fetchall(), recency_budget)

    return ids[:candidate_limit]


def recall_from_postgres(
    conn: psycopg.Connection,
    cue: CueState,
    *,
    before_global_seq: int | None = None,
    candidate_limit: int = 500,
) -> MemoryPacket:
    """Baseline indexed recall without association expansion."""
    if candidate_limit < cue.limit:
        raise ValueError("candidate_limit must be >= cue.limit")
    event_ids = _candidate_event_ids(
        conn,
        cue,
        before_global_seq=before_global_seq,
        candidate_limit=candidate_limit,
    )
    return recall(
        _load_events_by_ids(
            conn,
            event_ids,
            before_global_seq=before_global_seq,
        ),
        cue,
    )


def associative_recall_from_postgres(
    conn: psycopg.Connection,
    cue: CueState,
    *,
    before_global_seq: int | None = None,
    candidate_limit: int = 500,
    association_limit: int = 250,
    max_hops: int = 2,
    decay: float = 0.85,
    seed_event_ids: tuple[str, ...] = (),
) -> AssociativeMemoryPacket:
    """Indexed recall plus bounded traversal of persisted association routes.

    The candidate router supplies a bounded direct-candidate union. Direct
    candidates that clear the kernel minimum score become event-node sources;
    application-owned ``seed_event_ids`` are also first-class event-node sources
    even when their lexical score is low. Query/entity terms are the other source
    nodes. Reachable association targets are then fetched as canonical events and
    the pure associative kernel performs final deterministic ranking.

    The global-sequence boundary applies to direct candidates, explicit seeds,
    and every EVENT endpoint reached through the association projection. A
    full-history projection must never reveal evidence newer than the caller's
    cutoff.
    """
    if candidate_limit < cue.limit:
        raise ValueError("candidate_limit must be >= cue.limit")

    normalized_seed_ids = tuple(
        dict.fromkeys(value.strip() for value in seed_event_ids if value.strip())
    )
    seed_uuid_ids = [uuid.UUID(value) for value in normalized_seed_ids]
    candidate_ids = _candidate_event_ids(
        conn,
        cue,
        before_global_seq=before_global_seq,
        candidate_limit=candidate_limit,
    )
    all_direct_ids = list(dict.fromkeys([*candidate_ids, *seed_uuid_ids]))
    candidates = _load_events_by_ids(
        conn,
        all_direct_ids,
        before_global_seq=before_global_seq,
    )
    loaded_seed_ids = {event.event_id for event in candidates} & set(normalized_seed_ids)
    if loaded_seed_ids != set(normalized_seed_ids):
        raise ValueError("seed_event_ids must reference canonical events before the recall boundary")

    allowed_types = set(cue.source_types)
    directly_activated_ids = {
        event.event_id
        for event in candidates
        if (not allowed_types or event.event_type in allowed_types)
        and score_event(event, cue).total >= cue.minimum_score
    }
    directly_activated_ids.update(normalized_seed_ids)
    ignored = {
        token
        for value in cue.ignored_terms
        for token in tokenize(value)
    }
    cue_terms = tuple(
        sorted(
            {
                token
                for value in ((cue.query_text or ""), *cue.entities)
                for token in tokenize(value)
                if token not in ignored
            }
        )
    )
    associations = load_reachable_associations(
        conn,
        cue_terms=cue_terms,
        source_event_ids=sorted(directly_activated_ids),
        before_global_seq=before_global_seq,
        max_hops=max_hops,
        association_limit=association_limit,
    )

    referenced_ids: set[str] = set()
    for association in associations:
        if association.source_kind == "EVENT":
            referenced_ids.add(association.source)
        if association.target_kind == "EVENT":
            referenced_ids.add(association.target)

    loaded_ids = {event.event_id for event in candidates}
    extra_uuid_ids = [
        uuid.UUID(event_id)
        for event_id in sorted(referenced_ids - loaded_ids)
    ]
    events = candidates + _load_events_by_ids(
        conn,
        extra_uuid_ids,
        before_global_seq=before_global_seq,
    )

    return associative_recall(
        events,
        cue,
        associations,
        max_hops=max_hops,
        decay=decay,
        seed_event_ids=normalized_seed_ids,
    )
