"""PostgreSQL persistence and bounded loading for association projections."""
from __future__ import annotations

from typing import Iterable, Sequence
import uuid

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist.association_feature_projection import AssociationFeatureProjection
from prometheist.association_projection import (
    ASSOCIATION_PROJECTION_VERSION,
    derive_associations,
)
from prometheist.associative_memory import Association
from prometheist.memory_kernel import MemoryEvent, tokenize
from prometheist.memory_projection import build_projection


def rebuild_associations(
    conn: psycopg.Connection,
    events: Iterable[MemoryEvent],
) -> tuple[Association, ...]:
    """Replace disposable association rows and predecessor features deterministically."""

    ordered_events = tuple(events)
    associations = derive_associations(ordered_events)
    feature_projection = AssociationFeatureProjection()
    feature_entries = build_projection(ordered_events, feature_projection)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memory_association_entries")
        cur.execute(
            """
            DELETE FROM memory_projection_entries
            WHERE projection_name = %s AND projection_version = %s
            """,
            (feature_projection.name, feature_projection.version),
        )
        if feature_entries:
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
                    for entry in feature_entries
                ],
            )
        if associations:
            cur.executemany(
                """
                INSERT INTO memory_association_entries (
                    association_id, projection_version, source_kind, source,
                    target_kind, target, relationship, strength,
                    provenance_event_ids, required_cue_terms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        association.association_id,
                        ASSOCIATION_PROJECTION_VERSION,
                        association.source_kind,
                        association.source,
                        association.target_kind,
                        association.target,
                        association.relationship,
                        association.strength,
                        Json(list(association.provenance_event_ids)),
                        Json(list(association.required_cue_terms)),
                    )
                    for association in associations
                ],
            )
    return associations


def _association_from_row(row: dict) -> Association:
    return Association(
        association_id=row["association_id"],
        source_kind=row["source_kind"],
        source=row["source"],
        target_kind=row["target_kind"],
        target=row["target"],
        relationship=row["relationship"],
        strength=float(row["strength"]),
        provenance_event_ids=tuple(row["provenance_event_ids"] or ()),
        required_cue_terms=tuple(row["required_cue_terms"] or ()),
    )


def load_associations(conn: psycopg.Connection) -> tuple[Association, ...]:
    """Load the current derived association projection in stable ID order."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT association_id, source_kind, source, target_kind, target,
                   relationship, strength, provenance_event_ids,
                   required_cue_terms
            FROM memory_association_entries
            WHERE projection_version = %s
            ORDER BY association_id ASC
            """,
            (ASSOCIATION_PROJECTION_VERSION,),
        )
        return tuple(_association_from_row(row) for row in cur.fetchall())


def load_reachable_associations(
    conn: psycopg.Connection,
    *,
    cue_terms: Sequence[str],
    source_event_ids: Sequence[str] = (),
    before_global_seq: int | None = None,
    max_hops: int = 2,
    association_limit: int = 250,
) -> tuple[Association, ...]:
    """Load only associations reachable from the current activation frontier.

    This is the PostgreSQL counterpart to bounded spreading activation. Query
    terms and directly activated event IDs form the first frontier. Each hop
    loads only edges whose source node is already active, then exposes target
    nodes to the next hop. Relationship-specific ``required_cue_terms`` are
    enforced before an edge is admitted.

    When ``before_global_seq`` is supplied, every EVENT endpoint of an admitted
    association must also precede that boundary. This prevents a projection
    rebuilt from the full ledger from routing a historical/current-turn query
    through evidence that did not yet exist at the requested point in history.

    The loader never scans or returns authoritative event content. It only
    selects disposable routing hints; callers fetch canonical events separately.
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    if association_limit < 1:
        raise ValueError("association_limit must be >= 1")

    normalized_cue_terms = {
        token
        for value in cue_terms
        for token in tokenize(value)
    }
    frontier_terms = set(normalized_cue_terms)
    frontier_events = {str(event_id) for event_id in source_event_ids if str(event_id)}
    seen_nodes: set[tuple[str, str]] = set()
    selected: dict[str, Association] = {}

    cutoff_sql = ""
    cutoff_params: tuple[object, ...] = ()
    if before_global_seq is not None:
        # EVENT endpoints are stored as text because association nodes are typed
        # TERM/EVENT. CASE ensures UUID casting is attempted only for EVENT rows.
        cutoff_sql = """
                  AND (
                    a.source_kind <> 'EVENT'
                    OR EXISTS (
                        SELECT 1
                        FROM events source_event
                        WHERE source_event.event_id = CASE
                            WHEN a.source_kind = 'EVENT' THEN a.source::uuid
                            ELSE NULL
                        END
                          AND source_event.global_seq < %s
                    )
                  )
                  AND (
                    a.target_kind <> 'EVENT'
                    OR EXISTS (
                        SELECT 1
                        FROM events target_event
                        WHERE target_event.event_id = CASE
                            WHEN a.target_kind = 'EVENT' THEN a.target::uuid
                            ELSE NULL
                        END
                          AND target_event.global_seq < %s
                    )
                  )
        """
        cutoff_params = (before_global_seq, before_global_seq)

    with conn.cursor(row_factory=dict_row) as cur:
        for _hop in range(max_hops):
            frontier_terms = {
                value
                for value in frontier_terms
                if ("TERM", value) not in seen_nodes
            }
            frontier_events = {
                value
                for value in frontier_events
                if ("EVENT", value) not in seen_nodes
            }
            if not frontier_terms and not frontier_events:
                break

            seen_nodes.update(("TERM", value) for value in frontier_terms)
            seen_nodes.update(("EVENT", value) for value in frontier_events)
            remaining = association_limit - len(selected)
            if remaining <= 0:
                break

            cur.execute(
                f"""
                SELECT a.association_id, a.source_kind, a.source,
                       a.target_kind, a.target, a.relationship, a.strength,
                       a.provenance_event_ids, a.required_cue_terms
                FROM memory_association_entries AS a
                WHERE a.projection_version = %s
                  AND (
                    (a.source_kind = 'TERM' AND a.source = ANY(%s::text[]))
                    OR (a.source_kind = 'EVENT' AND a.source = ANY(%s::text[]))
                  )
                  {cutoff_sql}
                ORDER BY a.association_id ASC
                LIMIT %s
                """,
                (
                    ASSOCIATION_PROJECTION_VERSION,
                    sorted(frontier_terms),
                    sorted(frontier_events),
                    *cutoff_params,
                    remaining,
                ),
            )

            next_terms: set[str] = set()
            next_events: set[str] = set()
            for row in cur.fetchall():
                association = _association_from_row(row)
                required = {
                    token
                    for value in association.required_cue_terms
                    for token in tokenize(value)
                }
                if required and not required.issubset(normalized_cue_terms):
                    continue
                selected[association.association_id] = association
                if association.target_kind == "TERM":
                    next_terms.add(association.target)
                else:
                    next_events.add(association.target)

            frontier_terms = next_terms
            frontier_events = next_events

    return tuple(selected[key] for key in sorted(selected))
