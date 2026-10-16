"""Bounded local canonical context, expanded once during fixed retrieval.

Conversation ordering is a provenance relation, not a semantic episode label.
Neighbors remain attributed evidence; proximity never asserts relevance or truth.
"""
from __future__ import annotations

from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from persistent_cognition.models import EventType, MemoryEvidence, MemoryPacket

NEIGHBORHOOD_VERSION = "canonical-local-context-v1"
# Provisional empirical tunable: bound index work even across internal event noise.
CANONICAL_NEIGHBOR_SEQUENCE_WINDOW = 32


def expand_canonical_neighbors(
    conn: psycopg.Connection,
    packet: MemoryPacket,
    *,
    source_types: list[EventType],
    before_global_seq: int,
    item_limit: int,
    memory_request_id: UUID,
) -> MemoryPacket:
    """Preserve seeds, then append their nearest allowed predecessor/successor.

    Only original packet items seed this one-hop operation. Both endpoints must
    precede the caller's history boundary and satisfy its source policy. The
    sequence window bounds rows examined through idx_events_conversation_seq;
    this never scans a whole conversation or treats it as a global recall wall.
    The returned packet and trace are persisted by the retrieval stage.
    """
    if item_limit < 1:
        raise ValueError("item_limit must be positive")
    if len(packet.items) > item_limit:
        raise ValueError("initial packet exceeds the response memory item limit")
    result = packet.model_copy(deep=True)
    if not packet.items or not source_types or len(packet.items) == item_limit:
        return result

    allowed = [event_type.value for event_type in source_types]
    seen = {item.source_event_id for item in packet.items}
    routes: list[dict[str, str]] = []
    for seed in packet.items:
        if seed.event_type not in source_types or seed.global_seq >= before_global_seq:
            continue
        # Rehydrate the seed in SQL: do not trust copied conversation/sequence
        # metadata to navigate. The two indexed range scans each return at most
        # one permitted neighbor, skipping internal machinery within the window.
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT neighbor.*, seed.event_id AS seed_event_id
                FROM events AS seed
                CROSS JOIN LATERAL (
                    (SELECT e.*, 'PREVIOUS' AS direction
                     FROM events AS e
                     WHERE e.conversation_id = seed.conversation_id
                       AND e.conversation_seq < seed.conversation_seq
                       AND e.conversation_seq >= seed.conversation_seq - %(window)s
                       AND e.global_seq < %(cutoff)s
                       AND e.event_type = ANY(%(types)s)
                     ORDER BY e.conversation_seq DESC, e.event_id
                     LIMIT 1)
                    UNION ALL
                    (SELECT e.*, 'NEXT' AS direction
                     FROM events AS e
                     WHERE e.conversation_id = seed.conversation_id
                       AND e.conversation_seq > seed.conversation_seq
                       AND e.conversation_seq <= seed.conversation_seq + %(window)s
                       AND e.global_seq < %(cutoff)s
                       AND e.event_type = ANY(%(types)s)
                     ORDER BY e.conversation_seq ASC, e.event_id
                     LIMIT 1)
                ) AS neighbor
                WHERE seed.event_id = %(seed)s
                  AND seed.global_seq < %(cutoff)s
                  AND seed.event_type = ANY(%(types)s)
                ORDER BY ABS(neighbor.conversation_seq - seed.conversation_seq),
                         neighbor.conversation_seq, neighbor.event_id
                """,
                {
                    "seed": seed.source_event_id,
                    "cutoff": before_global_seq,
                    "types": allowed,
                    "window": CANONICAL_NEIGHBOR_SEQUENCE_WINDOW,
                },
            )
            rows = cur.fetchall()
        for row in rows:
            event_id = row["event_id"]
            if event_id in seen:
                continue
            content = row["payload_text"] or (row["payload"] or {}).get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            seen.add(event_id)
            result.items.append(
                MemoryEvidence(
                    source_event_id=event_id,
                    event_type=EventType(row["event_type"]),
                    source=row["source"],
                    created_at=row["created_at"],
                    conversation_id=row["conversation_id"],
                    conversation_seq=row["conversation_seq"],
                    global_seq=row["global_seq"],
                    content=content,
                    score=None,
                    retrieval_reasons=["CANONICAL_NEIGHBOR", row["direction"]],
                    provenance_event_ids=[seed.source_event_id],
                )
            )
            routes.append({
                "seed_event_id": str(seed.source_event_id),
                "event_id": str(event_id),
                "direction": row["direction"],
            })
            if len(result.items) == item_limit:
                break
        if len(result.items) == item_limit:
            break

    result.memory_request_id = memory_request_id
    result.need.limit = item_limit
    result.supported = bool(result.items)
    result.retrieval_trace = {
        "composition": NEIGHBORHOOD_VERSION,
        "base_memory_request_id": str(packet.memory_request_id),
        "base_retrieval_trace": result.retrieval_trace,
        "sequence_window": CANONICAL_NEIGHBOR_SEQUENCE_WINDOW,
        "item_limit": item_limit,
        "before_global_seq": before_global_seq,
        "source_types": allowed,
        "neighbor_routes": routes,
    }
    return result
