"""Use learned self representations as bounded pointers to canonical evidence.

A candidate hypothesis can help locate a source without becoming an established
belief. Only the original permitted source text enters the returned packet.
"""
from __future__ import annotations

from itertools import zip_longest
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from prometheist.memory_kernel import tokenize
from prometheist.models import EventType, MemoryEvidence, MemoryPacket

SELF_ROOT_NAVIGATION_VERSION = "self-memory-canonical-navigation-v1"
# Provisional; bounded independently of how many representations have accumulated.
SELF_ROOT_CANDIDATE_LIMIT = 8


def expand_self_memory_roots(
    conn: psycopg.Connection,
    packet: MemoryPacket,
    *,
    query_text: str,
    source_types: list[EventType],
    before_global_seq: int,
    item_limit: int,
    memory_request_id: UUID,
) -> MemoryPacket:
    """Search learned statements/tags and linked sources, then follow source links.

    Source policy and the exclusive history cutoff apply before truncation.
    Representation, resolution and link records must also predate the cutoff.
    Supporting and opposing roots are eligible; this operation never resolves a
    hypothesis, selects an answer, or admits a derived statement as canonical.
    """
    if item_limit < 1 or len(packet.items) > item_limit:
        raise ValueError("self-root navigation requires a valid bounded packet")
    result = packet.model_copy(deep=True)
    # Compound words contribute independent cues. Otherwise web-search parsing
    # can require the whole compound even when a learned tag names one component.
    terms = sorted(set(tokenize(query_text.replace("-", " ").replace("_", " "))))
    if not source_types or not terms or len(result.items) == item_limit:
        return result
    params = {
        "query": " OR ".join(f'"{term}"' for term in terms), "cutoff": before_global_seq,
        "types": [value.value for value in source_types],
        "candidate_limit": SELF_ROOT_CANDIDATE_LIMIT, "root_limit": item_limit,
    }
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT r.record_key AS representation_id, r.event_id AS representation_event_id,
                   s.event_id AS resolution_event_id, s.payload->>'status' AS status,
                   greatest(roots.rank, ts_rank(to_tsvector('english',
                       coalesce(r.payload->>'statement', '') || ' ' ||
                       replace(coalesce((r.payload->'context_tags')::text, ''), '_', ' ')),
                       websearch_to_tsquery('english', %(query)s))) AS rank
            FROM cognitive_heads r
            JOIN cognitive_heads s ON s.record_kind = 'self_resolution'
                                  AND s.record_key = r.record_key
            CROSS JOIN LATERAL (
                SELECT max(ts_rank(to_tsvector('english',
                           coalesce(nullif(e.payload_text, ''), e.payload->>'text', '')),
                           websearch_to_tsquery('english', %(query)s))) AS rank
                FROM cognitive_heads link
                JOIN events e ON e.event_id = (link.payload->>'root_event_id')::uuid
                WHERE link.record_kind = 'self_evidence'
                  AND link.payload->>'representation_id' = r.record_key
                  AND link.payload->>'relation' IN ('SUPPORTS', 'OPPOSES')
                  AND link.global_seq < %(cutoff)s AND e.global_seq < %(cutoff)s
                  AND e.event_type = ANY(%(types)s)
                  AND coalesce(nullif(e.payload_text, ''), e.payload->>'text', '') <> ''
            ) roots
            WHERE r.record_kind = 'self_representation'
              AND r.payload->>'subject' = 'self'
              AND r.global_seq < %(cutoff)s AND s.global_seq < %(cutoff)s
              AND s.payload->>'status' IN ('CANDIDATE', 'ESTABLISHED', 'CONTESTED')
              AND roots.rank IS NOT NULL
              AND (roots.rank > 0 OR to_tsvector('english',
                    coalesce(r.payload->>'statement', '') || ' ' ||
                    replace(coalesce((r.payload->'context_tags')::text, ''), '_', ' '))
                  @@ websearch_to_tsquery('english', %(query)s))
            ORDER BY rank DESC, r.record_key
            LIMIT %(candidate_limit)s
            """,
            params,
        )
        candidates = cur.fetchall()
        root_groups = []
        for candidate in candidates:
            cur.execute(
                """
                SELECT e.*, link.event_id AS link_event_id,
                       link.payload->>'relation' AS relation
                FROM cognitive_heads link
                JOIN events e ON e.event_id = (link.payload->>'root_event_id')::uuid
                WHERE link.record_kind = 'self_evidence'
                  AND link.payload->>'representation_id' = %(representation_id)s
                  AND link.payload->>'relation' IN ('SUPPORTS', 'OPPOSES')
                  AND link.global_seq < %(cutoff)s AND e.global_seq < %(cutoff)s
                  AND e.event_type = ANY(%(types)s)
                  AND coalesce(nullif(e.payload_text, ''), e.payload->>'text', '') <> ''
                ORDER BY e.global_seq DESC, e.event_id, link.record_key
                LIMIT %(root_limit)s
                """,
                {**params, "representation_id": candidate["representation_id"]},
            )
            root_groups.append([(candidate, root) for root in cur.fetchall()])

    seen = {item.source_event_id for item in result.items}
    routes = []
    # Take a root from each matched representation before taking a second root
    # from any one representation. A prolific schema cannot consume every slot.
    for row in zip_longest(*root_groups):
        for pair in row:
            if pair is None:
                continue
            candidate, root = pair
            routes.append({
                "representation_id": candidate["representation_id"],
                "status": candidate["status"], "event_id": str(root["event_id"]),
                "link_event_id": str(root["link_event_id"]), "relation": root["relation"],
            })
            if root["event_id"] in seen:
                continue
            seen.add(root["event_id"])
            result.items.append(MemoryEvidence(
                source_event_id=root["event_id"], event_type=EventType(root["event_type"]),
                source=root["source"], created_at=root["created_at"],
                conversation_id=root["conversation_id"], conversation_seq=root["conversation_seq"],
                global_seq=root["global_seq"], content=root["payload_text"] or root["payload"]["text"],
                retrieval_reasons=["SELF_MEMORY_ROOT", root["relation"]],
                provenance_event_ids=[candidate["representation_event_id"],
                                      candidate["resolution_event_id"], root["link_event_id"]],
            ))
            if len(result.items) == item_limit:
                break
        if len(result.items) == item_limit:
            break
    result.memory_request_id = memory_request_id
    result.need.limit = item_limit
    result.supported = bool(result.items)
    result.retrieval_trace = {
        "composition": SELF_ROOT_NAVIGATION_VERSION,
        "query_text": query_text, "candidate_limit": SELF_ROOT_CANDIDATE_LIMIT,
        "root_limit": item_limit, "before_global_seq": before_global_seq,
        "source_types": params["types"],
        "candidate_representation_ids": [item["representation_id"] for item in candidates],
        "root_routes": routes, "base_retrieval_trace": packet.retrieval_trace,
    }
    return result
