"""Retrieval Service: decides *how* to satisfy a RetrievalRequest.

The Primary Agent only describes the information it needs (RetrievalRequest);
this module decides which underlying technique(s) to use. Today: structured
filters + Postgres full-text search. Later: + vector similarity + reranking,
behind this same request/response shape.

Known v1 limitation: full-text search only matches shared vocabulary. Fully
paraphrased recall with no lexical overlap needs future embedding-based
semantic retrieval (pgvector) -- not implemented here.
"""
from __future__ import annotations

import re
import uuid
from typing import LiteralString, cast

import psycopg
from psycopg.rows import dict_row

from prometheist.models import EventType, RetrievalItem, RetrievalRequest, RetrievalResult

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _or_tsquery(text: str) -> str | None:
    """Builds an OR-of-words tsquery instead of plainto_tsquery's implicit
    AND, so a paraphrased query_text (e.g. LLM-generated, with extra words
    not in the original text) can still match on shared vocabulary.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return None
    return " | ".join(words)


def search(
    conn: psycopg.Connection,
    conversation_id: uuid.UUID,
    request: RetrievalRequest,
    before_conversation_seq: int | None = None,
    before_global_seq: int | None = None,
) -> RetrievalResult:
    """`conversation_seq` only orders events within one conversation; it has
    no meaning across conversations. So the current-turn exclusion cutoff and
    the ordering column both depend on scope: CURRENT_CONVERSATION uses
    `conversation_seq`/`before_conversation_seq`; ALL_CONVERSATIONS uses the
    globally-comparable `global_seq`/`before_global_seq`.
    """
    where_clauses: list[str] = []
    where_params: list[object] = []
    scope_is_current = request.conversation_scope == "CURRENT_CONVERSATION"

    if scope_is_current:
        where_clauses.append("conversation_id = %s")
        where_params.append(conversation_id)
        if before_conversation_seq is not None:
            where_clauses.append("conversation_seq < %s")
            where_params.append(before_conversation_seq)
    else:
        if before_global_seq is not None:
            where_clauses.append("global_seq < %s")
            where_params.append(before_global_seq)

    if request.source_types:
        where_clauses.append("event_type = ANY(%s)")
        where_params.append([t.value for t in request.source_types])

    rank_select = "NULL::float AS rank"
    rank_params: list[object] = []
    tsquery_str = _or_tsquery(request.query_text) if request.query_text else None
    if tsquery_str:
        rank_select = (
            "ts_rank(to_tsvector('english', coalesce(payload_text, '')), "
            "to_tsquery('english', %s)) AS rank"
        )
        rank_params = [tsquery_str]
        where_clauses.append(
            "to_tsvector('english', coalesce(payload_text, '')) @@ to_tsquery('english', %s)"
        )
        where_params.append(tsquery_str)

    order_column = "conversation_seq" if scope_is_current else "global_seq"
    if request.ordering == "CHRON_ASC":
        order_by = f"{order_column} ASC"
    elif request.ordering == "RELEVANCE" and tsquery_str:
        order_by = f"rank DESC, {order_column} DESC"
    else:
        order_by = f"{order_column} DESC"

    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

    sql = f"""
        SELECT event_id, event_type, created_at, conversation_id, conversation_seq,
               global_seq, payload, payload_text, {rank_select}
        FROM events
        WHERE {where_sql}
        ORDER BY {order_by}
        LIMIT %s
    """
    params = rank_params + where_params + [request.limit]

    with conn.cursor(row_factory=dict_row) as cur:
        # Every interpolated fragment above is a module-local literal; all caller
        # values remain bound parameters.
        cur.execute(cast(LiteralString, sql), params)
        rows = cur.fetchall()

    items = [
        RetrievalItem(
            source_event_id=row["event_id"],
            event_type=EventType(row["event_type"]),
            created_at=row["created_at"],
            conversation_id=row["conversation_id"],
            conversation_seq=row["conversation_seq"],
            global_seq=row["global_seq"],
            content=row["payload_text"] or _payload_to_text(row["payload"]),
            rank=row["rank"],
        )
        for row in rows
    ]
    return RetrievalResult(items=items)


def _payload_to_text(payload: dict) -> str:
    return payload.get("text", str(payload))
