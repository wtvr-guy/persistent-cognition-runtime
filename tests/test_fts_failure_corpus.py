"""Documented corpus of retrieval cases: which ones current (structured
filters + Postgres FTS) retrieval can and can't find. This is deliberately a
living document of *failures*, not just successes -- each xfail case is
concrete evidence for when pgvector/embedding-based semantic retrieval
becomes justified (spec: "the first prototype should prove complete event
persistence plus retrieval works before adding sophisticated memory
interpretation").

Run with: uv run pytest tests/test_fts_failure_corpus.py -v
"""
from __future__ import annotations

import uuid

import pytest

from prometheist import db, event_store, retrieval
from prometheist.models import EventType, RetrievalRequest


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _seed_and_search(conn, stored_text: str, query_text: str) -> bool:
    conversation_id = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()
    event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": stored_text},
        payload_text=stored_text,
    )
    result = retrieval.search(
        conn, conversation_id, RetrievalRequest(query_text=query_text, ordering="RELEVANCE", limit=5)
    )
    return any(stored_text in item.content for item in result.items)


# (stored_text, query_text, expected_to_be_found, note)
CASES = [
    (
        "The codename for Project Falcon is Blue Cedar.",
        "What codename did I give Project Falcon?",
        True,
        "shares 'codename' and 'Falcon' -- lexical overlap is enough for FTS",
    ),
    (
        "The codename for Project Falcon is Blue Cedar.",
        "What was the Falcon project codename?",
        True,
        "reordered but still shares 'Falcon' and 'codename'",
    ),
    (
        "The codename for Falcon is Blue Cedar.",
        "What was that tree-related name for the bird initiative?",
        False,
        "paraphrase with zero shared vocabulary (Falcon/bird, Blue Cedar/tree, "
        "codename/name) -- needs semantic (embedding-based) retrieval, not "
        "keyword FTS. NOTE: an earlier version of this case kept the word "
        "'Project' in both sentences, which accidentally matched via FTS -- "
        "even generic shared words are enough for OR-based FTS to hit.",
    ),
    (
        "Remember that the launch code for Project Oriole is X7Q-19-ALPHA.",
        "What launch code did I give for Oriole?",
        True,
        "shares 'launch code' and 'Oriole'",
    ),
    (
        "Remember that the launch code for Project Oriole is X7Q-19-ALPHA.",
        "What's the secret number for the bird-themed initiative?",
        False,
        "'secret number'/'launch code' and 'bird-themed initiative'/'Oriole' "
        "share no vocabulary -- another semantic-retrieval case",
    ),
]


@pytest.mark.parametrize("stored_text,query_text,expected_found,note", CASES)
def test_fts_recall_case(conn, stored_text, query_text, expected_found, note):
    found = _seed_and_search(conn, stored_text, query_text)
    assert found == expected_found, (
        f"{note}\nstored={stored_text!r} query={query_text!r} found={found} expected={expected_found}"
    )
