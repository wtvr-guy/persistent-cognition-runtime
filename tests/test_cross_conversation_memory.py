"""Acceptance test for system memory across fresh processes and sessions.

Process A (Conversation A) is told an arbitrary fact and exits completely.
Process B (a brand-new Conversation B) asks about it and must still find it
-- conversation boundaries are organizational metadata, not memory walls.

Requires a running local Ollama with the configured model pulled, and
Postgres reachable via DATABASE_URL. Skipped automatically if Ollama isn't
reachable.
"""
from __future__ import annotations

import uuid

import pytest

from prometheist import db, event_store
from prometheist.models import EventType
from tests._cli_helpers import ollama_available, print_transcript, run_once
from tests._native_artifact_assertions import (
    assert_response_evidence_receipt,
    interaction_id_for_prompt,
    print_artifact_receipt,
)

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not ollama_available(), reason="Ollama is not reachable"),
]


def _prompt_event(conversation_id: uuid.UUID, text: str):
    with db.get_connection() as conn:
        events = event_store.get_events_by_conversation(conn, conversation_id)
    return next(
        event
        for event in events
        if event.event_type is EventType.USER_PROMPT and event.payload.get("text") == text
    )


def test_cross_conversation_cross_process_memory_recall():
    conversation_a = uuid.uuid4()
    conversation_b = uuid.uuid4()
    random_fact = uuid.uuid4().hex[:8].upper()

    # Process A, Conversation A: persist the fact, then exit completely.
    sentence = f"The codename for Project Harrier is {random_fact}."
    run_once(sentence, conversation_a)
    source_event = _prompt_event(conversation_a, sentence)

    # Process B, a *different* conversation: no shared conversation_id, no
    # shared process/memory -- only the persisted event history in common.
    prompt = "What codename did I give Project Harrier?"
    answer = run_once(prompt, conversation_b)

    assert answer.strip()
    question_event = _prompt_event(conversation_b, prompt)
    print_transcript(f"\nCross-conversation recall — User:\n{prompt}")
    print_transcript(f"\nCross-conversation recall — Prometheist:\n{answer}")
    print_artifact_receipt(
        "Cross-conversation recall",
        assert_response_evidence_receipt(
            interaction_id=interaction_id_for_prompt(
                conversation_b,
                question_event.correlation_id,
            ),
            required_event_ids=(source_event.event_id,),
        ),
    )
    print_transcript("HUMAN REVIEW REQUIRED: judge the recall answer above.")
