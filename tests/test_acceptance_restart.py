"""Cross-process acceptance tests using the real local Ollama adapter.

Each ``run_once`` invocation launches a brand-new Python process. The tests are
skipped when Ollama is unavailable, but when run they exercise real fresh model
calls plus PostgreSQL-backed JIT memory rather than Python mock state.
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


def _review_response(
    *,
    label: str,
    prompt: str,
    answer: str,
    question_conversation: uuid.UUID,
    source_event_id: uuid.UUID,
) -> None:
    question_event = _prompt_event(question_conversation, prompt)
    assert answer.strip()
    print_transcript(f"\n{label} — User:\n{prompt}")
    print_transcript(f"\n{label} — Prometheist:\n{answer}")
    print_artifact_receipt(
        label,
        assert_response_evidence_receipt(
            interaction_id=interaction_id_for_prompt(
                question_conversation,
                question_event.correlation_id,
            ),
            required_event_ids=(source_event_id,),
        ),
    )
    print_transcript(f"HUMAN REVIEW REQUIRED: judge the {label.casefold()} above.")


@pytest.mark.parametrize(
    "codename_sentence,question",
    [
        (
            "The codename for Project Oriole is {fact}.",
            "What codename did I give Project Oriole?",
        ),
        (
            "The launch code for Project Falcon is {fact}.",
            "What launch code did I give for Project Falcon?",
        ),
    ],
)
def test_cross_process_restart_recalls_randomized_fact(codename_sentence, question):
    conversation_id = uuid.uuid4()
    random_fact = uuid.uuid4().hex[:8].upper()
    sentence = codename_sentence.format(fact=random_fact)

    run_once(sentence, conversation_id)
    source_event = _prompt_event(conversation_id, sentence)
    answer = run_once(question, conversation_id)

    _review_response(
        label="Cross-process restart recall",
        prompt=question,
        answer=answer,
        question_conversation=conversation_id,
        source_event_id=source_event.event_id,
    )


def test_cross_process_adaptive_recall_recalls_without_hidden_transcript():
    fact_conversation = uuid.uuid4()
    task_conversation = uuid.uuid4()
    random_fact = uuid.uuid4().hex[:8].upper()

    # Process A persists the event and exits completely.
    sentence = f"The codename for Project Oriole is {random_fact}."
    run_once(sentence, fact_conversation)
    source_event = _prompt_event(fact_conversation, sentence)

    # Process B uses the v2 Composer and deterministic Adaptive Recall before a
    # separate final responder receives bounded evidence.
    prompt = "Tell me the codename for Project Oriole from persistent memory."
    answer = run_once(prompt, task_conversation)

    _review_response(
        label="Cross-conversation Adaptive Recall",
        prompt=prompt,
        answer=answer,
        question_conversation=task_conversation,
        source_event_id=source_event.event_id,
    )
