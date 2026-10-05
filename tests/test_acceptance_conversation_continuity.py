"""Real-Ollama v0.7 continuity acceptance through fresh guarded workers.

Every turn launches a new CLI process; that controller launches every durable
stage in another freshly guarded process. The model therefore receives neither
an inherited transcript nor persistent worker state.

This experiment deliberately does not turn natural language into an exact-string
oracle. Native answers are printed for human review. The automated verdict proves
that canonical MemoryPacket provenance reached the fresh response worker through
its immutable LLM-invocation artifact, so a plausible-looking answer cannot hide
a continuity or evidence-delivery failure.
"""
from __future__ import annotations

import json
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


def _events(conversation_id: uuid.UUID):
    with db.get_connection() as conn:
        return event_store.get_events_by_conversation(conn, conversation_id)


def _trace(*conversation_ids: uuid.UUID) -> str:
    lines: list[str] = []
    for conversation_id in conversation_ids:
        lines.append(f"conversation={conversation_id}")
        for event in _events(conversation_id):
            payload = json.dumps(event.payload, sort_keys=True, default=str)
            lines.append(
                f"  seq={event.conversation_seq} type={event.event_type.value} "
                f"source={event.source} correlation={event.correlation_id} payload={payload}"
            )
    return "\n".join(lines)


def _event_for_text(conversation_id: uuid.UUID, event_type: EventType, text: str):
    for event in _events(conversation_id):
        if event.event_type is event_type and event.payload.get("text") == text:
            return event
    raise AssertionError(
        f"No {event_type.value} event with exact text {text!r}.\n{_trace(conversation_id)}"
    )


def _response_for_correlation(conversation_id: uuid.UUID, correlation_id: uuid.UUID):
    responses = [
        event
        for event in _events(conversation_id)
        if event.correlation_id == correlation_id
        and event.event_type is EventType.INTERACTION_RESPONSE
    ]
    if len(responses) != 1:
        raise AssertionError(
            f"Expected one INTERACTION_RESPONSE for {correlation_id}, got "
            f"{len(responses)}.\n{_trace(conversation_id)}"
        )
    return responses[0]


def _memory_source_ids_for_correlation(
    conversation_id: uuid.UUID,
    correlation_id: uuid.UUID,
) -> set[uuid.UUID]:
    source_ids: set[uuid.UUID] = set()
    for event in _events(conversation_id):
        if event.correlation_id != correlation_id or event.event_type is not EventType.MEMORY_PACKET:
            continue
        for item in event.payload.get("packet", {}).get("items", []):
            source_ids.add(uuid.UUID(item["source_event_id"]))
    return source_ids


def _seed_distractors(count: int = 12) -> None:
    with db.get_connection() as conn:
        for index in range(count):
            conversation_id = uuid.uuid4()
            text = (
                f"Unrelated historical note {index}: inventory marker "
                f"{uuid.uuid4().hex[:10].upper()} belongs to archive shelf {index}."
            )
            event_store.start_conversation(conn, conversation_id)
            event_store.record_event(
                conn,
                conversation_id=conversation_id,
                correlation_id=uuid.uuid4(),
                event_type=EventType.USER_PROMPT,
                source="user",
                payload={"text": text},
                payload_text=text,
            )


def _print_turn(number: int, prompt: str, answer: str) -> None:
    print_transcript(f"\nTurn {number} — User:\n{prompt}")
    print_transcript(f"\nTurn {number} — Prometheist:\n{answer}")


def test_stateless_four_turn_continuity_survives_sessions_and_distractors():
    historical_conversation = uuid.uuid4()
    active_conversation = uuid.uuid4()
    profile_token = f"VX-{uuid.uuid4().hex[:8].upper()}"
    plan_label = f"BlueHarbor-{uuid.uuid4().hex[:6].upper()}"

    historical_rule = (
        "For Project Kestrel, never use Docker; deploy PostgreSQL directly on Windows "
        f"because virtualization is disabled. I track that constraint under profile {profile_token}."
    )
    historical_answer = run_once(historical_rule, historical_conversation)
    print_transcript(f"\nHistorical seed — User:\n{historical_rule}")
    print_transcript(f"\nHistorical seed — Prometheist:\n{historical_answer}")
    historical_rule_event = _event_for_text(
        historical_conversation,
        EventType.USER_PROMPT,
        historical_rule,
    )
    _seed_distractors()

    turn1 = (
        f"I'm revisiting Project Kestrel. For this conversation, call the plan {plan_label}. "
        "I'm choosing between Docker Compose and PostgreSQL directly on Windows. "
        "Based on my established constraints, choose the compatible approach and "
        "briefly explain why."
    )
    answer1 = run_once(turn1, active_conversation)
    _print_turn(1, turn1, answer1)
    turn1_event = _event_for_text(active_conversation, EventType.USER_PROMPT, turn1)
    turn1_sources = _memory_source_ids_for_correlation(
        active_conversation,
        turn1_event.correlation_id,
    )
    failure_trace = _trace(historical_conversation, active_conversation)
    assert answer1.strip(), failure_trace
    assert historical_rule_event.event_id in turn1_sources, failure_trace
    print_artifact_receipt(
        "Turn 1",
        assert_response_evidence_receipt(
            interaction_id=interaction_id_for_prompt(
                active_conversation,
                turn1_event.correlation_id,
            ),
            required_event_ids=(historical_rule_event.event_id,),
        ),
    )

    turn2 = (
        "Which approach conflicts with my established Kestrel rule, and what constraint "
        "profile did I give that rule? Answer in a short sentence."
    )
    answer2 = run_once(turn2, active_conversation)
    _print_turn(2, turn2, answer2)
    turn2_event = _event_for_text(active_conversation, EventType.USER_PROMPT, turn2)
    answer2_event = _response_for_correlation(active_conversation, turn2_event.correlation_id)
    turn2_sources = _memory_source_ids_for_correlation(
        active_conversation,
        turn2_event.correlation_id,
    )
    failure_trace = _trace(historical_conversation, active_conversation)
    assert answer2.strip(), failure_trace
    assert historical_rule_event.event_id in turn2_sources, failure_trace
    assert turn1_event.event_id in turn2_sources, failure_trace
    print_artifact_receipt(
        "Turn 2",
        assert_response_evidence_receipt(
            interaction_id=interaction_id_for_prompt(
                active_conversation,
                turn2_event.correlation_id,
            ),
            required_event_ids=(historical_rule_event.event_id, turn1_event.event_id),
        ),
    )

    turn3 = (
        "What nickname are we using for this plan, and which approach did you just rule out? "
        "Answer naturally and briefly."
    )
    answer3 = run_once(turn3, active_conversation)
    _print_turn(3, turn3, answer3)
    turn3_event = _event_for_text(active_conversation, EventType.USER_PROMPT, turn3)
    answer3_event = _response_for_correlation(active_conversation, turn3_event.correlation_id)
    turn3_sources = _memory_source_ids_for_correlation(
        active_conversation,
        turn3_event.correlation_id,
    )
    failure_trace = _trace(historical_conversation, active_conversation)
    assert answer3.strip(), failure_trace
    assert turn1_event.event_id in turn3_sources, failure_trace
    assert answer2_event.event_id in turn3_sources, failure_trace
    print_artifact_receipt(
        "Turn 3",
        assert_response_evidence_receipt(
            interaction_id=interaction_id_for_prompt(
                active_conversation,
                turn3_event.correlation_id,
            ),
            required_event_ids=(turn1_event.event_id, answer2_event.event_id),
        ),
    )

    turn4 = (
        "Name that ruled-out approach and its underlying technical reason. "
        "Use the reason wording from my established rule, but answer naturally."
    )
    answer4 = run_once(turn4, active_conversation)
    _print_turn(4, turn4, answer4)
    turn4_event = _event_for_text(active_conversation, EventType.USER_PROMPT, turn4)
    turn4_sources = _memory_source_ids_for_correlation(
        active_conversation,
        turn4_event.correlation_id,
    )
    failure_trace = _trace(historical_conversation, active_conversation)
    assert answer4.strip(), failure_trace
    assert answer3_event.event_id in turn4_sources, failure_trace
    assert historical_rule_event.event_id in turn4_sources, failure_trace
    print_artifact_receipt(
        "Turn 4",
        assert_response_evidence_receipt(
            interaction_id=interaction_id_for_prompt(
                active_conversation,
                turn4_event.correlation_id,
            ),
            required_event_ids=(answer3_event.event_id, historical_rule_event.event_id),
        ),
    )

    active_events = _events(active_conversation)
    prompts = [event for event in active_events if event.event_type is EventType.USER_PROMPT]
    assert [event.payload["text"] for event in prompts] == [turn1, turn2, turn3, turn4]
    for prompt_event in (turn1_event, turn2_event, turn3_event, turn4_event):
        assert any(
            event.event_type is EventType.MEMORY_REQUEST
            and event.correlation_id == prompt_event.correlation_id
            for event in active_events
        ), failure_trace

    print(
        "\nSTRUCTURAL PASS: v0.7 task/worker continuity delivered recent and older "
        "canonical evidence to each fresh responder."
    )
    print("HUMAN REVIEW REQUIRED: judge the four printed Prometheist answers above.")
