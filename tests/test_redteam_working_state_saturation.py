from __future__ import annotations

import uuid

import pytest

from prometheist import db, event_store
from prometheist.attention_aperture import open_attention_aperture
from prometheist.interaction_contracts import (
    deterministic_interaction_event_id,
    deterministic_interaction_id,
)
from prometheist.interaction_working_state import (
    MAX_ACTIVE_EVENT_IDS,
    activate_working_state,
    load_working_state,
)
from prometheist.models import EventType


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _record_prompt(conn, conversation_id: uuid.UUID, text: str):
    event_store.start_conversation(conn, conversation_id)
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": text},
        payload_text=text,
    )


def test_newly_relevant_recall_can_enter_saturated_working_state(conn):
    historical_conversation = uuid.uuid4()
    active_conversation = uuid.uuid4()
    target = _record_prompt(
        conn,
        historical_conversation,
        "For Project Meridian, deployment must use PostgreSQL directly on Windows "
        "because virtualization is disabled.",
    )
    stale = [
        _record_prompt(
            conn,
            active_conversation,
            f"Legacy focus item {index}: bakery inventory marker {uuid.uuid4().hex[:8]}",
        )
        for index in range(MAX_ACTIVE_EVENT_IDS)
    ]
    activate_working_state(
        conn,
        interaction_id=uuid.uuid4(),
        conversation_id=active_conversation,
        correlation_id=uuid.uuid4(),
        activated_event_ids=[event.event_id for event in stale],
        activation_key="redteam-saturated-state",
    )

    correlation_id = uuid.uuid4()
    interaction_id = deterministic_interaction_id(active_conversation, correlation_id)
    prompt_event_id = deterministic_interaction_event_id(interaction_id, "user-prompt")
    current_text = "I'm switching to Project Meridian. What deployment constraint applies?"
    current = event_store.record_event(
        conn,
        conversation_id=active_conversation,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": current_text},
        payload_text=current_text,
        event_id=prompt_event_id,
    )
    packet = open_attention_aperture(
        conn,
        conversation_id=active_conversation,
        correlation_id=correlation_id,
        requester_task_id=uuid.uuid4(),
        user_text=current.payload["text"],
        before_global_seq=current.global_seq,
    )

    assert target.event_id in {item.source_event_id for item in packet.items}
    state = load_working_state(conn, active_conversation)
    assert state is not None
    assert state.active_event_ids[0] == prompt_event_id
    assert target.event_id in state.active_event_ids
