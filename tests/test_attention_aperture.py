import uuid

import pytest

from prometheist import db, event_store
from prometheist.attention_aperture import (
    ATTENTION_APERTURE_VERSION,
    DEFAULT_ATTENTION_APERTURE_LIMIT,
    MAX_ATTENTION_APERTURE_ITEMS,
    open_attention_aperture,
)
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


def test_every_percept_can_activate_relevant_history_without_model_planning(conn):
    historical_conversation = uuid.uuid4()
    active_conversation = uuid.uuid4()
    historical = _record_prompt(
        conn,
        historical_conversation,
        "For Project Atlas, never use Docker; deploy PostgreSQL directly on Windows "
        "because virtualization is disabled.",
    )
    current = _record_prompt(
        conn,
        active_conversation,
        "I'm revisiting Project Atlas and choosing between Docker Compose and "
        "PostgreSQL directly on Windows.",
    )

    packet = open_attention_aperture(
        conn,
        conversation_id=active_conversation,
        correlation_id=current.correlation_id,
        requester_task_id=uuid.uuid4(),
        user_text=current.payload["text"],
        before_global_seq=current.global_seq,
    )

    assert historical.event_id in {item.source_event_id for item in packet.items}
    assert len(packet.items) <= DEFAULT_ATTENTION_APERTURE_LIMIT
    assert packet.retrieval_trace["retrieval_role"] == "ATTENTION_ACTIVATION"
    assert packet.retrieval_trace["activation_kernel"]["kernel"] == "recall_from_postgres"

    events = event_store.get_events_by_conversation(conn, active_conversation)
    request = next(event for event in events if event.event_type is EventType.MEMORY_REQUEST)
    assert request.payload["retrieval_role"] == "ATTENTION_ACTIVATION"
    assert request.source.startswith(f"attention-aperture:{ATTENTION_APERTURE_VERSION}")


def test_aperture_activation_does_not_require_conversation_scope(conn):
    source_conversation = uuid.uuid4()
    current_conversation = uuid.uuid4()
    source = _record_prompt(
        conn,
        source_conversation,
        "Project Meridian uses the opaque marker MERIDIAN-8421.",
    )
    current = _record_prompt(
        conn,
        current_conversation,
        "What marker belongs to Project Meridian?",
    )

    packet = open_attention_aperture(
        conn,
        conversation_id=current_conversation,
        correlation_id=current.correlation_id,
        requester_task_id=uuid.uuid4(),
        user_text=current.payload["text"],
        before_global_seq=current.global_seq,
    )

    assert source.event_id in {item.source_event_id for item in packet.items}
    assert packet.need.conversation_id is None


def test_full_working_state_is_exposed_and_current_percept_survives_truncation(conn):
    conversation_id = uuid.uuid4()
    seeded = [
        _record_prompt(conn, conversation_id, f"Durable active fact {index}.")
        for index in range(MAX_ACTIVE_EVENT_IDS)
    ]
    activate_working_state(
        conn,
        interaction_id=uuid.uuid4(),
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        activated_event_ids=[event.event_id for event in seeded],
        activation_key="seed-full-state",
    )

    correlation_id = uuid.uuid4()
    interaction_id = deterministic_interaction_id(conversation_id, correlation_id)
    prompt_event_id = deterministic_interaction_event_id(interaction_id, "user-prompt")
    current = event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": "Which active fact still applies?"},
        payload_text="Which active fact still applies?",
        event_id=prompt_event_id,
    )

    packet = open_attention_aperture(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        requester_task_id=uuid.uuid4(),
        user_text=current.payload["text"],
        before_global_seq=current.global_seq,
    )

    packet_ids = {item.source_event_id for item in packet.items}
    assert {event.event_id for event in seeded} <= packet_ids
    assert len(packet.items) <= MAX_ATTENTION_APERTURE_ITEMS
    state = load_working_state(conn, conversation_id)
    assert state is not None
    assert state.active_event_ids[0] == prompt_event_id
    assert len(state.active_event_ids) == MAX_ACTIVE_EVENT_IDS
