import uuid

import pytest

from prometheist import db, event_store, jit_memory
from prometheist.interaction_working_state import (
    InteractionWorkingState,
    activate_working_state,
    load_working_state,
)
from prometheist.models import EventType


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _event(conn, conversation_id, text):
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": text},
        payload_text=text,
    )


def test_working_state_is_only_bounded_canonical_event_activation(conn):
    conversation_id = event_store.start_conversation(conn, uuid.uuid4())
    first = _event(conn, conversation_id, "first canonical event")
    second = _event(conn, conversation_id, "second canonical event")
    third = _event(conn, conversation_id, "third canonical event")

    state1 = activate_working_state(
        conn,
        interaction_id=uuid.uuid4(),
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        activated_event_ids=[second.event_id, first.event_id],
    )
    assert state1.revision == 1
    assert state1.active_event_ids == [second.event_id, first.event_id]

    state2 = activate_working_state(
        conn,
        interaction_id=uuid.uuid4(),
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        activated_event_ids=[third.event_id, second.event_id],
    )
    assert state2.revision == 2
    assert state2.active_event_ids == [third.event_id, second.event_id, first.event_id]
    assert load_working_state(conn, conversation_id) == state2

    # v0.7 deliberately does not persist parsed labels/options/conclusions here.
    assert set(InteractionWorkingState.model_fields) == {
        "version",
        "state_id",
        "revision",
        "conversation_ids",
        "active_event_ids",
    }


def test_jit_memory_rehydrates_active_canonical_events_before_lexical_fallback(conn):
    conversation_id = event_store.start_conversation(conn, uuid.uuid4())
    active = _event(
        conn,
        conversation_id,
        "BlueHarbor-ABC123 was the current plan label.",
    )

    packet = jit_memory.request_memory(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        requesting_component="test-working-state",
        need=jit_memory.build_memory_need(
            "lexically unrelated request",
            active_event_ids=[active.event_id],
        ),
        before_global_seq=None,
    )

    assert packet.supported is True
    assert packet.items[0].source_event_id == active.event_id
    assert packet.items[0].retrieval_reasons == ["ACTIVE_WORKING_STATE"]
    assert packet.retrieval_trace["working_state_event_ids"] == [str(active.event_id)]
