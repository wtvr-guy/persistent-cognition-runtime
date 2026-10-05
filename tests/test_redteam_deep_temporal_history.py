from __future__ import annotations

import uuid

import pytest

from prometheist import db, event_store, postgres_memory_kernel
from prometheist.jit_memory import ATTENTION_ACTIVATION_CANDIDATE_LIMIT
from prometheist.memory_kernel import CueState
from prometheist.models import EventType


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _record(conn, conversation_id: uuid.UUID, text: str):
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": text},
        payload_text=text,
    )


def test_original_fact_survives_more_same_topic_updates_than_candidate_window(conn):
    conversation_id = event_store.start_conversation(conn)
    original_code = f"ORIGIN-{uuid.uuid4().hex[:10].upper()}"
    target = _record(
        conn,
        conversation_id,
        f"The launch code for Project Falcon is {original_code}.",
    )
    for index in range(ATTENTION_ACTIVATION_CANDIDATE_LIMIT + 64):
        _record(
            conn,
            conversation_id,
            f"The launch code for Project Falcon is DECOY-{index:04d}.",
        )

    postgres_memory_kernel.rebuild(conn)
    packet = postgres_memory_kernel.recall_from_postgres(
        conn,
        CueState(
            query_text="What launch code did I originally give Project Falcon?",
            source_types=(EventType.USER_PROMPT.value,),
            limit=6,
        ),
        candidate_limit=ATTENTION_ACTIVATION_CANDIDATE_LIMIT,
    )

    assert str(target.event_id) in {event.event_id for event in packet.items}
