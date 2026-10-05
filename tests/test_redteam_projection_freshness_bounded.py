from __future__ import annotations

import uuid

import pytest

from prometheist import db, event_store, postgres_memory_kernel
from prometheist.attention_aperture import open_attention_aperture
from prometheist.interaction_contracts import (
    deterministic_interaction_event_id,
    deterministic_interaction_id,
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


def test_fresh_projection_does_not_reload_entire_event_ledger_on_every_percept(
    conn,
    monkeypatch,
):
    historical_conversation = uuid.uuid4()
    active_conversation = uuid.uuid4()
    event_store.start_conversation(conn, active_conversation)
    target = _record_prompt(
        conn,
        historical_conversation,
        "Project Borealis uses deployment marker BOREALIS-7719.",
    )
    for index in range(32):
        _record_prompt(
            conn,
            uuid.uuid4(),
            f"Unrelated archive event {index}: warehouse note {uuid.uuid4().hex[:10]}.",
        )
    postgres_memory_kernel.rebuild(conn)

    def reject_full_ledger_materialization(*args, **kwargs):
        raise AssertionError("ordinary recall reloaded the complete canonical event ledger")

    monkeypatch.setattr(
        postgres_memory_kernel,
        "load_events",
        reject_full_ledger_materialization,
    )
    correlation_id = uuid.uuid4()
    interaction_id = deterministic_interaction_id(active_conversation, correlation_id)
    prompt_event_id = deterministic_interaction_event_id(interaction_id, "user-prompt")
    current = event_store.record_event(
        conn,
        conversation_id=active_conversation,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": "What marker belongs to Project Borealis?"},
        payload_text="What marker belongs to Project Borealis?",
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
