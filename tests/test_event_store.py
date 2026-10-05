from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import uuid

import pytest

from prometheist import db, event_store
from prometheist.models import EventType


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def test_record_and_read_every_event_type(conn):
    conversation_id = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()

    for event_type, payload in [
        (EventType.USER_PROMPT, {"text": "hello"}),
        (EventType.AGENT_DECISION, {"action": "RESPOND_DIRECTLY"}),
        (EventType.RETRIEVAL_REQUEST, {"query_text": "x"}),
        (EventType.RETRIEVAL_RESULT, {"items": []}),
        (EventType.AGENT_RESPONSE, {"text": "hi"}),
        (EventType.SYSTEM_EVENT, {"note": "started"}),
        (EventType.ERROR, {"message": "boom"}),
    ]:
        event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            event_type=event_type,
            source="test",
            payload=payload,
        )

    events = event_store.get_events_by_conversation(conn, conversation_id)
    assert [e.event_type for e in events] == [
        EventType.USER_PROMPT,
        EventType.AGENT_DECISION,
        EventType.RETRIEVAL_REQUEST,
        EventType.RETRIEVAL_RESULT,
        EventType.AGENT_RESPONSE,
        EventType.SYSTEM_EVENT,
        EventType.ERROR,
    ]


def test_conversation_seq_is_monotonic_per_conversation(conn):
    conversation_id = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()

    events = [
        event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            event_type=EventType.USER_PROMPT,
            source="test",
            payload={"text": str(i)},
        )
        for i in range(5)
    ]

    seqs = [e.conversation_seq for e in events]
    assert seqs == sorted(seqs)
    assert seqs == list(range(1, 6))


def test_concurrent_writers_preserve_same_conversation_sequence(conn):
    """Independent local agents (simulated via concurrent DB connections) may safely append to one conversation."""
    conversation_id = event_store.start_conversation(conn)
    worker_count = 8
    start_together = Barrier(worker_count)

    def append(index: int):
        connection = db.get_connection()
        try:
            start_together.wait(timeout=10)
            return event_store.record_event(
                connection,
                conversation_id=conversation_id,
                correlation_id=uuid.uuid4(),
                event_type=EventType.SYSTEM_EVENT,
                source=f"concurrent-worker-{index}",
                payload={"worker": index},
            )
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(append, index) for index in range(worker_count)]
        written = [future.result(timeout=20) for future in futures]

    assert sorted(event.conversation_seq for event in written) == list(
        range(1, worker_count + 1)
    )
    assert len({event.global_seq for event in written}) == worker_count
    assert len({event.event_id for event in written}) == worker_count

    stored = event_store.get_events_by_conversation(conn, conversation_id)
    assert [event.conversation_seq for event in stored] == list(
        range(1, worker_count + 1)
    )
    assert {event.payload["worker"] for event in stored} == set(range(worker_count))


def test_global_seq_is_monotonic_across_conversations(conn):
    conv_a = event_store.start_conversation(conn)
    conv_b = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()

    ev_a = event_store.record_event(
        conn,
        conversation_id=conv_a,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": "a"},
    )
    ev_b = event_store.record_event(
        conn,
        conversation_id=conv_b,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": "b"},
    )
    assert ev_b.global_seq > ev_a.global_seq


def test_first_event_in_conversation(conn):
    conversation_id = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()
    first = event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": "first"},
    )
    event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": "second"},
    )

    result = event_store.get_first_event_in_conversation(conn, conversation_id)
    assert result.event_id == first.event_id
    assert result.payload["text"] == "first"


def test_event_store_has_no_update_or_delete_functions():
    exported = dir(event_store)
    assert not any(name.startswith("update_") for name in exported)
    assert not any(name.startswith("delete_") for name in exported)
