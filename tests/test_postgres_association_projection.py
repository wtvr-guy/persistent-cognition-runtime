import uuid

import pytest

from prometheist import db, event_store
from prometheist.models import EventType
from prometheist.postgres_association_projection import load_associations
from prometheist.postgres_memory_kernel import rebuild


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _record(conn, conversation_id, text, *, entities=()):
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        event_type=EventType.USER_PROMPT,
        source="test-user",
        payload={"text": text, "entities": list(entities)},
        payload_text=text,
    )


def test_rebuild_persists_deterministic_associations_without_touching_events(conn):
    conversation_id = event_store.start_conversation(conn)
    old = _record(conn, conversation_id, "I usually get a latte.", entities=("latte",))
    change = _record(
        conn,
        conversation_id,
        "I stopped adding milk to coffee and drink it black.",
        entities=("coffee",),
    )

    first = rebuild(conn)
    first_associations = load_associations(conn)
    second = rebuild(conn)
    second_associations = load_associations(conn)

    assert first["association_entry_count"] == 1
    assert second["association_entry_count"] == 1
    assert first["association_projection_digest"] == second["association_projection_digest"]
    assert second_associations == first_associations
    assert first_associations[0].relationship == "PREVIOUS_STATE"
    assert first_associations[0].source == str(change.event_id)
    assert first_associations[0].target == str(old.event_id)
    assert first_associations[0].provenance_event_ids == (
        str(old.event_id),
        str(change.event_id),
    )
    assert event_store.get_event_by_id(conn, old.event_id).event_id == old.event_id
    assert event_store.get_event_by_id(conn, change.event_id).event_id == change.event_id
