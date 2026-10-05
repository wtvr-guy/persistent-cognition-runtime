from __future__ import annotations

import uuid

import pytest

from prometheist import db, event_store, jit_memory, postgres_memory_kernel
from prometheist.association_feature_projection import AssociationFeatureProjection
from prometheist.models import EventType
from prometheist.postgres_association_projection import load_associations
from prometheist.projection_freshness import committed_projection_high_waters


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
        source="test",
        payload={"text": text, "entities": list(entities)},
        payload_text=text,
    )


def _forbid_lifetime_load(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("ordinary projection catch-up reloaded lifetime history")

    monkeypatch.setattr(postgres_memory_kernel, "load_events", fail)


def test_full_rebuild_seeds_matching_projection_high_waters(conn):
    conversation_id = event_store.start_conversation(conn)
    first = _record(conn, conversation_id, "I usually get a latte in the morning.")
    second = _record(conn, conversation_id, "I bought a used 2021 Toyota Corolla.")
    postgres_memory_kernel.rebuild(conn)

    lexical_high_water, feature_high_water = committed_projection_high_waters(conn)
    assert lexical_high_water == second.global_seq
    assert feature_high_water == second.global_seq
    assert lexical_high_water >= first.global_seq
    features = AssociationFeatureProjection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FROM memory_projection_entries
            WHERE projection_name = %s AND projection_version = %s
            """,
            (features.name, features.version),
        )
        assert cur.fetchone()[0] == 2


def test_incremental_catchup_bridges_previous_state_without_lifetime_load(
    conn,
    monkeypatch,
):
    conversation_id = event_store.start_conversation(conn)
    previous = _record(
        conn,
        conversation_id,
        "I usually get a latte in the morning.",
        entities=("latte",),
    )
    postgres_memory_kernel.rebuild(conn)
    current = _record(
        conn,
        conversation_id,
        "I stopped adding milk to coffee and drink it black.",
        entities=("coffee",),
    )
    _forbid_lifetime_load(monkeypatch)
    jit_memory._ensure_projection_fresh(conn, before_global_seq=current.global_seq + 1)

    edge = next(
        association
        for association in load_associations(conn)
        if association.relationship == "PREVIOUS_STATE"
    )
    assert edge.source == str(current.event_id)
    assert edge.target == str(previous.event_id)
    assert committed_projection_high_waters(conn) == (
        current.global_seq,
        current.global_seq,
    )


def test_incremental_catchup_bridges_resolution_without_lifetime_load(conn, monkeypatch):
    conversation_id = event_store.start_conversation(conn)
    unresolved = _record(
        conn,
        conversation_id,
        "My landlord still owes me the security deposit.",
        entities=("security deposit",),
    )
    postgres_memory_kernel.rebuild(conn)
    resolved = _record(
        conn,
        conversation_id,
        "The landlord returned the security deposit.",
        entities=("security deposit",),
    )
    _forbid_lifetime_load(monkeypatch)
    jit_memory._ensure_projection_fresh(conn, before_global_seq=resolved.global_seq + 1)

    edge = next(
        association
        for association in load_associations(conn)
        if association.relationship == "RESOLVED_BY"
    )
    assert edge.source == str(unresolved.event_id)
    assert edge.target == str(resolved.event_id)


def test_incremental_associations_match_clean_full_rebuild(conn):
    conversation_id = event_store.start_conversation(conn)
    _record(
        conn,
        conversation_id,
        "I usually get a latte in the morning.",
        entities=("latte",),
    )
    _record(
        conn,
        conversation_id,
        "My landlord still owes me the security deposit.",
        entities=("security deposit",),
    )
    postgres_memory_kernel.rebuild(conn)
    last = _record(
        conn,
        conversation_id,
        "I stopped adding milk to coffee and drink it black.",
        entities=("coffee",),
    )
    _record(
        conn,
        conversation_id,
        "The landlord returned the security deposit.",
        entities=("security deposit",),
    )
    _record(conn, conversation_id, "I bought a used 2024 Mazda CX-30 today.")

    jit_memory._ensure_projection_fresh(conn, before_global_seq=None)
    incrementally_derived = load_associations(conn)
    incremental_high_waters = committed_projection_high_waters(conn)
    postgres_memory_kernel.rebuild(conn)

    assert incrementally_derived == load_associations(conn)
    assert incremental_high_waters == committed_projection_high_waters(conn)
    assert incremental_high_waters[0] > last.global_seq
