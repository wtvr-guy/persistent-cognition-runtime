import uuid

import pytest

from prometheist import db, event_store
from prometheist.memory_kernel import CueState
from prometheist.models import EventType
from prometheist.postgres_memory_kernel import (
    associative_recall_from_postgres,
    rebuild,
    recall_from_postgres,
    verify,
)


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _record(
    conn,
    conversation_id,
    text,
    *,
    entities=(),
    event_type: EventType = EventType.USER_PROMPT,
):
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        event_type=event_type,
        source="test",
        payload={"text": text, "entities": list(entities)},
        payload_text=text,
    )


def test_rebuild_is_idempotent_and_events_remain_authoritative(conn):
    conversation_id = event_store.start_conversation(conn)
    first = _record(conn, conversation_id, "I prefer black coffee.")
    second = _record(conn, conversation_id, "My bicycle is blue.")
    before = [event.model_dump(mode="json") for event in event_store.get_events_by_conversation(conn, conversation_id)]

    first_rebuild = rebuild(conn)
    second_rebuild = rebuild(conn)
    after = [event.model_dump(mode="json") for event in event_store.get_events_by_conversation(conn, conversation_id)]

    assert first_rebuild["projection_digest"] == second_rebuild["projection_digest"]
    assert first_rebuild["source_event_count"] == second_rebuild["source_event_count"]
    assert before == after
    assert verify(conn).valid is True
    assert {first.event_id, second.event_id} == {
        event_store.get_event_by_id(conn, first.event_id).event_id,
        event_store.get_event_by_id(conn, second.event_id).event_id,
    }


def test_postgres_recall_returns_bounded_source_evidence(conn):
    conversation_id = event_store.start_conversation(conn)
    coffee = _record(conn, conversation_id, "I drink black coffee every morning.")
    _record(conn, conversation_id, "I bought a blue bicycle.")
    rebuild(conn)

    packet = recall_from_postgres(
        conn,
        CueState(query_text="What coffee do I drink?", entities=("coffee",), limit=1),
    )

    assert [event.event_id for event in packet.items] == [str(coffee.event_id)]
    assert packet.trace.items[0].selected is True
    assert "LEXICAL_CUE" in packet.trace.items[0].reasons


def test_postgres_candidate_selection_respects_ignored_terms_and_nfkc_entities(conn):
    conversation_id = event_store.start_conversation(conn)
    coffee = _record(conn, conversation_id, "Coffee is black.", entities=("Sarah",))
    _record(conn, conversation_id, "Jordan went for a walk.")
    _record(conn, conversation_id, "Jordan read a book.")
    rebuild(conn)

    packet = recall_from_postgres(
        conn,
        CueState(
            query_text="ＪＯＲＤＡＮ coffee",
            entities=("ＳＡＲＡＨ",),
            ignored_terms=("Jordan",),
            limit=1,
        ),
        candidate_limit=1,
    )

    assert [event.event_id for event in packet.items] == [str(coffee.event_id)]


def test_postgres_exact_entity_route_survives_newer_generic_crowdout(conn):
    conversation_id = event_store.start_conversation(conn)
    target = _record(
        conn,
        conversation_id,
        "The parcel was placed in the archive area.",
        entities=("locker-prb-0042",),
    )
    for index in range(12):
        _record(
            conn,
            conversation_id,
            f"New parcel note {index}: parcel processing was completed today.",
        )
    rebuild(conn)

    packet = recall_from_postgres(
        conn,
        CueState(
            query_text="Where is the parcel?",
            entities=("locker-prb-0042",),
            limit=1,
        ),
        candidate_limit=3,
    )

    assert [event.event_id for event in packet.items] == [str(target.event_id)]


def test_postgres_lexical_specificity_route_survives_newer_one_word_crowdout(conn):
    conversation_id = event_store.start_conversation(conn)
    target = _record(
        conn,
        conversation_id,
        "Parcel confirmation ZX-99 was stored on shelf amber.",
    )
    for index in range(12):
        _record(
            conn,
            conversation_id,
            f"Shelf inspection {index} was completed today.",
        )
    rebuild(conn)

    packet = recall_from_postgres(
        conn,
        CueState(
            query_text="parcel confirmation ZX-99 shelf",
            limit=1,
        ),
        candidate_limit=3,
    )

    assert [event.event_id for event in packet.items] == [str(target.event_id)]


def test_postgres_candidate_routes_apply_source_types_before_truncation(conn):
    conversation_id = event_store.start_conversation(conn)
    target = _record(
        conn,
        conversation_id,
        "Project Zephyr status is amber.",
        event_type=EventType.SYSTEM_EVENT,
    )
    for index in range(12):
        _record(
            conn,
            conversation_id,
            f"Project Zephyr status note {index} says amber.",
            event_type=EventType.USER_PROMPT,
        )
    rebuild(conn)

    packet = recall_from_postgres(
        conn,
        CueState(
            query_text="Project Zephyr status amber",
            source_types=(EventType.SYSTEM_EVENT.value,),
            limit=1,
        ),
        candidate_limit=1,
    )

    assert [event.event_id for event in packet.items] == [str(target.event_id)]


def test_postgres_associative_recall_traverses_previous_state_edge(conn):
    conversation_id = event_store.start_conversation(conn)
    latte = _record(conn, conversation_id, "I usually get a latte in the morning.", entities=("latte",))
    _record(
        conn,
        conversation_id,
        "I stopped adding milk to coffee and drink it black.",
        entities=("coffee",),
    )
    rebuild(conn)

    packet = associative_recall_from_postgres(
        conn,
        CueState(
            query_text="What did I drink before switching coffee habits?",
            limit=1,
        ),
    )

    assert [event.event_id for event in packet.items] == [str(latte.event_id)]
    assert packet.trace.items[0].association_hops[0].relationship == "PREVIOUS_STATE"


def test_postgres_associative_recall_surfaces_vehicle_disposition(conn):
    conversation_id = event_store.start_conversation(conn)
    bought = _record(conn, conversation_id, "I bought a used 2018 Ford Escape today.")
    sold = _record(conn, conversation_id, "I sold the Ford Escape to a neighbor last week.")
    rebuild(conn)

    packet = associative_recall_from_postgres(
        conn,
        CueState(query_text="Do I still own the vehicle?", limit=2),
    )

    assert [event.event_id for event in packet.items] == [
        str(sold.event_id),
        str(bought.event_id),
    ]
    relationships = {
        hop.relationship
        for item in packet.trace.items
        for hop in item.association_hops
    }
    assert {"CONCEPT_INSTANCE", "CONCEPT_DISPOSITION"}.issubset(relationships)


def test_postgres_associative_recall_does_not_use_ownership_edge_for_insurance(conn):
    conversation_id = event_store.start_conversation(conn)
    _record(conn, conversation_id, "I bought a used 2021 Toyota Corolla today.")
    rebuild(conn)

    packet = associative_recall_from_postgres(
        conn,
        CueState(query_text="Who insures my vehicle?", limit=5),
    )

    assert packet.items == ()


def test_postgres_associative_recall_does_not_cross_global_cutoff(conn):
    conversation_id = event_store.start_conversation(conn)
    _record(conn, conversation_id, "I walked to work because the weather was nice.")
    future_purchase = _record(
        conn,
        conversation_id,
        "I bought a used 2021 Toyota Corolla today.",
    )
    rebuild(conn)

    packet = associative_recall_from_postgres(
        conn,
        CueState(query_text="Do I still own the vehicle?", limit=5),
        before_global_seq=future_purchase.global_seq,
    )

    assert str(future_purchase.event_id) not in {
        event.event_id for event in packet.items
    }
    assert packet.items == ()


def test_explicit_canonical_event_seed_is_first_class_association_frontier(conn):
    conversation_id = event_store.start_conversation(conn)
    prior = _record(
        conn,
        conversation_id,
        "I usually get a latte in the morning.",
        entities=("latte",),
    )
    selected = _record(
        conn,
        conversation_id,
        "I stopped adding milk to coffee and drink it black.",
        entities=("coffee",),
    )
    rebuild(conn)

    packet = associative_recall_from_postgres(
        conn,
        CueState(query_text="What came before?", limit=2),
        seed_event_ids=(str(selected.event_id),),
        candidate_limit=2,
        max_hops=3,
    )

    assert f"event:{selected.event_id}" in packet.trace.cue_nodes
    assert str(prior.event_id) in {event.event_id for event in packet.items}
    prior_trace = next(
        item for item in packet.trace.items if item.event_id == str(prior.event_id)
    )
    assert prior_trace.associative_activation > 0
