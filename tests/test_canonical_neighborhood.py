from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

import pytest

from prometheist import db, event_store
from prometheist.canonical_neighborhood import (
    CANONICAL_NEIGHBOR_SEQUENCE_WINDOW,
    expand_canonical_neighbors,
)
from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from prometheist.percept_response_runtime import (
    retrieve_memory_package,
)
from tests._retrieval_fixtures import load_person_fidelity_corpus


@pytest.fixture
def conn():
    with db.get_connection() as connection:
        yield connection


def record(conn, conversation, text, event_type=EventType.USER_PROMPT):
    return event_store.record_event(
        conn,
        conversation_id=conversation,
        correlation_id=uuid4(),
        event_type=event_type,
        source="test",
        payload={"text": text},
        payload_text=text,
    )


def packet_for(*events):
    return MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="Why does this matter to me?"),
        supported=bool(events),
        items=[
            MemoryEvidence(
                source_event_id=e.event_id,
                event_type=e.event_type,
                source=e.source,
                created_at=e.created_at,
                conversation_id=e.conversation_id,
                conversation_seq=e.conversation_seq,
                global_seq=e.global_seq,
                content=e.payload["text"],
            ) for e in events
        ],
    )


def expand(conn, packet, cutoff, *, limit=20, types=None):
    return expand_canonical_neighbors(
        conn, packet,
        source_types=types if types is not None else [EventType.USER_PROMPT],
        before_global_seq=cutoff,
        item_limit=limit,
        memory_request_id=uuid4(),
    )


def test_fixed_retrieval_preserves_missing_neighbor_context(conn):
    conversation = event_store.start_conversation(conn)
    root = record(conn, conversation, "Dad and I restored a receiver after the storm.")
    record(conn, conversation, "internal packet", EventType.MEMORY_PACKET)
    seed = record(conn, conversation, "That weekend shaped my career and values.")
    other = event_store.start_conversation(conn)
    question = record(conn, other, "Why does this matter to me?")
    original = packet_for(seed)

    package = retrieve_memory_package(
        conn, SimpleNamespace(interaction_id=uuid4(), conversation_id=other,
                              correlation_id=uuid4(), task_id=uuid4(),
                              user_text=question.payload["text"], before_global_seq=question.global_seq),
        original, [EventType.USER_PROMPT],
    )
    assert {i.source_event_id for i in package.memory_packet.items} == {seed.event_id, root.event_id}
    assert next(i.content for i in package.memory_packet.items if i.source_event_id == root.event_id) == root.payload["text"]


def test_neighbors_are_one_hop_deduplicated_and_preserve_seeds_under_saturation(conn):
    conversation = event_store.start_conversation(conn)
    events = [record(conn, conversation, str(n)) for n in range(5)]
    packet = packet_for(events[2], events[3])
    result = expand(conn, packet, events[-1].global_seq + 1)
    assert [i.source_event_id for i in result.items] == [
        events[2].event_id, events[3].event_id, events[1].event_id, events[4].event_id,
    ]
    assert events[0].event_id not in {i.source_event_id for i in result.items}
    small = expand(conn, packet, events[-1].global_seq + 1, limit=3)
    assert [i.source_event_id for i in small.items] == [
        events[2].event_id, events[3].event_id, events[1].event_id,
    ]
    full = expand(None, packet, events[-1].global_seq + 1, limit=2)
    assert full == packet
    assert expand(None, packet, events[-1].global_seq + 1, types=[]) == packet
    with pytest.raises(ValueError, match="exceeds"):
        expand(None, packet, events[-1].global_seq + 1, limit=1)


def test_scope_cutoff_and_seed_metadata_are_enforced_at_canonical_source(conn):
    conversation = event_store.start_conversation(conn)
    blocked = record(conn, conversation, "observation", EventType.SYSTEM_EVENT)
    seed = record(conn, conversation, "user statement")
    current = record(conn, conversation, "current prompt")
    record(conn, conversation, "future statement")
    other = event_store.start_conversation(conn)
    record(conn, other, "unrelated evidence")
    packet = packet_for(seed)
    packet.items[0].conversation_id = other  # Navigation uses the canonical seed.
    packet.items[0].conversation_seq = 999
    result = expand(conn, packet, current.global_seq)
    assert len(result.items) == 1
    mixed = expand(conn, packet, current.global_seq, types=[
        EventType.USER_PROMPT, EventType.SYSTEM_EVENT,
    ])
    assert mixed.items[1].source_event_id == blocked.event_id
    invalid_seed = packet_for(current)
    invalid_seed.items[0].global_seq = 1  # Cannot forge past-boundary navigation.
    assert len(expand(conn, invalid_seed, current.global_seq).items) == 1
    disallowed = packet_for(blocked)
    disallowed.items[0].event_type = EventType.USER_PROMPT
    assert len(expand(conn, disallowed, current.global_seq).items) == 1


@pytest.mark.parametrize("noise_count", [
    CANONICAL_NEIGHBOR_SEQUENCE_WINDOW - 1,
    CANONICAL_NEIGHBOR_SEQUENCE_WINDOW,
])
def test_sequence_window_stops_at_bound_and_is_replay_stable(conn, noise_count):
    conversation = event_store.start_conversation(conn)
    far = record(conn, conversation, "outside the window")
    for _ in range(noise_count):
        record(conn, conversation, "internal", EventType.MEMORY_PACKET)
    seed = record(conn, conversation, "seed")
    neighbor = record(conn, conversation, "next source")
    packet = packet_for(seed)
    result = expand(conn, packet, neighbor.global_seq + 1)
    replay = expand(conn, packet, neighbor.global_seq + 1)
    expected = [seed.event_id, neighbor.event_id]
    if noise_count < CANONICAL_NEIGHBOR_SEQUENCE_WINDOW:
        expected.append(far.event_id)
    assert [i.source_event_id for i in result.items] == expected
    assert result.items == replay.items
    assert result.retrieval_trace == replay.retrieval_trace


def test_empty_full_and_invalid_budgets_do_not_query():
    packet = packet_for()
    assert expand(None, packet, 1) == packet
    with pytest.raises(ValueError, match="positive"):
        expand(None, packet, 1, limit=0)


@pytest.mark.parametrize("probe_id,seed_id", [
    ("pf-q001", "pf-e002"),
    ("pf-q002", "pf-e004"),
])
def test_frozen_failed_probe_seed_recovers_all_required_canonical_evidence(conn, probe_id, seed_id):
    # Freeze the retrieved seed from SELF-MEMORY-001_2026-09-27_000808.
    # Oracle labels below are used only for assertions, never for navigation.
    corpus = load_person_fidelity_corpus(
        Path(__file__).resolve().parents[1] / "tests/fixtures/retrieval_corpus.json"
    )
    conversations = {}
    events = {}
    for fixture in corpus.life_events:
        if fixture.conversation_id not in conversations:
            conversations[fixture.conversation_id] = event_store.start_conversation(conn)
        events[fixture.event_id] = record(
            conn, conversations[fixture.conversation_id], fixture.text, fixture.event_type,
        )
    probe = next(p for p in corpus.probes if p.probe_id == probe_id)
    packet = packet_for(events[seed_id])
    packet.need.query_text = probe.prompt
    cutoff = max(e.global_seq for e in events.values()) + 1
    result = expand(conn, packet, cutoff)
    required = {events[event_id].event_id for event_id in probe.required_event_ids}
    assert not required.issubset({i.source_event_id for i in packet.items})
    assert required == {i.source_event_id for i in result.items}
