import uuid

import pytest

from prometheist import db, event_store, jit_memory
from prometheist.models import EventType


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _record_text(conn, conversation_id, text, *, event_type=EventType.USER_PROMPT, source="user"):
    event_store.start_conversation(conn, conversation_id)
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        event_type=event_type,
        source=source,
        payload={"text": text},
        payload_text=text,
    )


def test_shared_memory_boundary_projects_new_events_without_manual_rebuild(conn):
    source_conversation = uuid.uuid4()
    request_conversation = uuid.uuid4()
    token = uuid.uuid4().hex[:10].upper()
    source_event = _record_text(
        conn,
        source_conversation,
        f"The codename for Project Oriole is {token}.",
    )
    current_prompt = _record_text(conn, request_conversation, "What is Project Oriole's codename?")

    need = jit_memory.build_memory_need(
        "Project Oriole codename",
        entities=["Project Oriole"],
        conversation_id=request_conversation,
    )
    packet = jit_memory.request_memory(
        conn,
        conversation_id=request_conversation,
        correlation_id=current_prompt.correlation_id,
        requesting_component="test-component",
        need=need,
        before_global_seq=current_prompt.global_seq,
    )

    assert packet.supported is True
    assert packet.items[0].source_event_id == source_event.event_id
    assert token in packet.items[0].content

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FROM memory_projection_entries
            WHERE source_event_id = %s AND projection_name = 'lexical'
            """,
            (source_event.event_id,),
        )
        assert cur.fetchone()[0] == 1


def test_attention_activation_surfaces_sparse_candidate_without_weakening_evidence_gate(conn):
    source_conversation = uuid.uuid4()
    request_conversation = uuid.uuid4()
    token = uuid.uuid4().hex[:10].upper()
    source_event = _record_text(
        conn,
        source_conversation,
        f"I want you to remember {token}.",
    )
    question = "What number did I ask you to remember, as we just discussed?"
    current_prompt = _record_text(conn, request_conversation, question)

    conservative = jit_memory.request_memory(
        conn,
        conversation_id=request_conversation,
        correlation_id=current_prompt.correlation_id,
        requesting_component="evidence-test",
        need=jit_memory.build_memory_need(question),
        before_global_seq=current_prompt.global_seq,
        memory_request_id=uuid.uuid4(),
    )
    assert conservative.supported is False
    assert conservative.items == []

    activation = jit_memory.request_attention_activation(
        conn,
        conversation_id=request_conversation,
        correlation_id=current_prompt.correlation_id,
        requesting_component="aperture-test",
        need=jit_memory.build_memory_need(question),
        before_global_seq=current_prompt.global_seq,
        memory_request_id=uuid.uuid4(),
    )

    assert activation.supported is True
    assert source_event.event_id in {item.source_event_id for item in activation.items}
    activated = next(item for item in activation.items if item.source_event_id == source_event.event_id)
    assert "ATTENTION_APERTURE" in activated.retrieval_reasons
    assert activation.retrieval_trace["retrieval_role"] == "ATTENTION_ACTIVATION"


def test_attention_activation_guarantees_active_state_beyond_recall_budget(conn):
    conversation_id = uuid.uuid4()
    active_events = [
        _record_text(conn, conversation_id, f"Active working-state fact {index}.")
        for index in range(7)
    ]
    critical_rule = _record_text(
        conn,
        conversation_id,
        "For Project Kestrel, never use Docker because virtualization is disabled.",
    )
    active_events.append(critical_rule)
    current_prompt = _record_text(
        conn,
        conversation_id,
        "Name the ruled-out approach and its underlying technical reason.",
    )
    need = jit_memory.build_memory_need(
        current_prompt.payload["text"],
        active_event_ids=[event.event_id for event in active_events],
        include_persisted_history=False,
        limit=6,
    )

    activation = jit_memory.request_attention_activation(
        conn,
        conversation_id=conversation_id,
        correlation_id=current_prompt.correlation_id,
        requesting_component="active-state-aperture-test",
        need=need,
        before_global_seq=current_prompt.global_seq,
        memory_request_id=uuid.uuid4(),
    )

    assert [item.source_event_id for item in activation.items] == [
        event.event_id for event in active_events
    ]
    assert critical_rule.event_id in {item.source_event_id for item in activation.items}
    assert activation.retrieval_trace["working_state_item_count"] == 8
    assert activation.retrieval_trace["baseline_recall_limit"] == 6
    assert activation.retrieval_trace["packet_item_limit"] == 14

    conservative = jit_memory.request_memory(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid.uuid4(),
        requesting_component="active-state-evidence-test",
        need=need,
        before_global_seq=current_prompt.global_seq,
        memory_request_id=uuid.uuid4(),
    )
    assert len(conservative.items) == 6
    assert critical_rule.event_id not in {item.source_event_id for item in conservative.items}


def test_memory_boundary_uses_supplemental_query_only_after_canonical_abstains(conn):
    source_conversation = uuid.uuid4()
    request_conversation = uuid.uuid4()
    token = uuid.uuid4().hex[:10].upper()
    source_event = _record_text(
        conn,
        source_conversation,
        f"I want you to remember {token}.",
    )
    question = "What number did I ask you to remember?"
    current_prompt = _record_text(conn, request_conversation, question)

    packet = jit_memory.request_memory(
        conn,
        conversation_id=request_conversation,
        correlation_id=current_prompt.correlation_id,
        requesting_component="interaction-test",
        need=jit_memory.build_memory_need(
            question,
            supplemental_query_texts=["remember"],
            conversation_id=request_conversation,
        ),
        before_global_seq=current_prompt.global_seq,
    )

    assert packet.supported is True
    assert source_event.event_id in {item.source_event_id for item in packet.items}
    assert token in packet.items[0].content
    assert packet.retrieval_trace["selected_query_role"] == "supplemental"
    attempts = packet.retrieval_trace["query_attempts"]
    assert [attempt["role"] for attempt in attempts] == ["canonical", "supplemental"]
    assert attempts[0]["supported"] is False
    assert attempts[1]["supported"] is True
    assert packet.retrieval_trace["kernel_trace"] == attempts[1]["kernel_trace"]


def test_working_state_context_composes_with_required_historical_evidence(conn):
    historical_conversation = uuid.uuid4()
    active_conversation = uuid.uuid4()
    historical = _record_text(
        conn,
        historical_conversation,
        "For Project Kestrel, never use Docker; deploy PostgreSQL directly on Windows "
        "because virtualization is disabled. I track that constraint under profile VX-TEST1234.",
    )
    active = _record_text(
        conn,
        active_conversation,
        "I'm revisiting Project Kestrel and choosing between Docker Compose and "
        "running PostgreSQL directly on Windows.",
    )
    current_prompt = _record_text(
        conn,
        active_conversation,
        "Which approach conflicts with my established rule, and what profile did I give it?",
    )

    packet = jit_memory.request_memory(
        conn,
        conversation_id=active_conversation,
        correlation_id=current_prompt.correlation_id,
        requesting_component="working-state-composition-test",
        need=jit_memory.build_memory_need(
            "Which approach conflicts with my established rule, and what profile did I give it?",
            supplemental_query_texts=["Project Kestrel established rule constraint profile"],
            entities=["Project Kestrel"],
            active_event_ids=[active.event_id],
            conversation_id=None,
        ),
        before_global_seq=current_prompt.global_seq,
    )

    source_ids = {item.source_event_id for item in packet.items}
    assert active.event_id in source_ids
    assert historical.event_id in source_ids
    assert packet.supported is True
    assert packet.retrieval_trace["working_state_event_ids"] == [str(active.event_id)]
    assert packet.retrieval_trace["selected_query_role"] in {"canonical", "supplemental"}


def test_associative_stage_keeps_current_semantics_for_guarded_association_edges(conn):
    conversation_id = uuid.uuid4()
    prior = _record_text(
        conn,
        conversation_id,
        "I usually get a latte in the morning.",
    )
    selected = _record_text(
        conn,
        conversation_id,
        "I stopped adding milk to coffee and drink it black.",
    )
    question = "What did I drink before that change?"
    current_prompt = _record_text(conn, conversation_id, question)

    packet = jit_memory.request_memory(
        conn,
        conversation_id=conversation_id,
        correlation_id=current_prompt.correlation_id,
        requesting_component="associative-semantics-test",
        need=jit_memory.build_memory_need(
            question,
            focus_event_ids=[selected.event_id],
            conversation_id=None,
            limit=3,
        ),
        before_global_seq=current_prompt.global_seq,
        memory_request_id=uuid.uuid4(),
        recall_stage=jit_memory.AdaptiveRecallStage.ASSOCIATIVE,
    )

    assert prior.event_id in {item.source_event_id for item in packet.items}
    assert selected.event_id not in {item.source_event_id for item in packet.items}
    assert packet.retrieval_trace["semantic_query_text"] == question
    attempt = packet.retrieval_trace["focus_attempts"][0]
    assert attempt["semantic_query_text"] == question
    cue_nodes = attempt["kernel_trace"]["cue_nodes"]
    assert "term:before" in cue_nodes
    assert f"event:{selected.event_id}" in cue_nodes


def test_memory_boundary_preserves_unknown_fact_abstention(conn):
    source_conversation = uuid.uuid4()
    request_conversation = uuid.uuid4()
    _record_text(conn, source_conversation, "My vehicle is a red Mazda crossover.")
    current_prompt = _record_text(conn, request_conversation, "Who insures my vehicle?")

    packet = jit_memory.request_memory(
        conn,
        conversation_id=request_conversation,
        correlation_id=current_prompt.correlation_id,
        requesting_component="test-component",
        need=jit_memory.build_memory_need("Who insures my vehicle?"),
        before_global_seq=current_prompt.global_seq,
    )

    assert packet.supported is False
    assert packet.items == []


def test_memory_request_and_packet_are_persisted_with_same_request_id(conn):
    conversation_id = uuid.uuid4()
    source_event = _record_text(conn, conversation_id, "Remember the passphrase cobalt raven.")
    current_prompt = _record_text(conn, conversation_id, "What passphrase did I give you?")

    packet = jit_memory.request_memory(
        conn,
        conversation_id=conversation_id,
        correlation_id=current_prompt.correlation_id,
        requesting_component="interaction-test",
        need=jit_memory.build_memory_need("passphrase cobalt raven"),
        before_global_seq=current_prompt.global_seq,
    )
    assert source_event.event_id in {item.source_event_id for item in packet.items}

    events = event_store.get_events_by_conversation(conn, conversation_id)
    request_event = next(event for event in events if event.event_type == EventType.MEMORY_REQUEST)
    packet_event = next(event for event in events if event.event_type == EventType.MEMORY_PACKET)

    assert request_event.correlation_id == current_prompt.correlation_id
    assert packet_event.correlation_id == current_prompt.correlation_id
    assert request_event.payload["memory_request_id"] == str(packet.memory_request_id)
    assert packet_event.payload["packet"]["memory_request_id"] == str(packet.memory_request_id)
    assert request_event.payload["origin"] == "INTERNAL_MEMORY"
    assert request_event.payload["retrieval_role"] == "EVIDENCE"


def test_control_plane_memory_events_do_not_become_default_evidence(conn):
    conversation_id = uuid.uuid4()
    source_event = _record_text(conn, conversation_id, "The launch phrase is silver comet.")
    first_prompt = _record_text(conn, conversation_id, "What is the launch phrase?")

    first = jit_memory.request_memory(
        conn,
        conversation_id=conversation_id,
        correlation_id=first_prompt.correlation_id,
        requesting_component="interaction-test",
        need=jit_memory.build_memory_need("launch phrase silver comet"),
        before_global_seq=first_prompt.global_seq,
    )
    assert source_event.event_id in {item.source_event_id for item in first.items}

    later_prompt = _record_text(conn, conversation_id, "Tell me the launch phrase again.")
    second = jit_memory.request_memory(
        conn,
        conversation_id=conversation_id,
        correlation_id=later_prompt.correlation_id,
        requesting_component="memory-analysis-test",
        need=jit_memory.build_memory_need("launch phrase silver comet"),
        before_global_seq=later_prompt.global_seq,
    )

    assert second.items
    assert all(item.event_type not in {EventType.MEMORY_REQUEST, EventType.MEMORY_PACKET} for item in second.items)
