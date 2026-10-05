from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from prometheist import db, event_store
from prometheist.cognitive_store import list_records, put_record
from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from prometheist.perception import normalize_user_interaction_percept
from prometheist.self_memory import (
    FutureOrientation,
    IdentityCentrality,
    PlasticityClass,
    SELF_SUBJECT,
    SelfPerspective,
    SelfRepresentationKind,
    SelfResolutionStatus,
    current_self_resolution,
    default_plasticity,
    resolve_self_representation,
    self_evidence,
)
from prometheist.self_reflection import (
    ReflectionEvidence,
    SelfSchemaProposal,
    SelfSchemaReview,
    apply_review,
    collect_consolidation_evidence,
    materialize_proposal,
    review_memory_packet,
)
from prometheist.situations import persist_situations


@pytest.fixture
def conn():
    with db.get_connection() as connection:
        yield connection


def _record_user_prompt(conn, text: str):
    conversation_id = uuid4()
    correlation_id = uuid4()
    event_store.start_conversation(conn, conversation_id)
    event = event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={"text": text},
        payload_text=text,
        event_id=uuid4(),
    )
    percept = normalize_user_interaction_percept(
        user_text=text,
        observed_at=event.created_at,
        correlation_id=correlation_id,
        source_event_id=event.event_id,
        conversation_id=conversation_id,
    )
    situations = persist_situations(conn, percept)
    return event, situations


def test_proposal_model_cannot_choose_plasticity():
    assert "plasticity" not in SelfSchemaProposal.model_fields
    assert default_plasticity(SelfRepresentationKind.EMBODIMENT_STATE) is (
        PlasticityClass.FAST
    )
    assert default_plasticity(SelfRepresentationKind.PREFERENCE) is (
        PlasticityClass.MEDIUM
    )
    assert default_plasticity(SelfRepresentationKind.VALUE) is PlasticityClass.SLOW
    assert default_plasticity(SelfRepresentationKind.NARRATIVE_HYPOTHESIS) is (
        PlasticityClass.VERY_SLOW
    )


def test_consolidation_evidence_includes_free_text_roots_and_real_situation_contexts(conn):
    first, _ = _record_user_prompt(conn, "I prefer systems I can inspect.")
    second, _ = _record_user_prompt(conn, "I also choose local control over convenience.")

    rows = list_records(conn, "situation_ingest")
    selected = [
        key
        for key, data in rows
        if str(first.event_id) in data["provenance"]
        or str(second.event_id) in data["provenance"]
    ]
    assert len(selected) == 2

    consolidation_id = uuid4()
    put_record(
        conn,
        "consolidation_input",
        str(consolidation_id),
        {
            "after_key": "",
            "record_keys": selected,
            "derived_at": datetime.now(timezone.utc).isoformat(),
            "next_cursor": selected[-1],
        },
        revision="1",
    )

    evidence = collect_consolidation_evidence(conn, consolidation_id)
    by_id = {item.event_id: item for item in evidence}

    assert first.event_id in by_id
    assert second.event_id in by_id
    assert len(by_id[first.event_id].context_refs) == 1
    assert len(by_id[second.event_id].context_refs) == 1
    assert by_id[first.event_id].context_refs != by_id[second.event_id].context_refs


def test_reflection_preserves_observation_time_separately_from_known_time(conn):
    historical = datetime(2021, 2, 3, 19, 10, tzinfo=timezone.utc)
    conversation_id = uuid4()
    correlation_id = uuid4()
    event_store.start_conversation(conn, conversation_id)
    root = event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="user",
        payload={
            "text": "I learned to value graceful failure during the radio outage.",
            "benchmark_fixture": {"occurred_at": historical.isoformat()},
        },
        payload_text="I learned to value graceful failure during the radio outage.",
        event_id=uuid4(),
    )
    percept = normalize_user_interaction_percept(
        user_text=root.payload["text"],
        observed_at=historical,
        correlation_id=correlation_id,
        source_event_id=root.event_id,
        conversation_id=conversation_id,
    )
    situation = persist_situations(conn, percept)[0]
    consolidation_id = uuid4()
    put_record(
        conn,
        "consolidation_input",
        str(consolidation_id),
        {
            "after_key": "",
            "record_keys": [
                f"{situation.situation_id}:{percept.percept_id}",
            ],
            "derived_at": datetime.now(timezone.utc).isoformat(),
            "next_cursor": None,
        },
        revision="1",
    )

    evidence = collect_consolidation_evidence(conn, consolidation_id)
    assert len(evidence) == 1
    assert datetime.fromisoformat(evidence[0].created_at) == historical

    proposal = SelfSchemaProposal(
        kind=SelfRepresentationKind.VALUE,
        perspective=SelfPerspective.AVOWED,
        statement="Graceful failure matters to me.",
        identity_centrality=IdentityCentrality.CENTRAL,
        support_indices=(0,),
    )
    representation, _ = materialize_proposal(
        conn,
        proposal=proposal,
        evidence=evidence,
        derived_at=datetime.now(timezone.utc),
    )
    stored = self_evidence(conn, representation.representation_id)

    assert len(stored) == 1
    assert stored[0].observed_at == historical
    assert stored[0].known_at == root.created_at
    assert stored[0].known_at > stored[0].observed_at


def test_materialization_uses_application_contexts_for_breadth(conn):
    first, _ = _record_user_prompt(conn, "I chose inspectability over speed.")
    second, _ = _record_user_prompt(conn, "I chose auditability over convenience.")
    evidence = (
        ReflectionEvidence(
            event_id=first.event_id,
            event_type=first.event_type,
            source=first.source,
            created_at=first.created_at.isoformat(),
            content=first.payload["text"],
            context_refs=("situation:first",),
        ),
        ReflectionEvidence(
            event_id=second.event_id,
            event_type=second.event_type,
            source=second.source,
            created_at=second.created_at.isoformat(),
            content=second.payload["text"],
            context_refs=("situation:second",),
        ),
    )
    proposal = SelfSchemaProposal(
        kind=SelfRepresentationKind.VALUE,
        perspective=SelfPerspective.INFERRED,
        statement="Auditability strongly influences important choices.",
        identity_centrality=IdentityCentrality.CENTRAL,
        context_tags=("architecture", "systems"),
        support_indices=(0, 1),
    )

    representation, _ = materialize_proposal(
        conn,
        proposal=proposal,
        evidence=evidence,
        derived_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    resolution = current_self_resolution(conn, representation.representation_id)
    assert resolution is not None
    assert resolution.status is SelfResolutionStatus.CANDIDATE

    established = resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.CENTRAL,
        counterevidence_checked=True,
        resolved_at=datetime.now(timezone.utc) + timedelta(minutes=2),
    )

    assert established.status is SelfResolutionStatus.ESTABLISHED
    assert established.metrics.context_count == 2
    assert {
        tag
        for item in self_evidence(conn, representation.representation_id)
        for tag in item.context_tags
    } == {"situation:first", "situation:second"}


def test_invalid_support_index_fails_before_creating_representation(conn):
    root, _ = _record_user_prompt(conn, "I like modular systems.")
    proposal = SelfSchemaProposal(
        kind=SelfRepresentationKind.PREFERENCE,
        perspective=SelfPerspective.AVOWED,
        statement="I like modular systems.",
        identity_centrality=IdentityCentrality.MODERATE,
        support_indices=(1,),
    )
    evidence = (
        ReflectionEvidence(
            event_id=root.event_id,
            event_type=root.event_type,
            source=root.source,
            created_at=root.created_at.isoformat(),
            content=root.payload["text"],
            context_refs=("situation:test",),
        ),
    )

    with pytest.raises(ValueError, match="unavailable support"):
        materialize_proposal(
            conn,
            proposal=proposal,
            evidence=evidence,
            derived_at=datetime.now(timezone.utc),
        )


def test_review_has_no_durable_status_authority():
    review = SelfSchemaReview(opposition_indices=(), rationale="No contrary evidence found.")
    assert review.opposition_indices == ()
    with pytest.raises(ValueError):
        SelfSchemaReview(verdict="ESTABLISH", opposition_indices=(), rationale="Trust me.")


def test_review_cannot_turn_a_support_root_into_opposition(conn):
    root, _ = _record_user_prompt(conn, "I prefer modular systems.")
    proposal = SelfSchemaProposal(
        kind=SelfRepresentationKind.PREFERENCE,
        perspective=SelfPerspective.AVOWED,
        statement="I prefer modular systems.",
        identity_centrality=IdentityCentrality.MODERATE,
        support_indices=(0,),
    )
    evidence = (
        ReflectionEvidence(
            event_id=root.event_id,
            event_type=root.event_type,
            source=root.source,
            created_at=root.created_at.isoformat(),
            content=root.payload["text"],
            context_refs=("situation:test",),
        ),
    )
    representation, _ = materialize_proposal(
        conn,
        proposal=proposal,
        evidence=evidence,
        derived_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="modular", limit=1),
        supported=True,
        items=[
            MemoryEvidence(
                source_event_id=root.event_id,
                event_type=root.event_type,
                source=root.source,
                created_at=root.created_at,
                conversation_id=root.conversation_id,
                conversation_seq=root.conversation_seq,
                global_seq=root.global_seq,
                content=root.payload["text"],
            )
        ],
    )
    review = SelfSchemaReview(
        opposition_indices=(0,),
        rationale="The model incorrectly selected the support root as contrary.",
    )

    with pytest.raises(ValueError, match="support root as opposition"):
        apply_review(
            conn, representation=representation, review=review,
            review_packet=packet,
            resolved_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        )
    assert current_self_resolution(conn, representation.representation_id).status is SelfResolutionStatus.CANDIDATE
    stored = self_evidence(conn, representation.representation_id)
    assert len(stored) == 1
    assert stored[0].relation.value == "SUPPORTS"


def test_review_memory_packet_removes_candidate_support_roots(conn, monkeypatch):
    support, _ = _record_user_prompt(conn, "I prefer modular systems.")
    related, _ = _record_user_prompt(conn, "I prefer reversible systems.")
    proposal = SelfSchemaProposal(
        kind=SelfRepresentationKind.PREFERENCE,
        perspective=SelfPerspective.AVOWED,
        statement="I prefer modular systems.",
        identity_centrality=IdentityCentrality.MODERATE,
        support_indices=(0,),
    )
    evidence = (
        ReflectionEvidence(
            event_id=support.event_id,
            event_type=support.event_type,
            source=support.source,
            created_at=support.created_at.isoformat(),
            content=support.payload["text"],
            context_refs=("situation:test",),
        ),
    )
    representation, _ = materialize_proposal(
        conn,
        proposal=proposal,
        evidence=evidence,
        derived_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    raw_packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text=representation.statement, limit=10),
        supported=True,
        items=[
            MemoryEvidence(
                source_event_id=support.event_id,
                event_type=support.event_type,
                source=support.source,
                created_at=support.created_at,
                conversation_id=support.conversation_id,
                conversation_seq=support.conversation_seq,
                global_seq=support.global_seq,
                content=support.payload["text"],
            ),
            MemoryEvidence(
                source_event_id=related.event_id,
                event_type=related.event_type,
                source=related.source,
                created_at=related.created_at,
                conversation_id=related.conversation_id,
                conversation_seq=related.conversation_seq,
                global_seq=related.global_seq,
                content=related.payload["text"],
            ),
        ],
        retrieval_trace={"fixture": True},
    )

    monkeypatch.setattr(
        "prometheist.self_reflection.jit_memory.request_memory",
        lambda *args, **kwargs: raw_packet,
    )

    packet = review_memory_packet(
        conn,
        representation=representation,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        before_global_seq=related.global_seq + 1,
    )

    assert [item.source_event_id for item in packet.items] == [related.event_id]
    assert packet.supported is True
    assert packet.retrieval_trace["self_review_support_roots_excluded"] == [
        str(support.event_id)
    ]


def test_review_opposition_index_is_application_validated(conn):
    root, _ = _record_user_prompt(conn, "I prefer modular systems.")
    proposal = SelfSchemaProposal(
        kind=SelfRepresentationKind.PREFERENCE,
        perspective=SelfPerspective.AVOWED,
        statement="I prefer modular systems.",
        identity_centrality=IdentityCentrality.MODERATE,
        support_indices=(0,),
    )
    evidence = (
        ReflectionEvidence(
            event_id=root.event_id,
            event_type=root.event_type,
            source=root.source,
            created_at=root.created_at.isoformat(),
            content=root.payload["text"],
            context_refs=("situation:test",),
        ),
    )
    representation, _ = materialize_proposal(
        conn,
        proposal=proposal,
        evidence=evidence,
        derived_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="modular", limit=1),
        supported=True,
        items=[
            MemoryEvidence(
                source_event_id=root.event_id,
                event_type=root.event_type,
                source=root.source,
                created_at=root.created_at,
                conversation_id=root.conversation_id,
                conversation_seq=root.conversation_seq,
                global_seq=root.global_seq,
                content=root.payload["text"],
            )
        ],
    )
    review = SelfSchemaReview(
        opposition_indices=(1,),
        rationale="Contrary evidence was claimed outside the supplied packet.",
    )

    with pytest.raises(ValueError, match="unavailable opposition"):
        apply_review(
            conn,
            representation=representation,
            review=review,
            review_packet=packet,
            resolved_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        )


def test_materialized_self_subject_is_fixed(conn):
    root, _ = _record_user_prompt(conn, "I want to become an AI engineer.")
    proposal = SelfSchemaProposal(
        kind=SelfRepresentationKind.PROSPECTIVE_SELF,
        perspective=SelfPerspective.ASPIRATIONAL,
        statement="I want to become an AI engineer.",
        identity_centrality=IdentityCentrality.CENTRAL,
        future_orientation=FutureOrientation.DESIRED,
        support_indices=(0,),
    )
    evidence = (
        ReflectionEvidence(
            event_id=root.event_id,
            event_type=root.event_type,
            source=root.source,
            created_at=root.created_at.isoformat(),
            content=root.payload["text"],
            context_refs=("situation:career",),
        ),
    )
    representation, _ = materialize_proposal(
        conn,
        proposal=proposal,
        evidence=evidence,
        derived_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert representation.subject == SELF_SUBJECT
