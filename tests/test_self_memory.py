from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from prometheist import db, event_store
from prometheist.models import EventType
from prometheist.response_policy import (
    HistoricalEvidenceScope,
    ResponseSurfaceMode,
)
from prometheist.self_memory import (
    FutureOrientation,
    IdentityCentrality,
    PlasticityClass,
    PredictionOutcome,
    SELF_SUBJECT,
    SelfContextAdmission,
    SelfEvidenceOrigin,
    SelfEvidenceRelation,
    SelfPerspective,
    SelfRepresentationKind,
    SelfResolutionStatus,
    activate_self_context,
    current_self_resolution,
    ensure_self_representation,
    inherit_parent_roots,
    load_working_self,
    persist_working_self,
    record_self_evidence,
    record_self_prediction,
    resolve_self_prediction,
    resolve_self_representation,
    self_evidence,
    self_predictions,
)
from prometheist.self_memory import _search_self_candidates


@pytest.fixture
def conn():
    with db.get_connection() as connection:
        yield connection


_BASE_TIME = datetime.now(timezone.utc) + timedelta(hours=1)


def _at(minutes: int) -> datetime:
    return _BASE_TIME + timedelta(minutes=minutes)


def _event(
    conn,
    text: str,
    *,
    event_type: EventType = EventType.USER_PROMPT,
    source: str = "user",
):
    conversation_id = uuid4()
    event_store.start_conversation(conn, conversation_id)
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=uuid4(),
        event_type=event_type,
        source=source,
        payload={"text": text},
        payload_text=text,
        event_id=uuid4(),
    )


def _representation(
    conn,
    *,
    kind=SelfRepresentationKind.PREFERENCE,
    perspective=SelfPerspective.AVOWED,
    statement="I prefer local-first systems.",
    plasticity=PlasticityClass.MEDIUM,
    context_tags=(),
    future_orientation=None,
):
    return ensure_self_representation(
        conn,
        subject=SELF_SUBJECT,
        kind=kind,
        perspective=perspective,
        statement=statement,
        plasticity=plasticity,
        created_at=_at(10),
        context_tags=context_tags,
        future_orientation=future_orientation,
    )


def _support(
    conn,
    representation,
    root,
    *,
    context_tags=(),
):
    return record_self_evidence(
        conn,
        representation=representation,
        root_event_id=root.event_id,
        relation=SelfEvidenceRelation.SUPPORTS,
        origin=SelfEvidenceOrigin.DIRECT,
        derivation_method="test/v1",
        known_at=root.created_at,
        context_tags=context_tags,
    )


def test_avowed_self_report_can_establish_from_one_user_root(conn):
    root = _event(conn, "I prefer local-first systems.")
    representation = _representation(conn)
    _support(conn, representation, root)

    resolution = resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.MODERATE,
        counterevidence_checked=True,
        resolved_at=_at(20),
    )

    assert resolution.status is SelfResolutionStatus.ESTABLISHED
    assert resolution.metrics.support_root_count == 1


def test_avowed_self_report_rejects_non_user_support_root(conn):
    root = _event(
        conn,
        "System inferred a preference.",
        event_type=EventType.SYSTEM_EVENT,
        source="system",
    )
    representation = _representation(conn)

    with pytest.raises(ValueError, match="USER_PROMPT"):
        _support(conn, representation, root)


def test_inferred_slow_value_requires_independent_cross_context_support(conn):
    representation = _representation(
        conn,
        kind=SelfRepresentationKind.VALUE,
        perspective=SelfPerspective.INFERRED,
        statement="Auditability strongly influences architectural choices.",
        plasticity=PlasticityClass.SLOW,
    )
    first = _event(conn, "I chose the design because I could audit it.")
    second = _event(conn, "I kept the reversible design even though it was slower.")

    _support(conn, representation, first, context_tags=("software",))
    one_root = resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.CENTRAL,
        counterevidence_checked=True,
        resolved_at=_at(20),
    )
    assert one_root.status is SelfResolutionStatus.CANDIDATE

    _support(conn, representation, second, context_tags=("work",))
    established = resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.CENTRAL,
        counterevidence_checked=True,
        resolved_at=_at(30),
    )
    assert established.status is SelfResolutionStatus.ESTABLISHED
    assert established.metrics.support_root_count == 2
    assert established.metrics.context_count == 2


def test_same_canonical_root_cannot_support_and_oppose_one_representation(conn):
    root = _event(conn, "I prefer local-first systems.")
    representation = _representation(conn)
    _support(conn, representation, root)

    with pytest.raises(
        ValueError,
        match="cannot both support and oppose",
    ):
        record_self_evidence(
            conn,
            representation=representation,
            root_event_id=root.event_id,
            relation=SelfEvidenceRelation.OPPOSES,
            origin=SelfEvidenceOrigin.DIRECT,
            derivation_method="test/v1",
            known_at=root.created_at,
        )


def test_opposition_forces_established_schema_to_contested(conn):
    representation = _representation(
        conn,
        kind=SelfRepresentationKind.TRAIT,
        perspective=SelfPerspective.INFERRED,
        statement="The person tends to avoid financial risk.",
        plasticity=PlasticityClass.SLOW,
    )
    first = _event(conn, "I chose the safer financial option.")
    second = _event(conn, "I rejected leverage because of the downside.")
    contrary = _event(conn, "I put the entire bankroll into one speculative bet.")
    _support(conn, representation, first, context_tags=("finance", "planning"))
    _support(conn, representation, second, context_tags=("finance", "risk"))
    resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.MODERATE,
        counterevidence_checked=True,
        resolved_at=_at(30),
    )
    record_self_evidence(
        conn,
        representation=representation,
        root_event_id=contrary.event_id,
        relation=SelfEvidenceRelation.OPPOSES,
        origin=SelfEvidenceOrigin.DIRECT,
        derivation_method="test/v1",
        known_at=contrary.created_at,
        context_tags=("finance",),
    )

    resolution = resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.MODERATE,
        counterevidence_checked=True,
        resolved_at=_at(40),
    )

    assert resolution.status is SelfResolutionStatus.CONTESTED
    assert resolution.metrics.opposition_root_count == 1


def test_derived_schema_does_not_multiply_shared_canonical_roots(conn):
    root_a = _event(conn, "I want systems I can inspect.")
    root_b = _event(conn, "I prefer tools I can run locally.")

    parent_a = _representation(
        conn,
        kind=SelfRepresentationKind.VALUE,
        perspective=SelfPerspective.AVOWED,
        statement="Auditability matters to me.",
    )
    parent_b = _representation(
        conn,
        kind=SelfRepresentationKind.VALUE,
        perspective=SelfPerspective.AVOWED,
        statement="Local control matters to me.",
    )
    _support(conn, parent_a, root_a)
    _support(conn, parent_b, root_a)
    _support(conn, parent_b, root_b)

    child = _representation(
        conn,
        kind=SelfRepresentationKind.NARRATIVE_HYPOTHESIS,
        perspective=SelfPerspective.INFERRED,
        statement="A recurring theme is technological autonomy.",
        plasticity=PlasticityClass.VERY_SLOW,
    )
    inherited = inherit_parent_roots(
        conn,
        child=child,
        parent_representation_ids=(
            parent_a.representation_id,
            parent_b.representation_id,
        ),
        known_at=_at(50),
        derivation_method="reflection/v1",
    )

    assert {item.root_event_id for item in inherited} == {
        root_a.event_id,
        root_b.event_id,
    }
    assert len(self_evidence(conn, child.representation_id)) == 2


def test_prediction_contradiction_challenges_established_schema(conn):
    root_a = _event(conn, "I usually choose the reversible option.")
    root_b = _event(conn, "I kept an escape hatch in another project.")
    representation = _representation(
        conn,
        kind=SelfRepresentationKind.DECISION_POLICY,
        perspective=SelfPerspective.INFERRED,
        statement="Under uncertainty, the person tends to prefer reversible choices.",
        plasticity=PlasticityClass.SLOW,
    )
    _support(conn, representation, root_a, context_tags=("software",))
    _support(conn, representation, root_b, context_tags=("planning",))
    resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.CENTRAL,
        counterevidence_checked=True,
        resolved_at=_at(20),
    )
    prediction = record_self_prediction(
        conn,
        representation_id=representation.representation_id,
        statement="The next comparable choice will preserve reversibility.",
        created_at=_at(30),
        context_tags=("software",),
    )
    outcome = _event(conn, "I chose the irreversible shortcut.")

    resolve_self_prediction(
        conn,
        representation_id=representation.representation_id,
        prediction_id=prediction.prediction_id,
        outcome=PredictionOutcome.CONTRADICTED,
        outcome_event_id=outcome.event_id,
        resolved_at=_at(40),
    )

    current = current_self_resolution(conn, representation.representation_id)
    assert current.status is SelfResolutionStatus.CONTESTED
    assert current.metrics.prediction_contradicted_count == 1
    assert self_predictions(conn, representation.representation_id)[0].outcome is (
        PredictionOutcome.CONTRADICTED
    )


def test_prediction_rejects_outcome_that_was_already_known(conn):
    root_a = _event(conn, "I usually preserve reversibility.")
    root_b = _event(conn, "I kept a fallback in another project.")
    representation = _representation(
        conn,
        kind=SelfRepresentationKind.DECISION_POLICY,
        perspective=SelfPerspective.INFERRED,
        statement="The person tends to preserve reversibility.",
        plasticity=PlasticityClass.SLOW,
    )
    _support(conn, representation, root_a, context_tags=("software",))
    _support(conn, representation, root_b, context_tags=("planning",))
    resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.CENTRAL,
        counterevidence_checked=True,
        resolved_at=_at(20),
    )

    already_known = _event(conn, "I chose the irreversible shortcut.")
    prediction = record_self_prediction(
        conn,
        representation_id=representation.representation_id,
        statement="The next comparable choice will preserve reversibility.",
        created_at=_at(30),
    )

    with pytest.raises(ValueError, match="knowledge cutoff"):
        resolve_self_prediction(
            conn,
            representation_id=representation.representation_id,
            prediction_id=prediction.prediction_id,
            outcome=PredictionOutcome.CONTRADICTED,
            outcome_event_id=already_known.event_id,
            resolved_at=_at(40),
        )


def test_prospective_self_is_distinct_from_current_self(conn):
    representation = _representation(
        conn,
        kind=SelfRepresentationKind.PROSPECTIVE_SELF,
        perspective=SelfPerspective.ASPIRATIONAL,
        statement="I want to become a full-time AI engineer.",
        plasticity=PlasticityClass.MEDIUM,
        future_orientation=FutureOrientation.DESIRED,
    )

    assert representation.kind is SelfRepresentationKind.PROSPECTIVE_SELF
    assert representation.future_orientation is FutureOrientation.DESIRED


def test_self_context_is_primary_only_when_policy_allows_derived_self(conn):
    root = _event(conn, "I prefer modular systems.")
    representation = _representation(
        conn,
        statement="The person prefers modular systems.",
    )
    _support(conn, representation, root)
    resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.CENTRAL,
        counterevidence_checked=True,
        resolved_at=_at(20),
    )

    modeled = activate_self_context(
        conn,
        query_text="What kinds of systems do I usually prefer?",
        evidence_scope=HistoricalEvidenceScope.SELF_MODEL,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )
    quoted = activate_self_context(
        conn,
        query_text="What exactly did I say about modular systems?",
        evidence_scope=HistoricalEvidenceScope.USER_AUTHORED,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )
    exact = activate_self_context(
        conn,
        query_text="Return my preference exactly.",
        evidence_scope=HistoricalEvidenceScope.SELF_MODEL,
        surface_mode=ResponseSurfaceMode.EXACT_SOURCE_SUBSTRING,
    )

    assert modeled.admission is SelfContextAdmission.PRIMARY_DERIVED_CONTEXT
    assert modeled.items[0].representation_id == representation.representation_id
    assert quoted.admission is SelfContextAdmission.ROUTING_ONLY
    assert exact.admission is SelfContextAdmission.ROUTING_ONLY


def test_same_self_representation_can_be_relearned_later(conn):
    first = ensure_self_representation(
        conn,
        subject=SELF_SUBJECT,
        kind=SelfRepresentationKind.PREFERENCE,
        perspective=SelfPerspective.AVOWED,
        statement="I prefer modular systems.",
        plasticity=PlasticityClass.MEDIUM,
        created_at=_at(10),
    )
    later = ensure_self_representation(
        conn,
        subject=SELF_SUBJECT,
        kind=SelfRepresentationKind.PREFERENCE,
        perspective=SelfPerspective.AVOWED,
        statement="I prefer modular systems.",
        plasticity=PlasticityClass.MEDIUM,
        created_at=_at(40),
    )

    assert later == first
    assert later.created_at == _at(10)


def test_general_current_questions_do_not_receive_unrelated_core_fallback(conn):
    root = _event(conn, "I strongly prefer modular systems.")
    representation = _representation(
        conn,
        statement="The person strongly prefers modular systems.",
    )
    _support(conn, representation, root)
    resolve_self_representation(
        conn,
        representation_id=representation.representation_id,
        requested_status=SelfResolutionStatus.ESTABLISHED,
        identity_centrality=IdentityCentrality.CENTRAL,
        counterevidence_checked=True,
        resolved_at=_at(20),
    )

    packet = activate_self_context(
        conn,
        query_text="What is the capital of France?",
        evidence_scope=HistoricalEvidenceScope.GENERAL_OR_CURRENT,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )

    assert packet.admission is SelfContextAdmission.PRIMARY_DERIVED_CONTEXT
    assert packet.items == ()


def test_self_search_uses_disjunctive_topic_cues_for_full_questions():
    class EmptySearch:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            self.calls.append((sql, params))
            return self

        def fetchall(self):
            return []

    conn = EmptySearch()
    assert _search_self_candidates(
        conn,
        "Do I like surprises? Give the answer I would consider accurate",
        entity_refs=(),
        limit=8,
        include_core_fallback=False,
    ) == ()
    sql, params = conn.calls[0]
    assert "websearch_to_tsquery('english', %s)" in sql
    assert params[0] == params[3]
    assert "surprise" in params[0].split(" OR ")
    assert " OR " in params[0]


def test_working_self_persists_bounded_activation(conn):
    packet = activate_self_context(
        conn,
        query_text="unrelated query",
        evidence_scope=HistoricalEvidenceScope.SELF_MODEL,
        surface_mode=ResponseSurfaceMode.NATURAL_LANGUAGE,
    )
    interaction_id = uuid4()
    working = persist_working_self(
        conn,
        interaction_id=interaction_id,
        query_text="unrelated query",
        self_context=packet,
        activated_at=_at(20),
        goal_refs=("goal:ship",),
        entity_refs=("person:alice",),
    )

    assert load_working_self(conn, interaction_id) == working
