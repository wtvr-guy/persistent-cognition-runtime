"""Layered, provenance-grounded self memory for Prometheist.

The self model is a graph of immutable representations plus append-only evidence
and resolution records. It is not a mutable profile and never replaces canonical
life history. Direct evidence always terminates in canonical event roots; derived
representations may reorganize those roots but cannot manufacture new evidence.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID, uuid5

import psycopg
from pydantic import Field, field_validator, model_validator

from prometheist import event_store
from prometheist.cognitive_store import (
    COGNITIVE_NAMESPACE,
    get_record,
    list_records,
    put_record,
    record_history,
    record_lock,
)
from prometheist.models import EventType
from prometheist.memory_kernel import tokenize
from prometheist.percept_context import FrozenRecord, Reference, Scalar, aware
from prometheist.response_policy import HistoricalEvidenceScope, ResponseSurfaceMode

SELF_REPRESENTATION_KIND = "self_representation"
SELF_EVIDENCE_KIND = "self_evidence"
SELF_RESOLUTION_KIND = "self_resolution"
SELF_EDGE_KIND = "self_edge"
SELF_PREDICTION_KIND = "self_prediction"
WORKING_SELF_KIND = "working_self"

SELF_MEMORY_POLICY = "self-memory/v1"
SELF_SUBJECT = "self"
MAX_SELF_CONTEXT_ITEMS = 8
MAX_CORE_SELF_ITEMS = 2
MAX_SELF_CONTEXT_TAGS = 8
MAX_SELF_STATEMENT_CHARS = 2048
MAX_WORKING_SELF_REFS = 16
MIN_SELF_CONTEXT_ITEMS = 1
MIN_GENERALIZED_SUPPORT_ROOTS = 2
MIN_SLOW_SCHEMA_CONTEXTS = 2

SelfStatement = Annotated[
    str,
    Field(min_length=1, max_length=MAX_SELF_STATEMENT_CHARS),
]


class SelfRepresentationKind(str, Enum):
    ROLE = "ROLE"
    PREFERENCE = "PREFERENCE"
    VALUE = "VALUE"
    TRAIT = "TRAIT"
    BEHAVIORAL_TENDENCY = "BEHAVIORAL_TENDENCY"
    DECISION_POLICY = "DECISION_POLICY"
    WORLDVIEW = "WORLDVIEW"
    SELF_CONCEPT = "SELF_CONCEPT"
    PROSPECTIVE_SELF = "PROSPECTIVE_SELF"
    RELATIONAL_SCHEMA = "RELATIONAL_SCHEMA"
    NARRATIVE_HYPOTHESIS = "NARRATIVE_HYPOTHESIS"
    PROCEDURAL_SELF = "PROCEDURAL_SELF"
    EMBODIMENT_STATE = "EMBODIMENT_STATE"


class SelfPerspective(str, Enum):
    AVOWED = "AVOWED"
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    ASPIRATIONAL = "ASPIRATIONAL"
    NORMATIVE = "NORMATIVE"
    SOCIAL_ATTRIBUTION = "SOCIAL_ATTRIBUTION"


class PlasticityClass(str, Enum):
    FAST = "FAST"
    MEDIUM = "MEDIUM"
    SLOW = "SLOW"
    VERY_SLOW = "VERY_SLOW"


class IdentityCentrality(str, Enum):
    PERIPHERAL = "PERIPHERAL"
    MODERATE = "MODERATE"
    CENTRAL = "CENTRAL"


def default_plasticity(kind: SelfRepresentationKind) -> PlasticityClass:
    """Application-owned learning timescale for each self-representation class."""

    if kind is SelfRepresentationKind.EMBODIMENT_STATE:
        return PlasticityClass.FAST
    if kind in {
        SelfRepresentationKind.ROLE,
        SelfRepresentationKind.PREFERENCE,
        SelfRepresentationKind.SELF_CONCEPT,
        SelfRepresentationKind.PROSPECTIVE_SELF,
        SelfRepresentationKind.RELATIONAL_SCHEMA,
        SelfRepresentationKind.PROCEDURAL_SELF,
    }:
        return PlasticityClass.MEDIUM
    if kind in {
        SelfRepresentationKind.VALUE,
        SelfRepresentationKind.TRAIT,
        SelfRepresentationKind.BEHAVIORAL_TENDENCY,
        SelfRepresentationKind.DECISION_POLICY,
        SelfRepresentationKind.WORLDVIEW,
    }:
        return PlasticityClass.SLOW
    if kind is SelfRepresentationKind.NARRATIVE_HYPOTHESIS:
        return PlasticityClass.VERY_SLOW
    raise ValueError(f"unsupported self representation kind: {kind.value}")


class SelfResolutionStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    ESTABLISHED = "ESTABLISHED"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


class SelfEvidenceRelation(str, Enum):
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"


class SelfEvidenceOrigin(str, Enum):
    DIRECT = "DIRECT"
    INHERITED = "INHERITED"
    PREDICTION_OUTCOME = "PREDICTION_OUTCOME"


class SelfEdgeRelation(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    GENERALIZES = "GENERALIZES"
    CONTEXTUALIZES = "CONTEXTUALIZES"
    EXCEPTION_TO = "EXCEPTION_TO"
    CHANGED_BY = "CHANGED_BY"
    PREDICTS = "PREDICTS"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"


class FutureOrientation(str, Enum):
    DESIRED = "DESIRED"
    EXPECTED = "EXPECTED"
    FEARED = "FEARED"


class PredictionOutcome(str, Enum):
    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    AMBIGUOUS = "AMBIGUOUS"


class SelfContextAdmission(str, Enum):
    PRIMARY_DERIVED_CONTEXT = "PRIMARY_DERIVED_CONTEXT"
    ROUTING_ONLY = "ROUTING_ONLY"


class SelfRepresentation(FrozenRecord):
    representation_id: UUID
    subject: Reference
    kind: SelfRepresentationKind
    perspective: SelfPerspective
    statement: SelfStatement
    context_tags: tuple[Reference, ...] = Field(
        default=(),
        max_length=MAX_SELF_CONTEXT_TAGS,
    )
    relationship_ref: Reference | None = None
    future_orientation: FutureOrientation | None = None
    procedure_ref: Reference | None = None
    embodiment_ref: Reference | None = None
    plasticity: PlasticityClass
    created_at: datetime
    policy_version: Reference = SELF_MEMORY_POLICY

    _created_aware = field_validator("created_at")(aware)

    @model_validator(mode="after")
    def typed_contract(self) -> "SelfRepresentation":
        if (
            self.kind is SelfRepresentationKind.PROSPECTIVE_SELF
            and self.future_orientation is None
        ):
            raise ValueError("prospective self requires future_orientation")
        if (
            self.kind is SelfRepresentationKind.RELATIONAL_SCHEMA
            and self.relationship_ref is None
        ):
            raise ValueError("relational schema requires relationship_ref")
        if (
            self.kind is SelfRepresentationKind.PROCEDURAL_SELF
            and self.procedure_ref is None
        ):
            raise ValueError("procedural self requires procedure_ref")
        if (
            self.kind is SelfRepresentationKind.EMBODIMENT_STATE
            and self.embodiment_ref is None
        ):
            raise ValueError("embodiment state requires embodiment_ref")
        return self


class SelfEvidence(FrozenRecord):
    evidence_id: UUID
    representation_id: UUID
    root_event_id: UUID
    relation: SelfEvidenceRelation
    origin: SelfEvidenceOrigin
    context_tags: tuple[Reference, ...] = Field(
        default=(),
        max_length=MAX_SELF_CONTEXT_TAGS,
    )
    observed_at: datetime
    known_at: datetime
    derivation_method: Reference

    _aware = field_validator("observed_at", "known_at")(aware)

    @model_validator(mode="after")
    def temporal_contract(self) -> "SelfEvidence":
        if self.known_at < self.observed_at:
            raise ValueError("known_at cannot precede observed_at")
        return self


class SelfEvidenceMetrics(FrozenRecord):
    support_root_count: int = Field(ge=0)
    opposition_root_count: int = Field(ge=0)
    source_type_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    context_count: int = Field(ge=0)
    prediction_confirmed_count: int = Field(default_factory=int, ge=0)
    prediction_contradicted_count: int = Field(default_factory=int, ge=0)
    prediction_ambiguous_count: int = Field(default_factory=int, ge=0)
    first_support_at: datetime | None = None
    last_support_at: datetime | None = None

    @field_validator("first_support_at", "last_support_at")
    @classmethod
    def optional_aware(cls, value: datetime | None) -> datetime | None:
        return aware(value) if value is not None else None


class SelfResolution(FrozenRecord):
    resolution_id: UUID
    representation_id: UUID
    status: SelfResolutionStatus
    identity_centrality: IdentityCentrality
    metrics: SelfEvidenceMetrics
    counterevidence_checked: bool
    resolved_at: datetime
    supersedes: UUID | None = None
    policy_version: Reference = SELF_MEMORY_POLICY

    _resolved_aware = field_validator("resolved_at")(aware)


class SelfEdge(FrozenRecord):
    edge_id: UUID
    source_representation_id: UUID
    target_representation_id: UUID
    relation: SelfEdgeRelation
    created_at: datetime
    derivation_method: Reference

    _created_aware = field_validator("created_at")(aware)

    @model_validator(mode="after")
    def no_self_loop(self) -> "SelfEdge":
        if self.source_representation_id == self.target_representation_id:
            raise ValueError("self-memory edges cannot self-loop")
        return self


class SelfPrediction(FrozenRecord):
    prediction_id: UUID
    representation_id: UUID
    statement: SelfStatement
    predicted_value: Scalar | None = None
    context_tags: tuple[Reference, ...] = Field(
        default=(),
        max_length=MAX_SELF_CONTEXT_TAGS,
    )
    created_at: datetime
    knowledge_cutoff_global_seq: int = Field(ge=0)
    outcome: PredictionOutcome | None = None
    outcome_event_id: UUID | None = None
    resolved_at: datetime | None = None

    _created_aware = field_validator("created_at")(aware)

    @field_validator("resolved_at")
    @classmethod
    def resolved_aware(cls, value: datetime | None) -> datetime | None:
        return aware(value) if value is not None else None

    @model_validator(mode="after")
    def outcome_contract(self) -> "SelfPrediction":
        complete = (
            self.outcome is not None
            and self.outcome_event_id is not None
            and self.resolved_at is not None
        )
        empty = (
            self.outcome is None
            and self.outcome_event_id is None
            and self.resolved_at is None
        )
        if not (complete or empty):
            raise ValueError("prediction outcome fields must be all present or all absent")
        return self


class SelfContextItem(FrozenRecord):
    representation_id: UUID
    kind: SelfRepresentationKind
    perspective: SelfPerspective
    statement: SelfStatement
    status: SelfResolutionStatus
    identity_centrality: IdentityCentrality
    context_tags: tuple[Reference, ...] = ()
    relationship_ref: Reference | None = None
    future_orientation: FutureOrientation | None = None
    metrics: SelfEvidenceMetrics
    support_event_types: tuple[EventType, ...] = ()


class SelfContextPacket(FrozenRecord):
    admission: SelfContextAdmission
    items: tuple[SelfContextItem, ...] = Field(
        default=(),
        max_length=MAX_SELF_CONTEXT_ITEMS,
    )


class WorkingSelf(FrozenRecord):
    interaction_id: UUID
    query_text: SelfStatement
    active_self: SelfContextPacket
    goal_refs: tuple[Reference, ...] = Field(default=(), max_length=MAX_WORKING_SELF_REFS)
    entity_refs: tuple[Reference, ...] = Field(default=(), max_length=MAX_WORKING_SELF_REFS)
    activated_at: datetime

    _activated_aware = field_validator("activated_at")(aware)


_SELF_EVIDENCE_ROOT_TYPES = frozenset(
    {
        EventType.USER_PROMPT,
        EventType.TOOL_RESULT,
        EventType.PERCEPT_OBSERVATION,
        EventType.SYSTEM_EVENT,
    }
)

_DIRECT_SELF_REPORT_PERSPECTIVES = frozenset(
    {
        SelfPerspective.AVOWED,
        SelfPerspective.ASPIRATIONAL,
        SelfPerspective.NORMATIVE,
    }
)

_GENERALIZED_KINDS = frozenset(
    {
        SelfRepresentationKind.VALUE,
        SelfRepresentationKind.TRAIT,
        SelfRepresentationKind.BEHAVIORAL_TENDENCY,
        SelfRepresentationKind.DECISION_POLICY,
        SelfRepresentationKind.NARRATIVE_HYPOTHESIS,
    }
)


def self_evidence_root_allowed(event_type: EventType) -> bool:
    return event_type in _SELF_EVIDENCE_ROOT_TYPES


def _normalized_statement(value: str) -> str:
    return " ".join(value.split()).casefold()


def representation_id_for(
    *,
    subject: str,
    kind: SelfRepresentationKind,
    perspective: SelfPerspective,
    statement: str,
    context_tags: tuple[str, ...] = (),
    relationship_ref: str | None = None,
    future_orientation: FutureOrientation | None = None,
    procedure_ref: str | None = None,
    embodiment_ref: str | None = None,
) -> UUID:
    identity = "|".join(
        (
            subject,
            kind.value,
            perspective.value,
            _normalized_statement(statement),
            ",".join(sorted(context_tags)),
            relationship_ref or "",
            future_orientation.value if future_orientation else "",
            procedure_ref or "",
            embodiment_ref or "",
        )
    )
    return uuid5(COGNITIVE_NAMESPACE, f"self-representation:{identity}")


def ensure_self_representation(
    conn: psycopg.Connection,
    *,
    subject: str,
    kind: SelfRepresentationKind,
    perspective: SelfPerspective,
    statement: str,
    plasticity: PlasticityClass,
    created_at: datetime,
    context_tags: tuple[str, ...] = (),
    relationship_ref: str | None = None,
    future_orientation: FutureOrientation | None = None,
    procedure_ref: str | None = None,
    embodiment_ref: str | None = None,
) -> SelfRepresentation:
    representation_id = representation_id_for(
        subject=subject,
        kind=kind,
        perspective=perspective,
        statement=statement,
        context_tags=context_tags,
        relationship_ref=relationship_ref,
        future_orientation=future_orientation,
        procedure_ref=procedure_ref,
        embodiment_ref=embodiment_ref,
    )
    key = str(representation_id)
    existing = get_record(conn, SELF_REPRESENTATION_KIND, key)
    expected = SelfRepresentation(
        representation_id=representation_id,
        subject=subject,
        kind=kind,
        perspective=perspective,
        statement=statement,
        context_tags=context_tags,
        relationship_ref=relationship_ref,
        future_orientation=future_orientation,
        procedure_ref=procedure_ref,
        embodiment_ref=embodiment_ref,
        plasticity=plasticity,
        created_at=created_at,
    )
    if existing is None:
        put_record(
            conn,
            SELF_REPRESENTATION_KIND,
            key,
            expected.model_dump(mode="json"),
            revision="1",
        )
        return expected
    stored = SelfRepresentation.model_validate(existing)
    if stored.model_copy(update={"created_at": created_at}) != expected:
        raise ValueError(f"conflicting immutable self representation: {representation_id}")
    return stored


def get_self_representation(
    conn: psycopg.Connection,
    representation_id: UUID,
) -> SelfRepresentation | None:
    value = get_record(conn, SELF_REPRESENTATION_KIND, str(representation_id))
    return SelfRepresentation.model_validate(value) if value is not None else None


def _root_event(conn: psycopg.Connection, root_event_id: UUID):
    event = event_store.get_event_by_id(conn, root_event_id)
    if event is None:
        raise ValueError(f"self evidence root event does not exist: {root_event_id}")
    if event.event_type not in _SELF_EVIDENCE_ROOT_TYPES:
        raise ValueError(
            f"event type cannot serve as independent self evidence: {event.event_type.value}"
        )
    return event


def record_self_evidence(
    conn: psycopg.Connection,
    *,
    representation: SelfRepresentation,
    root_event_id: UUID,
    relation: SelfEvidenceRelation,
    origin: SelfEvidenceOrigin,
    derivation_method: str,
    known_at: datetime,
    observed_at: datetime | None = None,
    context_tags: tuple[str, ...] = (),
) -> SelfEvidence:
    root = _root_event(conn, root_event_id)
    observed = observed_at or root.created_at
    aware(observed)
    aware(known_at)
    if (
        relation is SelfEvidenceRelation.SUPPORTS
        and representation.perspective in _DIRECT_SELF_REPORT_PERSPECTIVES
        and root.event_type is not EventType.USER_PROMPT
    ):
        raise ValueError(
            "direct self-report perspectives require USER_PROMPT support roots"
        )

    opposite_relation = (
        SelfEvidenceRelation.OPPOSES
        if relation is SelfEvidenceRelation.SUPPORTS
        else SelfEvidenceRelation.SUPPORTS
    )
    for existing_evidence in self_evidence(conn, representation.representation_id):
        if (
            existing_evidence.root_event_id == root_event_id
            and existing_evidence.relation is opposite_relation
        ):
            raise ValueError(
                "one canonical root cannot both support and oppose the same "
                f"self representation: {root_event_id}"
            )

    evidence_id = uuid5(
        COGNITIVE_NAMESPACE,
        (
            f"self-evidence:{representation.representation_id}:"
            f"{root_event_id}:{relation.value}:{origin.value}:{derivation_method}:"
            f"{','.join(sorted(context_tags))}"
        ),
    )
    key = f"{representation.representation_id}:{evidence_id}"
    expected = SelfEvidence(
        evidence_id=evidence_id,
        representation_id=representation.representation_id,
        root_event_id=root_event_id,
        relation=relation,
        origin=origin,
        context_tags=context_tags,
        observed_at=observed,
        known_at=known_at,
        derivation_method=derivation_method,
    )
    existing = get_record(conn, SELF_EVIDENCE_KIND, key)
    if existing is None:
        put_record(
            conn,
            SELF_EVIDENCE_KIND,
            key,
            expected.model_dump(mode="json"),
            revision="1",
        )
        return expected
    stored = SelfEvidence.model_validate(existing)
    if stored != expected:
        raise ValueError(f"conflicting immutable self evidence: {evidence_id}")
    return stored


def _all_records(
    conn: psycopg.Connection,
    kind: str,
) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    after_key = ""
    while True:
        page = list_records(conn, kind, after_key=after_key)
        if not page:
            return rows
        rows.extend(page)
        after_key = page[-1][0]


def self_evidence(
    conn: psycopg.Connection,
    representation_id: UUID,
) -> tuple[SelfEvidence, ...]:
    prefix = f"{representation_id}:"
    rows = conn.execute(
        """
        SELECT payload
        FROM cognitive_heads
        WHERE record_kind = %s
          AND starts_with(record_key, %s)
        ORDER BY record_key
        """,
        (SELF_EVIDENCE_KIND, prefix),
    ).fetchall()
    return tuple(SelfEvidence.model_validate(dict(row[0])) for row in rows)


def _evidence_metrics(
    conn: psycopg.Connection,
    representation_id: UUID,
) -> SelfEvidenceMetrics:
    evidence = self_evidence(conn, representation_id)
    support = {
        item.root_event_id: item
        for item in evidence
        if item.relation is SelfEvidenceRelation.SUPPORTS
    }
    opposition = {
        item.root_event_id: item
        for item in evidence
        if item.relation is SelfEvidenceRelation.OPPOSES
    }
    support_events = [_root_event(conn, root_id) for root_id in support]
    contexts = {
        tag
        for item in evidence
        for tag in item.context_tags
        if item.relation is SelfEvidenceRelation.SUPPORTS
    }
    support_times = [item.observed_at for item in support.values()]
    predictions = self_predictions(conn, representation_id)
    return SelfEvidenceMetrics(
        support_root_count=len(support),
        opposition_root_count=len(opposition),
        source_type_count=len({event.event_type for event in support_events}),
        source_count=len({event.source for event in support_events}),
        context_count=len(contexts),
        prediction_confirmed_count=sum(
            item.outcome is PredictionOutcome.CONFIRMED for item in predictions
        ),
        prediction_contradicted_count=sum(
            item.outcome is PredictionOutcome.CONTRADICTED for item in predictions
        ),
        prediction_ambiguous_count=sum(
            item.outcome is PredictionOutcome.AMBIGUOUS for item in predictions
        ),
        first_support_at=min(support_times) if support_times else None,
        last_support_at=max(support_times) if support_times else None,
    )


def current_self_resolution(
    conn: psycopg.Connection,
    representation_id: UUID,
) -> SelfResolution | None:
    value = get_record(conn, SELF_RESOLUTION_KIND, str(representation_id))
    return SelfResolution.model_validate(value) if value is not None else None


def self_resolution_history(
    conn: psycopg.Connection,
    representation_id: UUID,
) -> tuple[SelfResolution, ...]:
    return tuple(
        SelfResolution.model_validate(value)
        for value in record_history(
            conn,
            SELF_RESOLUTION_KIND,
            str(representation_id),
        )
    )


def resolve_self_representation(
    conn: psycopg.Connection,
    *,
    representation_id: UUID,
    requested_status: SelfResolutionStatus,
    identity_centrality: IdentityCentrality,
    counterevidence_checked: bool,
    resolved_at: datetime,
) -> SelfResolution:
    """Resolve one self representation under application-owned guardrails.

    Generalized inferred identity cannot become established from one episode.
    Slow identity structures require evidence across more than one context.
    Any known opposing canonical evidence forces CONTESTED rather than allowing
    a model-generated interpretation to explain the contradiction away.
    """

    aware(resolved_at)
    representation_data = get_record(
        conn,
        SELF_REPRESENTATION_KIND,
        str(representation_id),
    )
    if representation_data is None:
        raise KeyError(representation_id)
    representation = SelfRepresentation.model_validate(representation_data)

    with record_lock(conn, f"self-resolution:{representation_id}"):
        metrics = _evidence_metrics(conn, representation_id)
        status = requested_status
        if status is SelfResolutionStatus.ESTABLISHED:
            if not counterevidence_checked:
                raise ValueError(
                    "self representation cannot be established before counterevidence search"
                )
            if metrics.opposition_root_count:
                status = SelfResolutionStatus.CONTESTED
            elif not metrics.support_root_count:
                raise ValueError("self representation has no canonical support roots")
            elif (
                representation.kind in _GENERALIZED_KINDS
                and representation.perspective
                not in _DIRECT_SELF_REPORT_PERSPECTIVES
                and metrics.support_root_count < MIN_GENERALIZED_SUPPORT_ROOTS
            ):
                status = SelfResolutionStatus.CANDIDATE
            elif (
                representation.plasticity
                in {PlasticityClass.SLOW, PlasticityClass.VERY_SLOW}
                and representation.perspective
                not in _DIRECT_SELF_REPORT_PERSPECTIVES
                and metrics.context_count < MIN_SLOW_SCHEMA_CONTEXTS
            ):
                status = SelfResolutionStatus.CANDIDATE

        previous = current_self_resolution(conn, representation_id)
        state = (
            status,
            identity_centrality,
            metrics,
            counterevidence_checked,
        )
        if previous is not None:
            old_state = (
                previous.status,
                previous.identity_centrality,
                previous.metrics,
                previous.counterevidence_checked,
            )
            if state == old_state:
                return previous

        resolution_id = uuid5(
            COGNITIVE_NAMESPACE,
            (
                f"self-resolution:{representation_id}:{status.value}:"
                f"{identity_centrality.value}:{metrics.model_dump_json()}:"
                f"{counterevidence_checked}:{resolved_at.isoformat()}"
            ),
        )
        resolution = SelfResolution(
            resolution_id=resolution_id,
            representation_id=representation_id,
            status=status,
            identity_centrality=identity_centrality,
            metrics=metrics,
            counterevidence_checked=counterevidence_checked,
            resolved_at=resolved_at,
            supersedes=previous.resolution_id if previous else None,
        )
        put_record(
            conn,
            SELF_RESOLUTION_KIND,
            str(representation_id),
            resolution.model_dump(mode="json"),
            revision=str(resolution_id),
        )
        return resolution


def record_self_edge(
    conn: psycopg.Connection,
    *,
    source_representation_id: UUID,
    target_representation_id: UUID,
    relation: SelfEdgeRelation,
    created_at: datetime,
    derivation_method: str,
) -> SelfEdge:
    if get_record(
        conn,
        SELF_REPRESENTATION_KIND,
        str(source_representation_id),
    ) is None:
        raise KeyError(source_representation_id)
    if get_record(
        conn,
        SELF_REPRESENTATION_KIND,
        str(target_representation_id),
    ) is None:
        raise KeyError(target_representation_id)
    edge_id = uuid5(
        COGNITIVE_NAMESPACE,
        (
            f"self-edge:{source_representation_id}:"
            f"{relation.value}:{target_representation_id}"
        ),
    )
    edge = SelfEdge(
        edge_id=edge_id,
        source_representation_id=source_representation_id,
        target_representation_id=target_representation_id,
        relation=relation,
        created_at=created_at,
        derivation_method=derivation_method,
    )
    existing = get_record(conn, SELF_EDGE_KIND, str(edge_id))
    if existing is None:
        put_record(
            conn,
            SELF_EDGE_KIND,
            str(edge_id),
            edge.model_dump(mode="json"),
            revision="1",
        )
        return edge
    stored = SelfEdge.model_validate(existing)
    if stored != edge:
        raise ValueError(f"conflicting immutable self edge: {edge_id}")
    return stored


def inherit_parent_roots(
    conn: psycopg.Connection,
    *,
    child: SelfRepresentation,
    parent_representation_ids: tuple[UUID, ...],
    known_at: datetime,
    derivation_method: str,
) -> tuple[SelfEvidence, ...]:
    """Copy unique canonical roots, never derived nodes, into a child schema."""

    inherited: dict[UUID, SelfEvidence] = {}
    for parent_id in parent_representation_ids:
        record_self_edge(
            conn,
            source_representation_id=child.representation_id,
            target_representation_id=parent_id,
            relation=SelfEdgeRelation.DERIVED_FROM,
            created_at=known_at,
            derivation_method=derivation_method,
        )
        for item in self_evidence(conn, parent_id):
            if item.relation is not SelfEvidenceRelation.SUPPORTS:
                continue
            inherited[item.root_event_id] = record_self_evidence(
                conn,
                representation=child,
                root_event_id=item.root_event_id,
                relation=SelfEvidenceRelation.SUPPORTS,
                origin=SelfEvidenceOrigin.INHERITED,
                derivation_method=derivation_method,
                observed_at=item.observed_at,
                known_at=known_at,
                context_tags=item.context_tags,
            )
    return tuple(inherited.values())


def record_self_prediction(
    conn: psycopg.Connection,
    *,
    representation_id: UUID,
    statement: str,
    created_at: datetime,
    predicted_value: Scalar | None = None,
    context_tags: tuple[str, ...] = (),
    knowledge_cutoff_global_seq: int | None = None,
) -> SelfPrediction:
    if current_self_resolution(conn, representation_id) is None:
        raise ValueError("prediction requires a resolved self representation")
    if knowledge_cutoff_global_seq is None:
        row = conn.execute(
            "SELECT COALESCE(MAX(global_seq), 0) FROM events"
        ).fetchone()
        knowledge_cutoff_global_seq = int(row[0])
    prediction_id = uuid5(
        COGNITIVE_NAMESPACE,
        (
            f"self-prediction:{representation_id}:"
            f"{_normalized_statement(statement)}:{created_at.isoformat()}:"
            f"{knowledge_cutoff_global_seq}"
        ),
    )
    prediction = SelfPrediction(
        prediction_id=prediction_id,
        representation_id=representation_id,
        statement=statement,
        predicted_value=predicted_value,
        context_tags=context_tags,
        created_at=created_at,
        knowledge_cutoff_global_seq=knowledge_cutoff_global_seq,
    )
    key = f"{representation_id}:{prediction_id}"
    existing = get_record(conn, SELF_PREDICTION_KIND, key)
    if existing is None:
        put_record(
            conn,
            SELF_PREDICTION_KIND,
            key,
            prediction.model_dump(mode="json"),
            revision="open",
        )
        return prediction
    return SelfPrediction.model_validate(existing)


def self_predictions(
    conn: psycopg.Connection,
    representation_id: UUID,
) -> tuple[SelfPrediction, ...]:
    prefix = f"{representation_id}:"
    rows = conn.execute(
        """
        SELECT payload
        FROM cognitive_heads
        WHERE record_kind = %s
          AND starts_with(record_key, %s)
        ORDER BY record_key
        """,
        (SELF_PREDICTION_KIND, prefix),
    ).fetchall()
    return tuple(SelfPrediction.model_validate(dict(row[0])) for row in rows)


def resolve_self_prediction(
    conn: psycopg.Connection,
    *,
    representation_id: UUID,
    prediction_id: UUID,
    outcome: PredictionOutcome,
    outcome_event_id: UUID,
    resolved_at: datetime,
) -> SelfPrediction:
    root = _root_event(conn, outcome_event_id)
    key = f"{representation_id}:{prediction_id}"
    current_data = get_record(conn, SELF_PREDICTION_KIND, key)
    if current_data is None:
        raise KeyError(prediction_id)
    current = SelfPrediction.model_validate(current_data)
    if current.representation_id != representation_id:
        raise ValueError("prediction belongs to another self representation")
    if root.global_seq <= current.knowledge_cutoff_global_seq:
        raise ValueError(
            "prediction outcome existed at or before the prediction knowledge cutoff"
        )
    if current.outcome is not None:
        if (
            current.outcome is outcome
            and current.outcome_event_id == outcome_event_id
        ):
            return current
        raise ValueError("prediction already has a different outcome")
    resolved = current.model_copy(
        update={
            "outcome": outcome,
            "outcome_event_id": outcome_event_id,
            "resolved_at": resolved_at,
        }
    )
    put_record(
        conn,
        SELF_PREDICTION_KIND,
        key,
        resolved.model_dump(mode="json"),
        revision=f"outcome:{outcome_event_id}",
    )

    representation = get_self_representation(conn, representation_id)
    resolution = current_self_resolution(conn, representation_id)
    if representation is None or resolution is None:
        raise RuntimeError("prediction refers to unresolved self representation")
    if outcome is PredictionOutcome.CONFIRMED:
        relation = SelfEvidenceRelation.SUPPORTS
    elif outcome is PredictionOutcome.CONTRADICTED:
        relation = SelfEvidenceRelation.OPPOSES
    else:
        relation = None
    if relation is not None:
        record_self_evidence(
            conn,
            representation=representation,
            root_event_id=outcome_event_id,
            relation=relation,
            origin=SelfEvidenceOrigin.PREDICTION_OUTCOME,
            derivation_method="self-prediction/v1",
            observed_at=root.created_at,
            known_at=root.created_at,
            context_tags=current.context_tags,
        )
        resolve_self_representation(
            conn,
            representation_id=representation_id,
            requested_status=resolution.status,
            identity_centrality=resolution.identity_centrality,
            counterevidence_checked=resolution.counterevidence_checked,
            resolved_at=resolved_at,
        )
    return resolved


def _support_event_types(
    conn: psycopg.Connection,
    representation_id: UUID,
) -> tuple[EventType, ...]:
    types = {
        _root_event(conn, item.root_event_id).event_type
        for item in self_evidence(conn, representation_id)
        if item.relation is SelfEvidenceRelation.SUPPORTS
    }
    return tuple(sorted(types, key=lambda item: item.value))


def _context_item(
    conn: psycopg.Connection,
    representation: SelfRepresentation,
    resolution: SelfResolution,
) -> SelfContextItem:
    return SelfContextItem(
        representation_id=representation.representation_id,
        kind=representation.kind,
        perspective=representation.perspective,
        statement=representation.statement,
        status=resolution.status,
        identity_centrality=resolution.identity_centrality,
        context_tags=representation.context_tags,
        relationship_ref=representation.relationship_ref,
        future_orientation=representation.future_orientation,
        metrics=resolution.metrics,
        support_event_types=_support_event_types(
            conn,
            representation.representation_id,
        ),
    )


def _centrality_rank(value: IdentityCentrality) -> int:
    return {
        IdentityCentrality.PERIPHERAL: 0,
        IdentityCentrality.MODERATE: 1,
        IdentityCentrality.CENTRAL: 2,
    }[value]


def _search_self_candidates(
    conn: psycopg.Connection,
    query_text: str,
    *,
    entity_refs: tuple[str, ...],
    limit: int,
    include_core_fallback: bool,
) -> tuple[SelfContextItem, ...]:
    """Sparse activation over current self heads without scanning lifetime history."""

    # A question is a set of cues, not a phrase that every representation must
    # contain in full. plainto_tsquery joins its lexemes with AND, making even
    # an exact topical match disappear as soon as the question has other words.
    query_terms = sorted(set(tokenize(query_text)))
    lexical_query = " OR ".join(query_terms)
    lexical_rows = conn.execute(
        """
        SELECT r.payload, s.payload,
               ts_rank(
                   to_tsvector('english', coalesce(r.payload->>'statement', '')),
                   websearch_to_tsquery('english', %s)
               ) AS rank
        FROM cognitive_heads r
        JOIN cognitive_heads s
          ON s.record_kind = %s
         AND s.record_key = r.record_key
        WHERE r.record_kind = %s
          AND s.payload->>'status' IN ('ESTABLISHED', 'CONTESTED')
          AND to_tsvector(
                'english',
                coalesce(r.payload->>'statement', '')
              ) @@ websearch_to_tsquery('english', %s)
        ORDER BY rank DESC, r.record_key
        LIMIT %s
        """,
        (
            lexical_query,
            SELF_RESOLUTION_KIND,
            SELF_REPRESENTATION_KIND,
            lexical_query,
            limit,
        ),
    ).fetchall()

    candidates: dict[UUID, SelfContextItem] = {}
    for representation_data, resolution_data, _rank in lexical_rows:
        representation = SelfRepresentation.model_validate(dict(representation_data))
        resolution = SelfResolution.model_validate(dict(resolution_data))
        candidates[representation.representation_id] = _context_item(
            conn,
            representation,
            resolution,
        )

    if entity_refs and len(candidates) < limit:
        relationship_rows = conn.execute(
            """
            SELECT r.payload, s.payload
            FROM cognitive_heads r
            JOIN cognitive_heads s
              ON s.record_kind = %s
             AND s.record_key = r.record_key
            WHERE r.record_kind = %s
              AND s.payload->>'status' IN ('ESTABLISHED', 'CONTESTED')
              AND r.payload->>'relationship_ref' = ANY(%s)
            ORDER BY r.record_key
            LIMIT %s
            """,
            (
                SELF_RESOLUTION_KIND,
                SELF_REPRESENTATION_KIND,
                list(entity_refs),
                limit,
            ),
        ).fetchall()
        for representation_data, resolution_data in relationship_rows:
            representation = SelfRepresentation.model_validate(
                dict(representation_data)
            )
            if representation.representation_id in candidates:
                continue
            resolution = SelfResolution.model_validate(dict(resolution_data))
            candidates[representation.representation_id] = _context_item(
                conn,
                representation,
                resolution,
            )

    remaining = limit - len(candidates)
    core_limit = min(MAX_CORE_SELF_ITEMS, remaining) if include_core_fallback else 0
    if core_limit:
        central_rows = conn.execute(
            """
            SELECT r.payload, s.payload
            FROM cognitive_heads r
            JOIN cognitive_heads s
              ON s.record_kind = %s
             AND s.record_key = r.record_key
            WHERE r.record_kind = %s
              AND s.payload->>'status' = 'ESTABLISHED'
            ORDER BY
              CASE s.payload->>'identity_centrality'
                WHEN 'CENTRAL' THEN 2
                WHEN 'MODERATE' THEN 1
                ELSE 0
              END DESC,
              r.record_key
            LIMIT %s
            """,
            (
                SELF_RESOLUTION_KIND,
                SELF_REPRESENTATION_KIND,
                core_limit + len(candidates),
            ),
        ).fetchall()
        added = 0
        for representation_data, resolution_data in central_rows:
            representation = SelfRepresentation.model_validate(
                dict(representation_data)
            )
            if representation.representation_id in candidates:
                continue
            resolution = SelfResolution.model_validate(dict(resolution_data))
            candidates[representation.representation_id] = _context_item(
                conn,
                representation,
                resolution,
            )
            added += 1
            if added == core_limit:
                break

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -_centrality_rank(item.identity_centrality),
            item.status is SelfResolutionStatus.CONTESTED,
            item.kind.value,
            str(item.representation_id),
        ),
    )
    return tuple(ordered[:limit])


def self_context_admission(
    evidence_scope: HistoricalEvidenceScope,
    surface_mode: ResponseSurfaceMode,
) -> SelfContextAdmission:
    if (
        surface_mode is ResponseSurfaceMode.NATURAL_LANGUAGE
        and evidence_scope
        in {
            HistoricalEvidenceScope.DERIVED_INTERNAL,
            HistoricalEvidenceScope.SELF_MODEL,
            HistoricalEvidenceScope.GENERAL_OR_CURRENT,
        }
    ):
        return SelfContextAdmission.PRIMARY_DERIVED_CONTEXT
    return SelfContextAdmission.ROUTING_ONLY


def activate_self_context(
    conn: psycopg.Connection,
    *,
    query_text: str,
    evidence_scope: HistoricalEvidenceScope,
    surface_mode: ResponseSurfaceMode,
    entity_refs: tuple[str, ...] = (),
    limit: int = MAX_SELF_CONTEXT_ITEMS,
) -> SelfContextPacket:
    if not MIN_SELF_CONTEXT_ITEMS <= limit <= MAX_SELF_CONTEXT_ITEMS:
        raise ValueError("self-context limit exceeds bounded policy")
    return SelfContextPacket(
        admission=self_context_admission(evidence_scope, surface_mode),
        items=_search_self_candidates(
            conn,
            query_text,
            entity_refs=entity_refs,
            limit=limit,
            include_core_fallback=(
                surface_mode is ResponseSurfaceMode.NATURAL_LANGUAGE
                and evidence_scope
                in {
                    HistoricalEvidenceScope.SELF_MODEL,
                    HistoricalEvidenceScope.DERIVED_INTERNAL,
                }
            ),
        ),
    )


def persist_working_self(
    conn: psycopg.Connection,
    *,
    interaction_id: UUID,
    query_text: str,
    self_context: SelfContextPacket,
    activated_at: datetime,
    goal_refs: tuple[str, ...] = (),
    entity_refs: tuple[str, ...] = (),
) -> WorkingSelf:
    working = WorkingSelf(
        interaction_id=interaction_id,
        query_text=query_text,
        active_self=self_context,
        goal_refs=goal_refs,
        entity_refs=entity_refs,
        activated_at=activated_at,
    )
    put_record(
        conn,
        WORKING_SELF_KIND,
        str(interaction_id),
        working.model_dump(mode="json"),
        revision="1",
    )
    return working


def load_working_self(
    conn: psycopg.Connection,
    interaction_id: UUID,
) -> WorkingSelf | None:
    data = get_record(conn, WORKING_SELF_KIND, str(interaction_id))
    return WorkingSelf.model_validate(data) if data is not None else None


def self_context_supplemental_queries(
    packet: SelfContextPacket,
) -> list[str]:
    return [item.statement for item in packet.items]


def render_self_context(packet: SelfContextPacket | None) -> str:
    if packet is None or not packet.items:
        return ""
    lines = [
        "[Derived self-memory]",
        f"admission={packet.admission.value}",
        (
            "These are revisable derived self representations, not direct "
            "historical quotations or independent canonical evidence."
        ),
    ]
    for index, item in enumerate(packet.items):
        lines.extend(
            [
                f"{index}: kind={item.kind.value}",
                f"perspective={item.perspective.value}",
                f"status={item.status.value}",
                f"centrality={item.identity_centrality.value}",
                f"statement={item.statement}",
                (
                    "support="
                    f"{item.metrics.support_root_count} canonical roots; "
                    f"opposition={item.metrics.opposition_root_count}; "
                    f"contexts={item.metrics.context_count}"
                ),
                (
                    "support_event_types="
                    + ",".join(value.value for value in item.support_event_types)
                ),
            ]
        )
    return "\n".join(lines)


def self_context_event_refs(
    conn: psycopg.Connection,
    packet: SelfContextPacket | None,
) -> tuple[str, ...]:
    if packet is None:
        return ()
    roots: set[UUID] = set()
    for item in packet.items:
        for evidence in self_evidence(conn, item.representation_id):
            roots.add(evidence.root_event_id)
    return tuple(f"event:{value}" for value in sorted(roots, key=str))
