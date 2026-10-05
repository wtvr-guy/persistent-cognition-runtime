"""Append-only semantic assertions, evidence, and belief resolutions.

Canonical events remain the evidence of what was actually observed. Derived
semantic memory deliberately separates three concerns that must not be
collapsed:

* SemanticAssertion: a claim about one subject/property.
* SemanticEvidence: one provenance-bearing observation supporting or opposing
  an assertion.
* SemanticResolution: Prometheist's current accepted/ambiguous/unknown
  conclusion.

This separation prevents late historical evidence from becoming the current
belief merely because it was written later, lets corroboration accumulate
without duplicating claims, preserves contradictions, and supports true
bi-temporal queries over real-world validity time and system knowledge time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from uuid import UUID, uuid5

import psycopg
from pydantic import Field, field_validator, model_validator

from prometheist.cognitive_store import (
    COGNITIVE_NAMESPACE,
    get_record,
    list_records_with_prefix,
    put_record,
    record_history,
    record_lock,
)
from prometheist.percept_context import FrozenRecord, Reference, Scalar, aware

ASSERTION_KIND = "semantic_assertion"
EVIDENCE_KIND = "semantic_evidence"
RESOLUTION_KIND = "semantic_resolution"
RESOLUTION_POLICY = "latest-supported-observation/v2"


class EvidenceRelation(str, Enum):
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"


class EvidenceSourceKind(str, Enum):
    PERCEPT = "PERCEPT"
    ACTION = "ACTION"
    EVENT = "EVENT"


class ResolutionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


def _aware_optional(value: datetime | None) -> datetime | None:
    return aware(value) if value is not None else None


class SemanticAssertion(FrozenRecord):
    assertion_id: UUID
    subject: Reference
    property: Reference
    value: Scalar
    unit: Reference | None = None
    claim_valid_from: datetime | None = None
    claim_valid_until: datetime | None = None

    _validity_aware = field_validator("claim_valid_from", "claim_valid_until")(
        _aware_optional
    )

    @model_validator(mode="after")
    def validity_interval(self) -> "SemanticAssertion":
        if (
            self.claim_valid_from is not None
            and self.claim_valid_until is not None
            and self.claim_valid_until <= self.claim_valid_from
        ):
            raise ValueError("claim validity interval must be positive")
        return self


class SemanticEvidence(FrozenRecord):
    evidence_id: UUID
    assertion_id: UUID
    subject: Reference
    property: Reference
    source_kind: EvidenceSourceKind
    source_id: UUID
    relation: EvidenceRelation
    observed_at: datetime
    known_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    derivation_method: Reference

    _aware = field_validator("observed_at", "known_at")(aware)

    @model_validator(mode="after")
    def temporal_order(self) -> "SemanticEvidence":
        if self.known_at < self.observed_at:
            raise ValueError("known_at cannot precede observed_at")
        return self


class SemanticResolution(FrozenRecord):
    resolution_id: UUID
    subject: Reference
    property: Reference
    status: ResolutionStatus
    candidate_assertion_ids: tuple[UUID, ...] = ()
    selected_assertion_id: UUID | None = None
    support_observed_at: datetime | None = None
    resolution_policy: Reference = RESOLUTION_POLICY
    resolved_at: datetime
    supersedes: UUID | None = None

    _aware = field_validator("resolved_at")(aware)
    _support_observed_aware = field_validator("support_observed_at")(_aware_optional)

    @model_validator(mode="after")
    def status_contract(self) -> "SemanticResolution":
        if self.status is ResolutionStatus.ACCEPTED:
            if self.selected_assertion_id is None:
                raise ValueError("ACCEPTED resolution requires selected_assertion_id")
            if self.selected_assertion_id not in self.candidate_assertion_ids:
                raise ValueError("selected assertion must be a candidate")
        elif self.selected_assertion_id is not None:
            raise ValueError("only ACCEPTED resolutions may select an assertion")
        return self


class SemanticResolutionDecision(FrozenRecord):
    subject: Reference
    property: Reference
    status: ResolutionStatus
    candidate_assertion_ids: tuple[UUID, ...] = ()
    selected_assertion_id: UUID | None = None
    support_observed_at: datetime | None = None
    valid_at: datetime
    known_at: datetime
    resolution_policy: Reference = RESOLUTION_POLICY

    _aware = field_validator("valid_at", "known_at")(aware)
    _support_observed_aware = field_validator("support_observed_at")(_aware_optional)


@dataclass(frozen=True, slots=True)
class SemanticUpdateResult:
    assertion: SemanticAssertion
    evidence: SemanticEvidence
    resolution: SemanticResolution
    assertion_created: bool
    evidence_created: bool
    resolution_created: bool


def semantic_key(subject: str, property: str) -> str:
    return str(uuid5(COGNITIVE_NAMESPACE, f"semantic:{subject}\x1f{property}"))


def _canonical_scalar(value: Scalar) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def assertion_id_for(
    *,
    subject: str,
    property: str,
    value: Scalar,
    unit: str | None = None,
    claim_valid_from: datetime | None = None,
    claim_valid_until: datetime | None = None,
) -> UUID:
    parts = (
        subject,
        property,
        _canonical_scalar(value),
        unit or "",
        claim_valid_from.isoformat() if claim_valid_from else "",
        claim_valid_until.isoformat() if claim_valid_until else "",
    )
    return uuid5(COGNITIVE_NAMESPACE, "semantic-assertion:" + "\x1f".join(parts))


def _assertion_key(subject: str, property: str, assertion_id: UUID) -> str:
    return f"{semantic_key(subject, property)}:{assertion_id}"


def _evidence_key(subject: str, property: str, evidence_id: UUID) -> str:
    return f"{semantic_key(subject, property)}:{evidence_id}"


def _paged_kind(
    conn: psycopg.Connection,
    kind: str,
    subject: str,
    property: str,
) -> list[dict]:
    prefix = f"{semantic_key(subject, property)}:"
    values: list[dict] = []
    after_key = ""
    while True:
        page = list_records_with_prefix(
            conn, kind, prefix, after_key=after_key
        )
        if not page:
            return values
        values.extend(data for _, data in page)
        after_key = page[-1][0]


def semantic_assertions(
    conn: psycopg.Connection, subject: str, property: str
) -> tuple[SemanticAssertion, ...]:
    values = (
        SemanticAssertion.model_validate(item)
        for item in _paged_kind(conn, ASSERTION_KIND, subject, property)
    )
    return tuple(sorted(values, key=lambda item: str(item.assertion_id)))


def semantic_evidence(
    conn: psycopg.Connection, subject: str, property: str
) -> tuple[SemanticEvidence, ...]:
    values = (
        SemanticEvidence.model_validate(item)
        for item in _paged_kind(conn, EVIDENCE_KIND, subject, property)
    )
    return tuple(sorted(values, key=lambda item: (item.known_at, str(item.evidence_id))))


def current_semantic_resolution(
    conn: psycopg.Connection, subject: str, property: str
) -> SemanticResolution | None:
    value = get_record(conn, RESOLUTION_KIND, semantic_key(subject, property))
    return SemanticResolution.model_validate(value) if value is not None else None


def semantic_resolution_history(
    conn: psycopg.Connection, subject: str, property: str
) -> tuple[SemanticResolution, ...]:
    return tuple(
        SemanticResolution.model_validate(item)
        for item in record_history(conn, RESOLUTION_KIND, semantic_key(subject, property))
    )


def _support_observed_at(evidence: SemanticEvidence) -> datetime:
    """Observation-time ordering for competing evidence.

    Claim validity answers whether an assertion applies at a represented
    real-world time. It must never double as evidence-recency ordering.
    """

    return evidence.observed_at


def _applies_at(
    assertion: SemanticAssertion,
    evidence: SemanticEvidence,
    *,
    valid_at: datetime,
) -> bool:
    if assertion.claim_valid_from is not None:
        if assertion.claim_valid_from > valid_at:
            return False
        if assertion.claim_valid_until is not None and valid_at >= assertion.claim_valid_until:
            return False
        return True
    return evidence.observed_at <= valid_at


def semantic_resolution_as_of(
    conn: psycopg.Connection,
    subject: str,
    property: str,
    *,
    valid_at: datetime,
    known_at: datetime,
) -> SemanticResolutionDecision:
    """Resolve under independent real-world and system-knowledge clocks.

    This is intentionally a historical/forensic query. Normal current-memory
    reads use the incrementally maintained SemanticResolution head and do not
    scan lifetime evidence.
    """

    aware(valid_at)
    aware(known_at)
    assertions = {
        item.assertion_id: item
        for item in semantic_assertions(conn, subject, property)
    }
    candidates: list[tuple[datetime, SemanticAssertion]] = []
    latest_opposition: dict[UUID, datetime] = {}

    for item in semantic_evidence(conn, subject, property):
        if item.known_at > known_at:
            continue
        assertion = assertions.get(item.assertion_id)
        if assertion is None or not _applies_at(assertion, item, valid_at=valid_at):
            continue
        support_observed_at = _support_observed_at(item)
        if item.relation is EvidenceRelation.OPPOSES:
            prior = latest_opposition.get(item.assertion_id)
            if prior is None or support_observed_at > prior:
                latest_opposition[item.assertion_id] = support_observed_at
            continue
        candidates.append((support_observed_at, assertion))

    if not candidates:
        return SemanticResolutionDecision(
            subject=subject,
            property=property,
            status=ResolutionStatus.UNKNOWN,
            valid_at=valid_at,
            known_at=known_at,
        )

    latest = max(item[0] for item in candidates)
    latest_assertions = {
        item.assertion_id: item
        for support_observed_at, item in candidates
        if support_observed_at == latest
    }
    ordered = tuple(sorted(latest_assertions.values(), key=lambda item: str(item.assertion_id)))
    ids = tuple(item.assertion_id for item in ordered)
    semantic_values = {(item.value, item.unit) for item in ordered}
    contested = any(
        latest_opposition.get(item.assertion_id, latest) >= latest
        for item in ordered
        if item.assertion_id in latest_opposition
    )

    if len(semantic_values) != 1 or contested:
        return SemanticResolutionDecision(
            subject=subject,
            property=property,
            status=ResolutionStatus.AMBIGUOUS,
            candidate_assertion_ids=ids,
            support_observed_at=latest,
            valid_at=valid_at,
            known_at=known_at,
        )

    selected = ordered[0]
    return SemanticResolutionDecision(
        subject=subject,
        property=property,
        status=ResolutionStatus.ACCEPTED,
        candidate_assertion_ids=ids,
        selected_assertion_id=selected.assertion_id,
        support_observed_at=latest,
        valid_at=valid_at,
        known_at=known_at,
    )


def _advance_resolution(
    current: SemanticResolution | None,
    *,
    current_assertion: SemanticAssertion | None,
    assertion: SemanticAssertion,
    evidence: SemanticEvidence,
    resolved_at: datetime,
) -> tuple[ResolutionStatus, tuple[UUID, ...], UUID | None, datetime | None]:
    support_observed_at = _support_observed_at(evidence)

    if assertion.claim_valid_from is not None and assertion.claim_valid_from > resolved_at:
        if current is None:
            return ResolutionStatus.UNKNOWN, (), None, None
        return (
            current.status,
            current.candidate_assertion_ids,
            current.selected_assertion_id,
            current.support_observed_at,
        )
    if assertion.claim_valid_until is not None and resolved_at >= assertion.claim_valid_until:
        if current is None:
            return ResolutionStatus.UNKNOWN, (), None, None
        return (
            current.status,
            current.candidate_assertion_ids,
            current.selected_assertion_id,
            current.support_observed_at,
        )

    if (
        evidence.relation is EvidenceRelation.SUPPORTS
        and current is not None
        and current.status is ResolutionStatus.ACCEPTED
        and current.selected_assertion_id == assertion.assertion_id
    ):
        if current.support_observed_at is None or support_observed_at > current.support_observed_at:
            return (
                current.status,
                current.candidate_assertion_ids,
                current.selected_assertion_id,
                support_observed_at,
            )
        return (
            current.status,
            current.candidate_assertion_ids,
            current.selected_assertion_id,
            current.support_observed_at,
        )

    if evidence.relation is EvidenceRelation.OPPOSES:
        if current is None or assertion.assertion_id not in current.candidate_assertion_ids:
            if current is None:
                return ResolutionStatus.UNKNOWN, (), None, None
            return (
                current.status,
                current.candidate_assertion_ids,
                current.selected_assertion_id,
                current.support_observed_at,
            )
        if current.support_observed_at is not None and support_observed_at < current.support_observed_at:
            return (
                current.status,
                current.candidate_assertion_ids,
                current.selected_assertion_id,
                current.support_observed_at,
            )
        return (
            ResolutionStatus.AMBIGUOUS,
            current.candidate_assertion_ids,
            None,
            support_observed_at,
        )

    if current is None or current.support_observed_at is None or support_observed_at > current.support_observed_at:
        return ResolutionStatus.ACCEPTED, (assertion.assertion_id,), assertion.assertion_id, support_observed_at

    if support_observed_at < current.support_observed_at:
        return (
            current.status,
            current.candidate_assertion_ids,
            current.selected_assertion_id,
            current.support_observed_at,
        )

    candidates = tuple(
        sorted(
            set((*current.candidate_assertion_ids, assertion.assertion_id)),
            key=str,
        )
    )
    if (
        current.status is ResolutionStatus.ACCEPTED
        and current_assertion is not None
        and (current_assertion.value, current_assertion.unit)
        == (assertion.value, assertion.unit)
    ):
        return (
            ResolutionStatus.ACCEPTED,
            candidates,
            current.selected_assertion_id,
            current.support_observed_at,
        )
    if assertion.assertion_id in current.candidate_assertion_ids:
        return (
            current.status,
            current.candidate_assertion_ids,
            current.selected_assertion_id,
            current.support_observed_at,
        )

    return ResolutionStatus.AMBIGUOUS, candidates, None, support_observed_at


def _persist_resolution(
    conn: psycopg.Connection,
    *,
    subject: str,
    property: str,
    status: ResolutionStatus,
    candidates: tuple[UUID, ...],
    selected: UUID | None,
    support_observed_at: datetime | None,
    resolved_at: datetime,
    trigger_evidence_id: UUID,
) -> tuple[SemanticResolution, bool]:
    current = current_semantic_resolution(conn, subject, property)
    state = (status, candidates, selected, support_observed_at)
    if current is not None:
        current_state = (
            current.status,
            current.candidate_assertion_ids,
            current.selected_assertion_id,
            current.support_observed_at,
        )
        if state == current_state:
            return current, False

    resolution_id = uuid5(
        COGNITIVE_NAMESPACE,
        (
            f"semantic-resolution:{semantic_key(subject, property)}:"
            f"{trigger_evidence_id}:{status.value}:"
            f"{','.join(str(item) for item in candidates)}:{selected or ''}:"
            f"{support_observed_at.isoformat() if support_observed_at else ''}"
        ),
    )
    resolution = SemanticResolution(
        resolution_id=resolution_id,
        subject=subject,
        property=property,
        status=status,
        candidate_assertion_ids=candidates,
        selected_assertion_id=selected,
        support_observed_at=support_observed_at,
        resolved_at=resolved_at,
        supersedes=current.resolution_id if current else None,
    )
    put_record(
        conn,
        RESOLUTION_KIND,
        semantic_key(subject, property),
        resolution.model_dump(mode="json"),
        revision=str(resolution_id),
    )
    return resolution, True


def record_semantic_evidence(
    conn: psycopg.Connection,
    *,
    subject: str,
    property: str,
    value: Scalar,
    source_id: UUID,
    observed_at: datetime,
    known_at: datetime,
    resolved_at: datetime,
    source_kind: EvidenceSourceKind = EvidenceSourceKind.PERCEPT,
    confidence: float,
    derivation_method: str,
    unit: str | None = None,
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
    claim_valid_from: datetime | None = None,
    claim_valid_until: datetime | None = None,
) -> SemanticUpdateResult:
    """Record one observation and advance current resolution atomically."""

    aware(observed_at)
    aware(known_at)
    aware(resolved_at)
    _aware_optional(claim_valid_from)
    _aware_optional(claim_valid_until)
    if known_at < observed_at:
        raise ValueError("known_at cannot precede observed_at")
    if resolved_at < known_at:
        raise ValueError("resolved_at cannot precede known_at")

    key = semantic_key(subject, property)
    with record_lock(conn, f"semantic:{key}"):
        assertion_id = assertion_id_for(
            subject=subject,
            property=property,
            value=value,
            unit=unit,
            claim_valid_from=claim_valid_from,
            claim_valid_until=claim_valid_until,
        )
        assertion_key = _assertion_key(subject, property, assertion_id)
        stored_assertion = get_record(conn, ASSERTION_KIND, assertion_key)
        assertion_created = stored_assertion is None
        if stored_assertion is None:
            assertion = SemanticAssertion(
                assertion_id=assertion_id,
                subject=subject,
                property=property,
                value=value,
                unit=unit,
                claim_valid_from=claim_valid_from,
                claim_valid_until=claim_valid_until,
            )
            put_record(
                conn,
                ASSERTION_KIND,
                assertion_key,
                assertion.model_dump(mode="json"),
                revision="1",
            )
        else:
            assertion = SemanticAssertion.model_validate(stored_assertion)

        evidence_id = uuid5(
            COGNITIVE_NAMESPACE,
            (
                f"semantic-evidence:{assertion_id}:{source_kind.value}:{source_id}:"
                f"{relation.value}:{derivation_method}"
            ),
        )
        evidence_key = _evidence_key(subject, property, evidence_id)
        stored_evidence = get_record(conn, EVIDENCE_KIND, evidence_key)
        evidence_created = stored_evidence is None
        expected = SemanticEvidence(
            evidence_id=evidence_id,
            assertion_id=assertion_id,
            subject=subject,
            property=property,
            source_kind=source_kind,
            source_id=source_id,
            relation=relation,
            observed_at=observed_at,
            known_at=known_at,
            confidence=confidence,
            derivation_method=derivation_method,
        )
        if stored_evidence is None:
            evidence = expected
            put_record(
                conn,
                EVIDENCE_KIND,
                evidence_key,
                evidence.model_dump(mode="json"),
                revision="1",
            )
        else:
            evidence = SemanticEvidence.model_validate(stored_evidence)
            if evidence != expected:
                raise ValueError(
                    f"conflicting immutable semantic evidence retry: {evidence_id}"
                )

        current = current_semantic_resolution(conn, subject, property)
        current_assertion = None
        if current is not None and current.selected_assertion_id is not None:
            current_data = get_record(
                conn,
                ASSERTION_KIND,
                _assertion_key(
                    subject,
                    property,
                    current.selected_assertion_id,
                ),
            )
            if current_data is None:
                raise ValueError(
                    "semantic resolution references a missing selected assertion"
                )
            current_assertion = SemanticAssertion.model_validate(current_data)
        status, candidates, selected, support_observed_at = _advance_resolution(
            current,
            current_assertion=current_assertion,
            assertion=assertion,
            evidence=evidence,
            resolved_at=resolved_at,
        )
        resolution, resolution_created = _persist_resolution(
            conn,
            subject=subject,
            property=property,
            status=status,
            candidates=candidates,
            selected=selected,
            support_observed_at=support_observed_at,
            resolved_at=resolved_at,
            trigger_evidence_id=evidence_id,
        )
        return SemanticUpdateResult(
            assertion=assertion,
            evidence=evidence,
            resolution=resolution,
            assertion_created=assertion_created,
            evidence_created=evidence_created,
            resolution_created=resolution_created,
        )


def selected_assertion(
    conn: psycopg.Connection,
    resolution: SemanticResolution | SemanticResolutionDecision,
) -> SemanticAssertion | None:
    if resolution.selected_assertion_id is None:
        return None
    for item in semantic_assertions(conn, resolution.subject, resolution.property):
        if item.assertion_id == resolution.selected_assertion_id:
            return item
    raise ValueError(
        f"resolution references missing assertion: {resolution.selected_assertion_id}"
    )
