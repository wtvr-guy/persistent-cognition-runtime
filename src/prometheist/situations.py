"""Overlapping, bounded situation snapshots with provenance and prediction error."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid5

import psycopg
from pydantic import Field

from prometheist.cognitive_store import COGNITIVE_NAMESPACE, get_record, put_record, record_lock
from prometheist.expectations import Expectation, PredictionError, compare_expectation
from prometheist.percept_context import FrozenRecord, Observation
from prometheist.perception import Percept, SalienceAssessment, evaluate_salience

SITUATION_WINDOW = 64


class ObservedState(FrozenRecord):
    observation: Observation
    percept_id: UUID
    observed_at: datetime
    source_event_id: UUID | None = None


class SituationRelation(FrozenRecord):
    from_percept: UUID
    to_percept: UUID
    relation: Literal["BEFORE", "ADAPTER_REPORTED_CAUSE"]


class Situation(FrozenRecord):
    situation_id: UUID
    snapshot_id: UUID
    epistemic_status: Literal["DERIVED"] = "DERIVED"
    percept_ids: tuple[UUID, ...] = Field(max_length=SITUATION_WINDOW)
    entity_refs: tuple[str, ...] = Field(max_length=SITUATION_WINDOW)
    active_goal_refs: tuple[str, ...] = Field(max_length=SITUATION_WINDOW)
    temporal_relations: tuple[SituationRelation, ...] = Field(max_length=SITUATION_WINDOW)
    causal_relations: tuple[SituationRelation, ...] = Field(max_length=SITUATION_WINDOW)
    expected_state: tuple[Expectation, ...] = Field(max_length=SITUATION_WINDOW)
    observed_state: tuple[ObservedState, ...] = Field(max_length=SITUATION_WINDOW)
    prediction_errors: tuple[PredictionError, ...] = Field(max_length=SITUATION_WINDOW)
    prediction_error: float = Field(ge=0.0)
    salience: SalienceAssessment
    uncertainty: int = Field(ge=0, le=3)
    provenance: tuple[UUID, ...] = Field(max_length=SITUATION_WINDOW)
    created_at: datetime
    updated_at: datetime
    supersedes: UUID | None = None
    retention_hint: Literal["ROUTINE", "RECONCILE", "EPISODIC_PRIORITY"]


def _recent(values):
    return tuple(dict.fromkeys(values))[-SITUATION_WINDOW:]


def situation_keys(percept: Percept) -> tuple[str, ...]:
    entities = percept.context.entity_refs or tuple(
        dict.fromkeys(value.subject for value in percept.context.observations)
    )
    if entities:
        return tuple(f"entity:{value}" for value in entities)
    if percept.context.task_refs:
        return tuple(f"task:{value}" for value in percept.context.task_refs)
    return (f"correlation:{percept.correlation_id}",)


def form_situation(
    key: str, percept: Percept, expectations: tuple[Expectation, ...], previous: Situation | None,
) -> Situation:
    situation_id = uuid5(COGNITIVE_NAMESPACE, key)
    if previous is not None and previous.situation_id != situation_id:
        raise ValueError("situation identity mismatch")
    states = {(state.observation.subject, state.observation.property): state
              for state in previous.observed_state} if previous else {}
    expected = {value.expectation_id: value for value in previous.expected_state} if previous else {}
    errors = {(value.subject, value.property): value for value in previous.prediction_errors} if previous else {}
    for value in expectations:
        if value.supersedes:
            expected.pop(value.supersedes, None)
        expected[value.expectation_id] = value
    for observation in percept.context.observations:
        property_key = (observation.subject, observation.property)
        if property_key in states and states[property_key].observed_at > percept.observed_at:
            continue  # Late evidence remains canonical; it cannot replace a newer observation.
        states.pop(property_key, None)
        states[property_key] = ObservedState(
            observation=observation,
            percept_id=percept.percept_id,
            observed_at=percept.observed_at,
            source_event_id=percept.source_event_id,
        )
        errors.pop(property_key, None)
        # Choose the latest declared expectation for a property; confidence is
        # uncertainty evidence, not a license to call low-confidence values true.
        matching = [value for value in expected.values()
                    if (value.subject, value.property) == property_key]
        if matching:
            expectation = max(matching, key=lambda value: (value.valid_from, str(value.expectation_id)))
            errors[property_key] = compare_expectation(
                expectation, observation, percept_id=percept.percept_id, observed_at=percept.observed_at,
            )
    magnitude = max((value.magnitude * value.confidence for value in errors.values()
                     if value.magnitude is not None), default=0.0)
    goals = _recent((*previous.active_goal_refs, *percept.context.active_goal_refs)) if previous else percept.context.active_goal_refs
    # Score the aggregate discrepancy and goal relevance. Source assertions stay
    # attached to the latest percept; they never become action permissions.
    context = percept.context.model_copy(update={"active_goal_refs": goals[:16]})
    salience = evaluate_salience(percept.model_copy(update={"context": context}),
                                prediction_error=magnitude, novelty=previous is None)
    temporal = list(previous.temporal_relations) if previous else []
    if previous and previous.percept_ids:
        left, right = previous.percept_ids[-1], percept.percept_id
        if previous.updated_at > percept.observed_at:
            left, right = right, left
        temporal.append(SituationRelation(from_percept=left, to_percept=right, relation="BEFORE"))
    causal = list(previous.causal_relations) if previous else []
    causal.extend(SituationRelation(from_percept=value, to_percept=percept.percept_id,
                                    relation="ADAPTER_REPORTED_CAUSE")
                  for value in percept.context.causal_percept_refs)
    return Situation(
        situation_id=situation_id,
        snapshot_id=uuid5(situation_id, f"{previous.snapshot_id if previous else 'initial'}:{percept.percept_id}"),
        percept_ids=_recent((*(previous.percept_ids if previous else ()), percept.percept_id)),
        entity_refs=_recent((*(previous.entity_refs if previous else ()), *percept.context.entity_refs,
                             *(value.subject for value in percept.context.observations))),
        active_goal_refs=goals, temporal_relations=tuple(temporal[-SITUATION_WINDOW:]),
        causal_relations=tuple(causal[-SITUATION_WINDOW:]),
        expected_state=tuple(expected.values())[-SITUATION_WINDOW:],
        observed_state=tuple(states.values())[-SITUATION_WINDOW:],
        prediction_errors=tuple(errors.values())[-SITUATION_WINDOW:], prediction_error=magnitude,
        salience=salience, uncertainty=salience.signals.uncertainty_score,
        provenance=_recent((*(previous.provenance if previous else ()),
                            *((percept.source_event_id,) if percept.source_event_id else ()))),
        created_at=previous.created_at if previous else percept.observed_at,
        updated_at=max(previous.updated_at, percept.observed_at) if previous else percept.observed_at,
        supersedes=previous.snapshot_id if previous else None,
        retention_hint="EPISODIC_PRIORITY" if magnitude >= 1.0 else "RECONCILE" if magnitude > 0.0 else "ROUTINE",
    )


def persist_situations(conn: psycopg.Connection, percept: Percept) -> tuple[Situation, ...]:
    expectations = []
    for reference in percept.context.expectation_refs:
        data = get_record(conn, "expectation", str(reference))
        if data is None:
            raise ValueError(f"unknown expectation reference: {reference}")
        expectations.append(Expectation.model_validate(data))
    results = []
    for key in sorted(situation_keys(percept)):
        situation_id = uuid5(COGNITIVE_NAMESPACE, key)
        with record_lock(conn, f"situation:{situation_id}"):
            receipt_key = f"{situation_id}:{percept.percept_id}"
            receipt = get_record(conn, "situation_ingest", receipt_key)
            if receipt:
                results.append(Situation.model_validate(receipt))
                continue
            data = get_record(conn, "situation", str(situation_id))
            previous = Situation.model_validate(data) if data else None
            # Recover a crash after the snapshot but before its intake receipt.
            if previous and percept.percept_id == previous.percept_ids[-1]:
                situation = previous
            else:
                situation = form_situation(key, percept, tuple(expectations), previous)
            payload = situation.model_dump(mode="json")
            put_record(conn, "situation", str(situation_id), payload, revision=str(situation.snapshot_id))
            put_record(conn, "situation_ingest", receipt_key, payload, revision="1")
            results.append(situation)
    return tuple(results)


def register_expectation(conn: psycopg.Connection, expectation: Expectation) -> None:
    put_record(conn, "expectation", str(expectation.expectation_id),
               expectation.model_dump(mode="json"), revision="1")
