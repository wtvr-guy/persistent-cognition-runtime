"""Deterministic contention-preemption primitives for JIT Attention."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, model_validator

from prometheist.attention_resources import (
    ExecutionResourceClass,
    execution_resource_class_sort_key,
)


PREEMPTION_POLICY_VERSION = "v0.7-e-contention-v1"
_PREEMPTION_NAMESPACE = UUID("c3c215e5-d960-42c7-8b07-e6b8607575bc")


class PreemptionEventType(str, Enum):
    REQUESTED = "REQUESTED"
    CHECKPOINT_ACKNOWLEDGED = "CHECKPOINT_ACKNOWLEDGED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"


class PendingPreemption(BaseModel):
    """Durable replacement intent waiting on one or more safe checkpoints."""

    preemption_id: UUID
    target_task_id: UUID
    target_task_revision: int = Field(ge=0)
    victim_task_ids: list[UUID]
    victim_assignment_ids: list[UUID]
    checkpoint_victim_task_ids: list[UUID]
    checkpointed_victim_task_ids: list[UUID] = Field(default_factory=list)
    requested_epoch_sequence: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_intent(self) -> "PendingPreemption":
        victims = set(self.victim_task_ids)
        checkpoint_victims = set(self.checkpoint_victim_task_ids)
        checkpointed = set(self.checkpointed_victim_task_ids)
        if not victims:
            raise ValueError("A pending preemption must identify at least one victim")
        if len(victims) != len(self.victim_task_ids):
            raise ValueError("victim_task_ids must not contain duplicates")
        if len(self.victim_assignment_ids) != len(set(self.victim_assignment_ids)):
            raise ValueError("victim_assignment_ids must not contain duplicates")
        if len(self.victim_assignment_ids) != len(self.victim_task_ids):
            raise ValueError("Every victim task must have one victim assignment id")
        if self.target_task_id in victims:
            raise ValueError("A preemption target cannot also be its victim")
        if not checkpoint_victims:
            raise ValueError("Pending preemption requires a checkpoint-only victim")
        if len(checkpoint_victims) != len(self.checkpoint_victim_task_ids):
            raise ValueError("checkpoint_victim_task_ids must not contain duplicates")
        if not checkpoint_victims.issubset(victims):
            raise ValueError("Checkpoint victims must be selected victims")
        if len(checkpointed) != len(self.checkpointed_victim_task_ids):
            raise ValueError("checkpointed_victim_task_ids must not contain duplicates")
        if not checkpointed.issubset(checkpoint_victims):
            raise ValueError("Only checkpoint victims may acknowledge a checkpoint")
        return self

    @property
    def ready_to_execute(self) -> bool:
        return set(self.checkpointed_victim_task_ids) == set(
            self.checkpoint_victim_task_ids
        )


class PreemptionEvent(BaseModel):
    """Append-only causal record for one contention-preemption lifecycle step."""

    event_id: UUID
    preemption_id: UUID
    event_type: PreemptionEventType
    scheduler_cycle: int = Field(ge=0)
    epoch_sequence: int = Field(ge=0)
    target_task_id: UUID
    victim_task_ids: list[UUID]
    victim_assignment_ids: list[UUID]
    checkpoint_task_id: UUID | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_event(self) -> "PreemptionEvent":
        if not self.victim_task_ids:
            raise ValueError("A preemption event must identify at least one victim")
        if len(self.victim_task_ids) != len(set(self.victim_task_ids)):
            raise ValueError("victim_task_ids must not contain duplicates")
        if len(self.victim_assignment_ids) != len(set(self.victim_assignment_ids)):
            raise ValueError("victim_assignment_ids must not contain duplicates")
        if len(self.victim_task_ids) != len(self.victim_assignment_ids):
            raise ValueError("Every victim task must have one victim assignment id")
        if self.target_task_id in self.victim_task_ids:
            raise ValueError("A preemption target cannot also be its victim")
        if self.event_type is PreemptionEventType.CHECKPOINT_ACKNOWLEDGED:
            if self.checkpoint_task_id not in self.victim_task_ids:
                raise ValueError("Checkpoint event must identify a selected victim")
        elif self.checkpoint_task_id is not None:
            raise ValueError("Only checkpoint events identify a checkpoint task")
        return self


@dataclass(frozen=True)
class PreemptionCandidate:
    """One assignment eligible to release capacity for a blocked task."""

    task_id: UUID
    assignment_id: UUID
    base_priority: int
    checkpoint_required: bool
    current_order: int
    released_units: dict[ExecutionResourceClass, int]


def select_minimum_victims(
    *,
    deficits: dict[ExecutionResourceClass, int],
    candidates: list[PreemptionCandidate],
) -> list[PreemptionCandidate] | None:
    """Select the minimum victim count, then apply one deterministic total order.

    The dynamic program caps coverage at each actual deficit. Its state space is
    therefore bounded by the blocked task's explicit resource requirement, not
    by host timing or worker races.
    """

    if any(units < 0 for units in deficits.values()):
        raise ValueError("Resource deficits must not be negative")
    if len({candidate.task_id for candidate in candidates}) != len(candidates):
        raise ValueError("Preemption candidates must have unique task ids")
    if len({candidate.assignment_id for candidate in candidates}) != len(candidates):
        raise ValueError("Preemption candidates must have unique assignment ids")
    if any(
        units < 0
        for candidate in candidates
        for units in candidate.released_units.values()
    ):
        raise ValueError("Candidate released units must not be negative")

    required_classes = sorted(
        (resource_class for resource_class, units in deficits.items() if units > 0),
        key=execution_resource_class_sort_key,
    )
    if not required_classes:
        return []

    ordered = sorted(candidates, key=_candidate_sort_key)
    target = tuple(deficits[resource_class] for resource_class in required_classes)
    states: dict[tuple[int, ...], tuple[int, ...]] = {
        tuple(0 for _ in required_classes): ()
    }
    for index, candidate in enumerate(ordered):
        contribution = tuple(
            candidate.released_units.get(resource_class, 0)
            for resource_class in required_classes
        )
        if not any(contribution):
            continue
        updated = dict(states)
        for coverage, selected in states.items():
            next_coverage = tuple(
                min(target[position], coverage[position] + contribution[position])
                for position in range(len(target))
            )
            next_selected = selected + (index,)
            existing = updated.get(next_coverage)
            if existing is None or _selection_key(next_selected) < _selection_key(
                existing
            ):
                updated[next_coverage] = next_selected
        states = updated

    selected = states.get(target)
    if selected is None:
        return None
    return [ordered[index] for index in selected]


def deterministic_preemption_id(
    *,
    target_task_id: UUID,
    target_task_revision: int,
    victim_assignment_ids: list[UUID],
    requested_epoch_sequence: int,
) -> UUID:
    if target_task_revision < 0:
        raise ValueError("target_task_revision must be >= 0")
    if requested_epoch_sequence < 1:
        raise ValueError("requested_epoch_sequence must be >= 1")
    if not victim_assignment_ids:
        raise ValueError("At least one victim assignment id is required")
    if len(victim_assignment_ids) != len(set(victim_assignment_ids)):
        raise ValueError("victim_assignment_ids must not contain duplicates")
    victims = ",".join(str(value) for value in victim_assignment_ids)
    return uuid5(
        _PREEMPTION_NAMESPACE,
        "|".join(
            [
                str(target_task_id),
                str(target_task_revision),
                victims,
                str(requested_epoch_sequence),
            ]
        ),
    )


def make_preemption_event(
    intent: PendingPreemption,
    *,
    event_type: PreemptionEventType,
    scheduler_cycle: int,
    epoch_sequence: int,
    reason: str,
    checkpoint_task_id: UUID | None = None,
) -> PreemptionEvent:
    return make_selection_preemption_event(
        preemption_id=intent.preemption_id,
        target_task_id=intent.target_task_id,
        victim_task_ids=intent.victim_task_ids,
        victim_assignment_ids=intent.victim_assignment_ids,
        event_type=event_type,
        scheduler_cycle=scheduler_cycle,
        epoch_sequence=epoch_sequence,
        reason=reason,
        checkpoint_task_id=checkpoint_task_id,
    )


def make_selection_preemption_event(
    *,
    preemption_id: UUID,
    target_task_id: UUID,
    victim_task_ids: list[UUID],
    victim_assignment_ids: list[UUID],
    event_type: PreemptionEventType,
    scheduler_cycle: int,
    epoch_sequence: int,
    reason: str,
    checkpoint_task_id: UUID | None = None,
) -> PreemptionEvent:
    checkpoint = "none" if checkpoint_task_id is None else str(checkpoint_task_id)
    event_id = uuid5(
        preemption_id,
        f"preemption-event:{event_type.value}:{checkpoint}",
    )
    return PreemptionEvent(
        event_id=event_id,
        preemption_id=preemption_id,
        event_type=event_type,
        scheduler_cycle=scheduler_cycle,
        epoch_sequence=epoch_sequence,
        target_task_id=target_task_id,
        victim_task_ids=list(victim_task_ids),
        victim_assignment_ids=list(victim_assignment_ids),
        checkpoint_task_id=checkpoint_task_id,
        reason=reason,
    )


def pending_preemption_sort_key(intent: PendingPreemption) -> tuple[int, str]:
    return (intent.requested_epoch_sequence, intent.preemption_id.hex)


def pending_preemption_snapshot_key(intent: PendingPreemption) -> str:
    return intent.model_dump_json()


def _candidate_sort_key(
    candidate: PreemptionCandidate,
) -> tuple[int, int, int, str]:
    # Lowest importance first, then work that can yield immediately, then the
    # newest assignment so established same-priority work keeps its identity.
    return (
        -candidate.base_priority,
        1 if candidate.checkpoint_required else 0,
        -candidate.current_order,
        candidate.task_id.hex,
    )


def _selection_key(selected: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    return (len(selected), selected)
