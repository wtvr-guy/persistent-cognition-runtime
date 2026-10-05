"""Durable scheduling-epoch and assignment primitives for JIT Attention."""
from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, model_validator

from prometheist.attention_resources import (
    ResourceReservation,
    resource_reservation_sort_key,
)
from prometheist.attention_observation import ResourceObservationSnapshot
from prometheist.attention_preemption import (
    PendingPreemption,
    PreemptionEvent,
    PreemptionEventType,
    pending_preemption_snapshot_key,
    pending_preemption_sort_key,
)


ASSIGNMENT_POLICY_VERSION = "v0.7-d-epoch-v1"
_EPOCH_NAMESPACE = UUID("d82c9bb5-1c09-4cc6-a28d-ff46aa1ca7cd")


class AssignmentStatus(str, Enum):
    """Increment D exposes only ready, committed assignments to workers."""

    READY = "READY"


class SchedulingEpochStatus(str, Enum):
    PLANNED = "PLANNED"
    COMMITTED = "COMMITTED"


class DurableAssignment(BaseModel):
    """Worker-visible entitlement to advance one durable task.

    Increment D deliberately stops before worker claims, leases, heartbeats, or
    effects. Those lifecycle fields belong to the durable worker protocol.
    """

    assignment_id: UUID
    task_id: UUID
    task_revision: int = Field(ge=0)
    created_epoch_sequence: int = Field(ge=1)
    reservation_ids: list[UUID] = Field(default_factory=list)
    status: AssignmentStatus = AssignmentStatus.READY

    @model_validator(mode="after")
    def validate_reservation_ids(self) -> "DurableAssignment":
        if len(self.reservation_ids) != len(set(self.reservation_ids)):
            raise ValueError("reservation_ids must not contain duplicates")
        return self


class SchedulingEpoch(BaseModel):
    """Complete immutable assignment/reservation decision for one epoch."""

    epoch_id: UUID
    sequence: int = Field(ge=1)
    previous_epoch_id: UUID | None = None
    scheduler_cycle: int = Field(ge=0)
    admission_policy_version: str = Field(min_length=1)
    assignment_policy_version: str = Field(
        default=ASSIGNMENT_POLICY_VERSION,
        min_length=1,
    )
    status: SchedulingEpochStatus
    assignment_ids: list[UUID] = Field(default_factory=list)
    reservations: list[ResourceReservation] = Field(default_factory=list)
    preemption_policy_version: str | None = None
    pending_preemptions: list[PendingPreemption] = Field(default_factory=list)
    preemption_event_ids: list[UUID] = Field(default_factory=list)
    executed_preemption_ids: list[UUID] = Field(default_factory=list)
    cancelled_preemption_ids: list[UUID] = Field(default_factory=list)
    resource_observation_id: UUID | None = None
    resource_safety_policy_version: str | None = None

    @model_validator(mode="after")
    def normalize_complete_set(self) -> "SchedulingEpoch":
        if len(self.assignment_ids) != len(set(self.assignment_ids)):
            raise ValueError("assignment_ids must not contain duplicates")
        reservation_ids = [item.reservation_id for item in self.reservations]
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError("epoch reservations must not contain duplicate ids")
        self.reservations = sorted(
            self.reservations,
            key=resource_reservation_sort_key,
        )
        if self.preemption_policy_version is not None and not (
            self.preemption_policy_version.strip()
        ):
            raise ValueError("preemption_policy_version must not be empty")
        if self.preemption_policy_version is None and any(
            (
                self.pending_preemptions,
                self.preemption_event_ids,
                self.executed_preemption_ids,
                self.cancelled_preemption_ids,
            )
        ):
            raise ValueError("Preemption epoch data requires a policy version")
        self.pending_preemptions = sorted(
            self.pending_preemptions,
            key=pending_preemption_sort_key,
        )
        for field_name, identifiers in (
            ("preemption_event_ids", self.preemption_event_ids),
            ("executed_preemption_ids", self.executed_preemption_ids),
            ("cancelled_preemption_ids", self.cancelled_preemption_ids),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if set(self.executed_preemption_ids).intersection(
            self.cancelled_preemption_ids
        ):
            raise ValueError("An epoch cannot execute and cancel one preemption")
        if (self.resource_observation_id is None) != (
            self.resource_safety_policy_version is None
        ):
            raise ValueError(
                "Resource observation id and safety policy version must appear together"
            )
        if (
            self.resource_safety_policy_version is not None
            and not self.resource_safety_policy_version.strip()
        ):
            raise ValueError("resource_safety_policy_version must not be empty")
        return self


class SchedulingEpochPlan(BaseModel):
    """Provisional epoch that is not worker-visible until durability commits."""

    epoch: SchedulingEpoch
    admitted_task_ids: list[UUID] = Field(default_factory=list)
    unadmitted_task_ids: list[UUID] = Field(default_factory=list)
    assignments: list[DurableAssignment] = Field(default_factory=list)
    pending_preemptions: list[PendingPreemption] = Field(default_factory=list)
    preemption_events: list[PreemptionEvent] = Field(default_factory=list)
    resource_observation: ResourceObservationSnapshot | None = None

    @model_validator(mode="after")
    def validate_complete_plan(self) -> "SchedulingEpochPlan":
        if self.epoch.status is not SchedulingEpochStatus.PLANNED:
            raise ValueError("A scheduling epoch plan must be PLANNED")

        admitted = set(self.admitted_task_ids)
        unadmitted = set(self.unadmitted_task_ids)
        if len(admitted) != len(self.admitted_task_ids):
            raise ValueError("admitted_task_ids must not contain duplicates")
        if len(unadmitted) != len(self.unadmitted_task_ids):
            raise ValueError("unadmitted_task_ids must not contain duplicates")
        if admitted.intersection(unadmitted):
            raise ValueError("admitted and unadmitted task ids must be disjoint")
        if any(
            reservation.task_id not in admitted
            for reservation in self.epoch.reservations
        ):
            raise ValueError("every epoch reservation must belong to an admitted task")

        assignment_ids = [item.assignment_id for item in self.assignments]
        assignment_task_ids = [item.task_id for item in self.assignments]
        if assignment_ids != self.epoch.assignment_ids:
            raise ValueError("epoch assignment_ids must match assignments in order")
        if assignment_task_ids != self.admitted_task_ids:
            raise ValueError("one assignment must exist for every admitted task in order")

        reservations_by_task: dict[UUID, set[UUID]] = {}
        for reservation in self.epoch.reservations:
            reservations_by_task.setdefault(reservation.task_id, set()).add(
                reservation.reservation_id
            )
        for assignment in self.assignments:
            if set(assignment.reservation_ids) != reservations_by_task.get(
                assignment.task_id,
                set(),
            ):
                raise ValueError(
                    "assignment reservation_ids must match its complete reservation set"
                )

        normalized_pending = sorted(
            self.pending_preemptions,
            key=pending_preemption_sort_key,
        )
        if [pending_preemption_snapshot_key(value) for value in normalized_pending] != [
            pending_preemption_snapshot_key(value)
            for value in self.epoch.pending_preemptions
        ]:
            raise ValueError("epoch pending preemptions must match the plan snapshot")
        event_ids = [event.event_id for event in self.preemption_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("preemption_events must not contain duplicate ids")
        if event_ids != self.epoch.preemption_event_ids:
            raise ValueError("epoch preemption event ids must match events in order")
        executed_ids = [
            event.preemption_id
            for event in self.preemption_events
            if event.event_type is PreemptionEventType.EXECUTED
        ]
        cancelled_ids = [
            event.preemption_id
            for event in self.preemption_events
            if event.event_type is PreemptionEventType.CANCELLED
        ]
        if executed_ids != self.epoch.executed_preemption_ids:
            raise ValueError("epoch executed preemption ids must match its events")
        if cancelled_ids != self.epoch.cancelled_preemption_ids:
            raise ValueError("epoch cancelled preemption ids must match its events")

        pending_targets: set[UUID] = set()
        pending_victims: set[UUID] = set()
        for intent in normalized_pending:
            if intent.target_task_id in pending_targets:
                raise ValueError("A task may be the target of only one pending preemption")
            if pending_victims.intersection(intent.victim_task_ids):
                raise ValueError("A victim may belong to only one pending preemption")
            pending_targets.add(intent.target_task_id)
            pending_victims.update(intent.victim_task_ids)
        if pending_targets.intersection(pending_victims):
            raise ValueError("Pending targets and victims must be disjoint")
        if not pending_targets.issubset(unadmitted):
            raise ValueError("Pending preemption targets must remain unadmitted")
        if not pending_victims.issubset(admitted):
            raise ValueError("Pending preemption victims must remain admitted")
        if self.epoch.resource_observation_id is None:
            if self.resource_observation is not None:
                raise ValueError("Unreferenced resource observation in epoch plan")
        elif (
            self.resource_observation is None
            or self.resource_observation.observation_id
            != self.epoch.resource_observation_id
            or self.resource_observation.safety_policy_version
            != self.epoch.resource_safety_policy_version
            or self.resource_observation.scheduler_cycle
            != self.epoch.scheduler_cycle
        ):
            raise ValueError("Epoch must reference its exact resource observation")
        self.pending_preemptions = normalized_pending
        return self


def deterministic_assignment_id(
    task_id: UUID,
    *,
    task_revision: int,
    created_epoch_sequence: int,
    reservations: list[ResourceReservation],
) -> UUID:
    """Return a stable id for one concrete assignment generation.

    A task that remains assigned across replacement epochs keeps the same
    ``DurableAssignment`` object and therefore the same identifier.  If it is
    released and later readmitted, ``created_epoch_sequence`` distinguishes
    the new entitlement from the immutable historical assignment even when
    task revision and resource reservations are otherwise unchanged.
    """

    if task_revision < 0:
        raise ValueError("task_revision must be >= 0")
    if created_epoch_sequence < 1:
        raise ValueError("created_epoch_sequence must be >= 1")
    canonical_reservations = sorted(reservations, key=resource_reservation_sort_key)
    reservation_key = ",".join(
        f"{item.reservation_id}:{item.units}"
        for item in canonical_reservations
    )
    return uuid5(
        task_id,
        (
            "prometheist-assignment:"
            f"{task_revision}:{created_epoch_sequence}:{reservation_key}"
        ),
    )


def deterministic_epoch_id(
    *,
    sequence: int,
    previous_epoch_id: UUID | None,
    scheduler_cycle: int,
    admission_policy_version: str,
    assignment_policy_version: str,
    assignment_ids: list[UUID],
    reservations: list[ResourceReservation],
    preemption_policy_version: str | None = None,
    pending_preemptions: list[PendingPreemption] | None = None,
    preemption_event_ids: list[UUID] | None = None,
    executed_preemption_ids: list[UUID] | None = None,
    cancelled_preemption_ids: list[UUID] | None = None,
    resource_observation_id: UUID | None = None,
    resource_safety_policy_version: str | None = None,
) -> UUID:
    """Return the stable identity of one complete scheduling decision."""

    if sequence < 1:
        raise ValueError("sequence must be >= 1")
    if scheduler_cycle < 0:
        raise ValueError("scheduler_cycle must be >= 0")
    previous = "root" if previous_epoch_id is None else str(previous_epoch_id)
    assignment_key = ",".join(str(value) for value in assignment_ids)
    reservation_key = ",".join(
        f"{item.reservation_id}:{item.units}"
        for item in sorted(reservations, key=resource_reservation_sort_key)
    )
    canonical_parts = [
        str(sequence),
        previous,
        str(scheduler_cycle),
        admission_policy_version,
        assignment_policy_version,
        assignment_key,
        reservation_key,
    ]
    pending = sorted(
        pending_preemptions or [],
        key=pending_preemption_sort_key,
    )
    event_ids = preemption_event_ids or []
    executed_ids = executed_preemption_ids or []
    cancelled_ids = cancelled_preemption_ids or []
    if preemption_policy_version is None:
        if pending or event_ids or executed_ids or cancelled_ids:
            raise ValueError("Preemption epoch data requires a policy version")
    else:
        if not preemption_policy_version.strip():
            raise ValueError("preemption_policy_version must not be empty")
        canonical_parts.extend(
            [
                preemption_policy_version,
                ",".join(pending_preemption_snapshot_key(value) for value in pending),
                ",".join(str(value) for value in event_ids),
                ",".join(str(value) for value in executed_ids),
                ",".join(str(value) for value in cancelled_ids),
            ]
        )
    if (resource_observation_id is None) != (
        resource_safety_policy_version is None
    ):
        raise ValueError(
            "Resource observation id and safety policy version must appear together"
        )
    if resource_observation_id is not None:
        if not resource_safety_policy_version or not (
            resource_safety_policy_version.strip()
        ):
            raise ValueError("resource_safety_policy_version must not be empty")
        canonical_parts.extend(
            [str(resource_observation_id), resource_safety_policy_version]
        )
    canonical = "|".join(canonical_parts)
    return uuid5(_EPOCH_NAMESPACE, canonical)
