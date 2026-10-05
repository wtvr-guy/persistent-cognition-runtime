"""Deterministic JIT Attention scheduling primitives.

JIT Attention owns execution focus, not semantic interpretation.  Agents may
propose work, but this module derives scheduling priority from structured
metadata and applies deterministic queue, service-guarantee, dependency, and
interruption rules without an LLM in the scheduling loop.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, model_validator

from prometheist.attention_assignments import (
    ASSIGNMENT_POLICY_VERSION,
    AssignmentStatus,
    DurableAssignment,
    SchedulingEpoch,
    SchedulingEpochPlan,
    SchedulingEpochStatus,
    deterministic_assignment_id,
    deterministic_epoch_id,
)
from prometheist.attention_preemption import (
    PREEMPTION_POLICY_VERSION,
    PendingPreemption,
    PreemptionCandidate,
    PreemptionEvent,
    PreemptionEventType,
    deterministic_preemption_id,
    make_preemption_event,
    make_selection_preemption_event,
    pending_preemption_sort_key,
    select_minimum_victims,
)
from prometheist.attention_resources import (
    ExecutionResource,
    ExecutionResourceClass,
    ProcessResourceEstimate,
    RESOURCE_ADMISSION_POLICY_VERSION,
    ResourceEstimateSource,
    ResourceAdmissionPlan,
    ResourceRequirement,
    ResourceReservation,
    deterministic_reservation_id,
    execution_resource_class_sort_key,
    execution_resource_sort_key,
    resource_requirement_sort_key,
    resource_reservation_sort_key,
)
from prometheist.attention_observation import (
    RESOURCE_SAFETY_POLICY_VERSION,
    ResourceObservationError,
    ResourceObservationSnapshot,
    ResourceSafetyPolicy,
)


class PriorityClass(IntEnum):
    """Lower numeric values have higher scheduling priority."""

    P0 = 0  # emergency / integrity / safety
    P1 = 1  # user-blocking interactive work
    P2 = 2  # active user-requested work
    P3 = 3  # supporting work
    P4 = 4  # maintenance
    P5 = 5  # opportunistic background work


class TaskCriticality(str, Enum):
    EMERGENCY = "EMERGENCY"
    USER_BLOCKING = "USER_BLOCKING"
    USER_REQUESTED = "USER_REQUESTED"
    SUPPORTING = "SUPPORTING"
    MAINTENANCE = "MAINTENANCE"
    OPPORTUNISTIC = "OPPORTUNISTIC"


CRITICALITY_PRIORITY: dict[TaskCriticality, PriorityClass] = {
    TaskCriticality.EMERGENCY: PriorityClass.P0,
    TaskCriticality.USER_BLOCKING: PriorityClass.P1,
    TaskCriticality.USER_REQUESTED: PriorityClass.P2,
    TaskCriticality.SUPPORTING: PriorityClass.P3,
    TaskCriticality.MAINTENANCE: PriorityClass.P4,
    TaskCriticality.OPPORTUNISTIC: PriorityClass.P5,
}


class ServiceClass(str, Enum):
    INTERACTIVE = "INTERACTIVE"
    USER_WORK = "USER_WORK"
    SUPPORT = "SUPPORT"
    MAINTENANCE = "MAINTENANCE"
    BACKGROUND = "BACKGROUND"


class InterruptionPolicy(str, Enum):
    """How a running task may yield to higher-priority work."""

    PREEMPTIBLE = "PREEMPTIBLE"
    CHECKPOINT_ONLY = "CHECKPOINT_ONLY"
    ATOMIC = "ATOMIC"


class TaskStatus(str, Enum):
    NEW = "NEW"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FocusAction(str, Enum):
    IDLE = "IDLE"
    START = "START"
    CONTINUE = "CONTINUE"
    PREEMPT = "PREEMPT"
    WAIT_FOR_CHECKPOINT = "WAIT_FOR_CHECKPOINT"


class ServiceGuarantee(BaseModel):
    """Queue-cycle guarantee used to prevent deterministic starvation.

    Once ``max_wait_cycles`` is reached, a queued task is promoted to at least
    ``guaranteed_priority``.  Guarantees never override a running task's
    interruption policy; an ATOMIC task still runs to completion.
    """

    max_wait_cycles: int | None = Field(default=None, ge=0)
    guaranteed_priority: PriorityClass


DEFAULT_SERVICE_GUARANTEES: dict[ServiceClass, ServiceGuarantee] = {
    ServiceClass.INTERACTIVE: ServiceGuarantee(
        max_wait_cycles=1,
        guaranteed_priority=PriorityClass.P1,
    ),
    ServiceClass.USER_WORK: ServiceGuarantee(
        max_wait_cycles=4,
        guaranteed_priority=PriorityClass.P2,
    ),
    ServiceClass.SUPPORT: ServiceGuarantee(
        max_wait_cycles=8,
        guaranteed_priority=PriorityClass.P2,
    ),
    ServiceClass.MAINTENANCE: ServiceGuarantee(
        max_wait_cycles=32,
        guaranteed_priority=PriorityClass.P2,
    ),
    ServiceClass.BACKGROUND: ServiceGuarantee(
        max_wait_cycles=128,
        guaranteed_priority=PriorityClass.P3,
    ),
}


class SchedulingMetadata(BaseModel):
    """Structured inputs from which scheduling behavior is derived."""

    criticality: TaskCriticality
    service_class: ServiceClass
    interruption_policy: InterruptionPolicy = InterruptionPolicy.PREEMPTIBLE
    deadline: datetime | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    required_resource_classes: list[ExecutionResourceClass] = Field(default_factory=list)
    resource_requirements: list[ResourceRequirement] = Field(default_factory=list)
    process_resource_estimate: ProcessResourceEstimate | None = None
    dependency_ids: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_resource_contract(self) -> "SchedulingMetadata":
        requirements_by_class: dict[ExecutionResourceClass, ResourceRequirement] = {}
        for requirement in self.resource_requirements:
            if requirement.resource_class in requirements_by_class:
                raise ValueError("resource_requirements must contain each class at most once")
            requirements_by_class[requirement.resource_class] = requirement

        if self.process_resource_estimate is not None:
            for requirement in (
                self.process_resource_estimate.resource_requirements()
            ):
                existing = requirements_by_class.get(requirement.resource_class)
                if existing is not None and existing.units != requirement.units:
                    raise ValueError(
                        "process_resource_estimate conflicts with an explicit "
                        f"{requirement.resource_class.value} requirement"
                    )
                requirements_by_class[requirement.resource_class] = requirement

        required_classes = set(self.required_resource_classes)
        required_classes.update(requirements_by_class)
        if (
            self.process_resource_estimate is not None
            and ExecutionResourceClass.LLM_INFERENCE in required_classes
            and self.process_resource_estimate.llm_slots != 1
        ):
            raise ValueError(
                "an LLM-backed process estimate must reserve exactly one LLM slot"
            )
        self.required_resource_classes = sorted(
            required_classes,
            key=execution_resource_class_sort_key,
        )
        self.resource_requirements = sorted(
            requirements_by_class.values(),
            key=resource_requirement_sort_key,
        )
        return self

    def quantitative_resource_requirements(self) -> list[ResourceRequirement]:
        """Return the canonical quantitative contract, including v0.7-B inputs."""

        explicit = {
            requirement.resource_class: requirement
            for requirement in self.resource_requirements
        }
        return [
            explicit.get(
                resource_class,
                ResourceRequirement(resource_class=resource_class, units=1),
            ).model_copy(deep=True)
            for resource_class in self.required_resource_classes
        ]

    def conservative_process_estimate(
        self,
        *,
        policy: ResourceSafetyPolicy,
    ) -> ProcessResourceEstimate:
        """Derive the persisted fallback used until profiling data exists."""

        explicit = {
            requirement.resource_class: requirement.units
            for requirement in self.quantitative_resource_requirements()
        }
        llm_slots = explicit.get(ExecutionResourceClass.LLM_INFERENCE, 0)
        if llm_slots > 1:
            raise ValueError(
                "one bounded process estimate cannot request multiple LLM slots"
            )
        memory_default = (
            policy.default_llm_process_memory_mib
            if llm_slots
            else policy.default_process_memory_mib
        )
        return ProcessResourceEstimate(
            cpu_units=explicit.get(
                ExecutionResourceClass.CPU_GENERAL,
                policy.default_process_cpu_units,
            ),
            memory_mib=explicit.get(
                ExecutionResourceClass.MEMORY_RAM,
                memory_default,
            ),
            llm_slots=llm_slots,
            source=ResourceEstimateSource.CONSERVATIVE_DEFAULT,
            basis=(
                f"{policy.policy_version}: conservative fallback pending "
                "profiled or historical peak data"
            ),
        )


class AttentionTask(BaseModel):
    """Durable task representation; no LLM context is required to resume it."""

    task_id: UUID
    task_key: str = Field(min_length=1)
    created_seq: int = Field(ge=1)
    metadata: SchedulingMetadata
    parent_task_id: UUID | None = None
    status: TaskStatus = TaskStatus.NEW
    enqueued_cycle: int = Field(default=0, ge=0)
    revision: int = Field(default=0, ge=0)
    resumable_state: dict[str, Any] = Field(default_factory=dict)


class TaskTransition(BaseModel):
    transition_id: UUID
    task_id: UUID
    revision: int = Field(ge=1)
    scheduler_cycle: int = Field(ge=0)
    from_status: TaskStatus
    to_status: TaskStatus
    reason: str


class FocusDecision(BaseModel):
    action: FocusAction
    active_task_id: UUID | None = None
    candidate_task_id: UUID | None = None
    reason: str


class SchedulerSnapshot(BaseModel):
    cycle: int = Field(ge=0)
    active_task_id: UUID | None = None
    pending_preemption_task_id: UUID | None = None
    tasks: list[AttentionTask]
    execution_resources: list[ExecutionResource] = Field(default_factory=list)
    admission_policy_version: str = Field(
        default=RESOURCE_ADMISSION_POLICY_VERSION,
        min_length=1,
    )
    admitted_task_ids: list[UUID] = Field(default_factory=list)
    resource_reservations: list[ResourceReservation] = Field(default_factory=list)
    epoch_sequence: int = Field(default=0, ge=0)
    assignment_policy_version: str = Field(
        default=ASSIGNMENT_POLICY_VERSION,
        min_length=1,
    )
    preemption_policy_version: str = Field(
        default=PREEMPTION_POLICY_VERSION,
        min_length=1,
    )
    preemption_state_revision: int = Field(default=0, ge=0)
    pending_preemptions: list[PendingPreemption] = Field(default_factory=list)
    current_epoch: SchedulingEpoch | None = None
    assignments: list[DurableAssignment] = Field(default_factory=list)
    resource_safety_required: bool = False
    resource_safety_policy_version: str = Field(
        default=RESOURCE_SAFETY_POLICY_VERSION,
        min_length=1,
    )
    current_resource_observation: ResourceObservationSnapshot | None = None


def deterministic_task_id(namespace: UUID, task_key: str) -> UUID:
    """Return a stable task id for a stable namespace + task key."""

    normalized = task_key.strip()
    if not normalized:
        raise ValueError("task_key must not be empty")
    return uuid5(namespace, f"prometheist-task:{normalized}")


def derive_priority(metadata: SchedulingMetadata) -> PriorityClass:
    """Derive base priority solely from explicit structured metadata."""

    return CRITICALITY_PRIORITY[metadata.criticality]


def _normalize_deadline(deadline: datetime | None) -> datetime:
    if deadline is None:
        return datetime.max.replace(tzinfo=timezone.utc)
    if deadline.tzinfo is None:
        return deadline.replace(tzinfo=timezone.utc)
    return deadline.astimezone(timezone.utc)


class JITAttentionScheduler:
    """Deterministic executive scheduler for Prometheist task focus.

    The scheduler has no wall-clock dependency.  Service guarantees use the
    monotonic ``cycle`` counter so identical input/state produces identical
    decisions across machines and restarts.
    """

    def __init__(
        self,
        *,
        service_guarantees: dict[ServiceClass, ServiceGuarantee] | None = None,
        resource_safety_required: bool = False,
        resource_safety_policy_version: str = RESOURCE_SAFETY_POLICY_VERSION,
    ) -> None:
        configured = DEFAULT_SERVICE_GUARANTEES if service_guarantees is None else service_guarantees
        self.service_guarantees = {
            service_class: guarantee.model_copy(deep=True)
            for service_class, guarantee in configured.items()
        }
        missing = set(ServiceClass) - set(self.service_guarantees)
        if missing:
            missing_names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"Missing service guarantees for: {missing_names}")
        self.tasks: dict[UUID, AttentionTask] = {}
        self.resources: dict[str, ExecutionResource] = {}
        self.admission_policy_version = RESOURCE_ADMISSION_POLICY_VERSION
        self.assignment_policy_version = ASSIGNMENT_POLICY_VERSION
        self.preemption_policy_version = PREEMPTION_POLICY_VERSION
        self.preemption_state_revision = 0
        self._persisted_preemption_state_revision = 0
        self.admitted_task_ids: list[UUID] = []
        self._resource_reservations: dict[UUID, ResourceReservation] = {}
        self.epoch_sequence = 0
        self.current_epoch: SchedulingEpoch | None = None
        self._assignments: dict[UUID, DurableAssignment] = {}
        self._pending_preemptions: dict[UUID, PendingPreemption] = {}
        self._preemption_events: list[PreemptionEvent] = []
        self._pending_epoch_plan: SchedulingEpochPlan | None = None
        self.resource_safety_required = resource_safety_required
        normalized_safety_version = resource_safety_policy_version.strip()
        if not normalized_safety_version:
            raise ValueError("resource_safety_policy_version must not be empty")
        self.resource_safety_policy_version = normalized_safety_version
        self._current_resource_observation: ResourceObservationSnapshot | None = None
        self._pending_resource_observation: ResourceObservationSnapshot | None = None
        self.active_task_id: UUID | None = None
        self.pending_preemption_task_id: UUID | None = None
        self.cycle = 0
        self.transitions: list[TaskTransition] = []
        self._persisted_transition_count = 0

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SchedulerSnapshot,
        *,
        service_guarantees: dict[ServiceClass, ServiceGuarantee] | None = None,
    ) -> "JITAttentionScheduler":
        scheduler = cls(
            service_guarantees=service_guarantees,
            resource_safety_required=snapshot.resource_safety_required,
            resource_safety_policy_version=(
                snapshot.resource_safety_policy_version
            ),
        )
        task_ids = [task.task_id for task in snapshot.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Snapshot contains duplicate task ids")
        scheduler.tasks = {task.task_id: task.model_copy(deep=True) for task in snapshot.tasks}
        resource_ids = [resource.resource_id for resource in snapshot.execution_resources]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("Snapshot contains duplicate execution resource ids")
        scheduler.resources = {
            resource.resource_id: resource.model_copy(deep=True)
            for resource in snapshot.execution_resources
        }
        scheduler.admission_policy_version = snapshot.admission_policy_version
        admitted_task_ids = list(snapshot.admitted_task_ids)
        if len(admitted_task_ids) != len(set(admitted_task_ids)):
            raise ValueError("Snapshot contains duplicate admitted task ids")
        scheduler.admitted_task_ids = admitted_task_ids
        reservation_ids = [
            reservation.reservation_id
            for reservation in snapshot.resource_reservations
        ]
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError("Snapshot contains duplicate resource reservation ids")
        scheduler._resource_reservations = {
            reservation.reservation_id: reservation.model_copy(deep=True)
            for reservation in snapshot.resource_reservations
        }
        scheduler.epoch_sequence = snapshot.epoch_sequence
        scheduler.assignment_policy_version = snapshot.assignment_policy_version
        scheduler.preemption_policy_version = snapshot.preemption_policy_version
        scheduler.preemption_state_revision = snapshot.preemption_state_revision
        scheduler._persisted_preemption_state_revision = (
            snapshot.preemption_state_revision
        )
        scheduler.current_epoch = (
            snapshot.current_epoch.model_copy(deep=True)
            if snapshot.current_epoch is not None
            else None
        )
        assignment_ids = [assignment.assignment_id for assignment in snapshot.assignments]
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("Snapshot contains duplicate assignment ids")
        scheduler._assignments = {
            assignment.assignment_id: assignment.model_copy(deep=True)
            for assignment in snapshot.assignments
        }
        preemption_ids = [
            intent.preemption_id for intent in snapshot.pending_preemptions
        ]
        if len(preemption_ids) != len(set(preemption_ids)):
            raise ValueError("Snapshot contains duplicate pending preemption ids")
        scheduler._pending_preemptions = {
            intent.preemption_id: intent.model_copy(deep=True)
            for intent in snapshot.pending_preemptions
        }
        scheduler._current_resource_observation = (
            snapshot.current_resource_observation.model_copy(deep=True)
            if snapshot.current_resource_observation is not None
            else None
        )
        scheduler.active_task_id = snapshot.active_task_id
        scheduler.pending_preemption_task_id = snapshot.pending_preemption_task_id
        scheduler.cycle = snapshot.cycle
        scheduler._validate_snapshot()
        return scheduler

    @classmethod
    def for_local_host(
        cls,
        *,
        service_guarantees: dict[ServiceClass, ServiceGuarantee] | None = None,
        resource_safety_policy_version: str = RESOURCE_SAFETY_POLICY_VERSION,
    ) -> "JITAttentionScheduler":
        """Construct the host-aware scheduler used by real local execution."""

        return cls(
            service_guarantees=service_guarantees,
            resource_safety_required=True,
            resource_safety_policy_version=resource_safety_policy_version,
        )

    def snapshot(self) -> SchedulerSnapshot:
        self._validate_snapshot()
        return SchedulerSnapshot(
            cycle=self.cycle,
            active_task_id=self.active_task_id,
            pending_preemption_task_id=self.pending_preemption_task_id,
            tasks=[
                task.model_copy(deep=True)
                for task in sorted(self.tasks.values(), key=lambda item: (item.created_seq, item.task_id.hex))
            ],
            execution_resources=self.execution_resources(),
            admission_policy_version=self.admission_policy_version,
            admitted_task_ids=list(self.admitted_task_ids),
            resource_reservations=self.resource_reservations(),
            epoch_sequence=self.epoch_sequence,
            assignment_policy_version=self.assignment_policy_version,
            preemption_policy_version=self.preemption_policy_version,
            preemption_state_revision=self.preemption_state_revision,
            pending_preemptions=self.pending_preemptions(),
            current_epoch=(
                self.current_epoch.model_copy(deep=True)
                if self.current_epoch is not None
                else None
            ),
            assignments=self.worker_visible_assignments(),
            resource_safety_required=self.resource_safety_required,
            resource_safety_policy_version=self.resource_safety_policy_version,
            current_resource_observation=(
                self._current_resource_observation.model_copy(deep=True)
                if self._current_resource_observation is not None
                else None
            ),
        )

    def configure_execution_resource(self, resource: ExecutionResource) -> ExecutionResource:
        """Create or update one durable resource definition deterministically.

        A resource id may change capacity, enabled state, or metadata, but it may
        not silently change resource class or invalidate a committed
        reservation. Configuration changes invalidate only provisional plans.
        """

        existing = self.resources.get(resource.resource_id)
        if existing is not None and existing.resource_class is not resource.resource_class:
            raise ValueError(
                f"Execution resource {resource.resource_id!r} cannot change class "
                f"from {existing.resource_class.value} to {resource.resource_class.value}"
            )
        stored = resource.model_copy(deep=True)
        if self.current_epoch is not None:
            committed_units = sum(
                reservation.units
                for reservation in self._resource_reservations.values()
                if reservation.resource_id == stored.resource_id
            )
            if committed_units > stored.admissible_capacity:
                raise ValueError(
                    f"Execution resource {stored.resource_id!r} cannot reduce safe "
                    "capacity below committed reservations"
                )
        self._invalidate_scheduling_plan()
        self.resources[stored.resource_id] = stored
        return stored.model_copy(deep=True)

    def execution_resources(self, *, enabled_only: bool = False) -> list[ExecutionResource]:
        resources = [
            resource
            for resource in self.resources.values()
            if not enabled_only or resource.enabled
        ]
        resources.sort(key=execution_resource_sort_key)
        return [resource.model_copy(deep=True) for resource in resources]

    def resource_reservations(self) -> list[ResourceReservation]:
        reservations = list(self._resource_reservations.values())
        reservations.sort(key=resource_reservation_sort_key)
        return [reservation.model_copy(deep=True) for reservation in reservations]

    def enable_resource_safety(self, *, policy_version: str) -> None:
        """Require observed capacity for every future worker-visible epoch."""

        normalized = policy_version.strip()
        if not normalized:
            raise ValueError("policy_version must not be empty")
        if self.current_epoch is not None and (
            self.current_epoch.resource_observation_id is None
        ):
            raise ResourceObservationError(
                "configured-only assignments must drain before host safety is enabled"
            )
        self._invalidate_scheduling_plan()
        self.resource_safety_required = True
        self.resource_safety_policy_version = normalized

    def assess_unestimated_processes(
        self,
        *,
        policy: ResourceSafetyPolicy,
    ) -> list[UUID]:
        """Persist conservative guesses until profiling/history can replace them."""

        if policy.policy_version != self.resource_safety_policy_version:
            raise ResourceObservationError(
                "resource estimate policy does not match scheduler safety policy"
            )
        assessed: list[UUID] = []
        for task in sorted(
            self.tasks.values(),
            key=lambda value: (value.created_seq, value.task_id.hex),
        ):
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                continue
            if task.metadata.process_resource_estimate is not None:
                continue
            if task.task_id in self.admitted_task_ids:
                raise ResourceObservationError(
                    "an already-assigned task has no process resource estimate"
                )
            estimate = task.metadata.conservative_process_estimate(policy=policy)
            metadata_payload = task.metadata.model_dump(mode="python")
            metadata_payload["process_resource_estimate"] = estimate
            metadata = SchedulingMetadata.model_validate(metadata_payload)
            self.tasks[task.task_id] = task.model_copy(
                update={"metadata": metadata, "revision": task.revision + 1},
                deep=True,
            )
            assessed.append(task.task_id)
        if assessed:
            self._invalidate_scheduling_plan()
        return assessed

    def attach_resource_observation(
        self,
        observation: ResourceObservationSnapshot,
        *,
        evaluated_at: datetime,
    ) -> None:
        """Attach one fresh, complete input for the next deterministic epoch."""

        if not self.resource_safety_required:
            raise ResourceObservationError(
                "host resource safety must be enabled before attaching observations"
            )
        observation.assert_fresh(at=evaluated_at)
        if observation.scheduler_cycle != self.cycle + 1:
            raise ResourceObservationError(
                "resource observation does not target the next scheduler cycle"
            )
        if (
            observation.safety_policy_version
            != self.resource_safety_policy_version
        ):
            raise ResourceObservationError(
                "resource observation policy does not match scheduler policy"
            )

        capacity_by_id = observation.capacity_by_resource_id()
        resources = self.execution_resources(enabled_only=False)
        if set(capacity_by_id) != {resource.resource_id for resource in resources}:
            raise ResourceObservationError(
                "resource observation must cover every configured resource exactly"
            )
        committed_by_resource: dict[str, int] = {}
        for reservation in self._resource_reservations.values():
            committed_by_resource[reservation.resource_id] = (
                committed_by_resource.get(reservation.resource_id, 0)
                + reservation.units
            )
        for resource in resources:
            capacity = capacity_by_id[resource.resource_id]
            if (
                capacity.resource_class is not resource.resource_class
                or capacity.configured_capacity != resource.capacity
                or capacity.configured_headroom_units != resource.system_headroom
                or capacity.committed_units
                != committed_by_resource.get(resource.resource_id, 0)
            ):
                raise ResourceObservationError(
                    "resource observation does not match configured/committed state"
                )

        self._invalidate_scheduling_plan()
        self._pending_resource_observation = observation.model_copy(deep=True)

    def current_resource_observation(
        self,
    ) -> ResourceObservationSnapshot | None:
        if self._current_resource_observation is None:
            return None
        return self._current_resource_observation.model_copy(deep=True)

    def pending_resource_observation(
        self,
    ) -> ResourceObservationSnapshot | None:
        if self._pending_resource_observation is None:
            return None
        return self._pending_resource_observation.model_copy(deep=True)

    def worker_visible_assignments(self) -> list[DurableAssignment]:
        """Return only the complete assignment set of the committed epoch."""

        if (
            self.current_epoch is None
            or self.current_epoch.status is not SchedulingEpochStatus.COMMITTED
        ):
            return []
        return [
            self._assignments[assignment_id].model_copy(deep=True)
            for assignment_id in self.current_epoch.assignment_ids
        ]

    def pending_preemptions(self) -> list[PendingPreemption]:
        """Return durable checkpoint-gated replacement intents in stable order."""

        return [
            intent.model_copy(deep=True)
            for intent in sorted(
                self._pending_preemptions.values(),
                key=pending_preemption_sort_key,
            )
        ]

    def preemption_events(self) -> list[PreemptionEvent]:
        """Return append-only preemption events created by this scheduler instance."""

        return [event.model_copy(deep=True) for event in self._preemption_events]

    def checkpoint_pending_preemption(
        self,
        victim_task_id: UUID,
        *,
        resumable_state: dict[str, Any] | None = None,
    ) -> PendingPreemption:
        """Acknowledge one declared safe checkpoint without releasing capacity.

        The current assignment and reservations remain worker-visible. A later
        scheduling epoch atomically publishes the replacement only after every
        checkpoint-only victim selected by the intent has acknowledged.
        """

        matches = [
            intent
            for intent in self._pending_preemptions.values()
            if victim_task_id in intent.checkpoint_victim_task_ids
        ]
        if not matches:
            raise KeyError(victim_task_id)
        if len(matches) != 1:
            raise RuntimeError("A checkpoint victim belongs to multiple intents")
        intent = matches[0]
        if victim_task_id in intent.checkpointed_victim_task_ids:
            return intent.model_copy(deep=True)

        self._invalidate_scheduling_plan()
        if resumable_state is not None:
            self.tasks[victim_task_id] = self.tasks[victim_task_id].model_copy(
                update={"resumable_state": dict(resumable_state)},
                deep=True,
            )
        checkpointed = list(intent.checkpointed_victim_task_ids)
        checkpointed.append(victim_task_id)
        updated = intent.model_copy(
            update={"checkpointed_victim_task_ids": checkpointed},
            deep=True,
        )
        self._pending_preemptions[updated.preemption_id] = updated
        self.preemption_state_revision += 1
        self._append_preemption_event(
            make_preemption_event(
                updated,
                event_type=PreemptionEventType.CHECKPOINT_ACKNOWLEDGED,
                scheduler_cycle=self.cycle,
                epoch_sequence=self.epoch_sequence,
                checkpoint_task_id=victim_task_id,
                reason="checkpoint-only victim acknowledged a safe boundary",
            )
        )
        self._validate_snapshot()
        return updated.model_copy(deep=True)

    def pending_scheduling_epoch(self) -> SchedulingEpochPlan | None:
        """Return a copy of the provisional, deliberately invisible epoch plan."""

        if self._pending_epoch_plan is None:
            return None
        return self._pending_epoch_plan.model_copy(deep=True)

    def plan_scheduling_epoch(self) -> SchedulingEpochPlan:
        """Plan the next complete deterministic assignment/reservation epoch.

        Planning is side-effect free with respect to committed worker state.
        The durability adapter makes this plan authoritative only after its
        epoch, assignments, reservation snapshot, and scheduler pointer commit
        in one PostgreSQL transaction.
        """

        if self._pending_epoch_plan is not None:
            return self._pending_epoch_plan.model_copy(deep=True)

        self._validate_snapshot()
        planned_cycle = self.cycle + 1
        next_sequence = self.epoch_sequence + 1
        enabled_resources = self.execution_resources(enabled_only=True)
        resource_observation: ResourceObservationSnapshot | None = None
        if self.resource_safety_required:
            resource_observation = self._pending_resource_observation
            if resource_observation is None:
                raise ResourceObservationError(
                    "a fresh resource observation is required before epoch planning"
                )
            if resource_observation.scheduler_cycle != planned_cycle:
                raise ResourceObservationError(
                    "pending resource observation targets the wrong scheduler cycle"
                )
            unestimated = [
                task.task_id
                for task in self.tasks.values()
                if task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}
                and task.metadata.process_resource_estimate is None
            ]
            if unestimated:
                raise ResourceObservationError(
                    "every unfinished task requires a persisted process estimate"
                )
            observed_capacity = resource_observation.capacity_by_resource_id()
            remaining = {
                resource.resource_id: observed_capacity[
                    resource.resource_id
                ].admission_capacity
                for resource in enabled_resources
            }
        else:
            remaining = {
                resource.resource_id: resource.admissible_capacity
                for resource in enabled_resources
            }
        allow_new_admission = (
            resource_observation is None or resource_observation.healthy
        )
        resources_by_class: dict[ExecutionResourceClass, list[ExecutionResource]] = {
            resource_class: [] for resource_class in ExecutionResourceClass
        }
        for resource in enabled_resources:
            resources_by_class[resource.resource_class].append(resource)

        assignments: list[DurableAssignment] = []
        reservations: list[ResourceReservation] = []
        admitted: list[UUID] = []

        if self.current_epoch is not None:
            for assignment_id in self.current_epoch.assignment_ids:
                assignment = self._assignments[assignment_id].model_copy(deep=True)
                if self.tasks[assignment.task_id].status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                }:
                    # The terminal transition is durable task state. Its old
                    # assignment/reservation remains authoritative only until
                    # this replacement epoch commits without it.
                    continue
                task_reservations = self._reservations_for_task(assignment.task_id)
                self._consume_preserved_reservations(
                    task_reservations,
                    remaining=remaining,
                )
                assignments.append(assignment)
                reservations.extend(task_reservations)
                admitted.append(assignment.task_id)
        elif self.admitted_task_ids:
            # Upgrade an Increment C reservation-only snapshot without churn.
            for task_id in self.admitted_task_ids:
                task_reservations = self._reservations_for_task(task_id)
                self._consume_preserved_reservations(
                    task_reservations,
                    remaining=remaining,
                )
                assignments.append(
                    self._new_assignment(
                        task_id,
                        reservations=task_reservations,
                        created_epoch_sequence=next_sequence,
                    )
                )
                reservations.extend(task_reservations)
                admitted.append(task_id)

        # Only work that was authoritative before this planning pass may be a
        # victim. A provisional assignment never churns another provisional
        # assignment in the same deterministic pass.
        preemptible_task_ids = set(admitted)
        plan_events: list[PreemptionEvent] = []
        carried_pending: list[PendingPreemption] = []
        protected_victim_ids: set[UUID] = set()
        waiting_target_ids: set[UUID] = set()
        released_victim_ids: set[UUID] = set()

        def admit_with_allocation(
            task_id: UUID,
            allocation: tuple[list[ResourceReservation], dict[str, int]],
        ) -> None:
            nonlocal remaining
            task_reservations, remaining = allocation
            task_reservations.sort(key=resource_reservation_sort_key)
            assignments.append(
                self._new_assignment(
                    task_id,
                    reservations=task_reservations,
                    created_epoch_sequence=next_sequence,
                )
            )
            reservations.extend(task_reservations)
            admitted.append(task_id)

        def release_selected(victim_task_ids: set[UUID]) -> None:
            nonlocal assignments, reservations, admitted
            for reservation in reservations:
                if reservation.task_id in victim_task_ids:
                    remaining[reservation.resource_id] += reservation.units
            assignments = [
                assignment
                for assignment in assignments
                if assignment.task_id not in victim_task_ids
            ]
            reservations = [
                reservation
                for reservation in reservations
                if reservation.task_id not in victim_task_ids
            ]
            admitted = [
                task_id for task_id in admitted if task_id not in victim_task_ids
            ]
            released_victim_ids.update(victim_task_ids)

        # Stable checkpoint intents are resolved before new victim selection.
        for intent in self.pending_preemptions():
            if not allow_new_admission:
                carried_pending.append(intent)
                protected_victim_ids.update(intent.victim_task_ids)
                waiting_target_ids.add(intent.target_task_id)
                continue
            target = self.tasks[intent.target_task_id]
            direct_allocation = self._allocate_task_resources(
                target,
                remaining=remaining,
                resources_by_class=resources_by_class,
            )
            if direct_allocation is not None:
                admit_with_allocation(intent.target_task_id, direct_allocation)
                plan_events.append(
                    make_preemption_event(
                        intent,
                        event_type=PreemptionEventType.CANCELLED,
                        scheduler_cycle=planned_cycle,
                        epoch_sequence=next_sequence,
                        reason=(
                            "target became admissible without releasing selected victims"
                        ),
                    )
                )
                waiting_target_ids.add(intent.target_task_id)
                continue

            if not intent.ready_to_execute:
                carried_pending.append(intent)
                protected_victim_ids.update(intent.victim_task_ids)
                waiting_target_ids.add(intent.target_task_id)
                self._hold_available_task_capacity(
                    target,
                    remaining=remaining,
                    resources_by_class=resources_by_class,
                )
                continue

            releasable_remaining = dict(remaining)
            for reservation in reservations:
                if reservation.task_id in set(intent.victim_task_ids):
                    releasable_remaining[reservation.resource_id] += reservation.units
            replacement = self._allocate_task_resources(
                target,
                remaining=releasable_remaining,
                resources_by_class=resources_by_class,
            )
            if replacement is None:
                plan_events.append(
                    make_preemption_event(
                        intent,
                        event_type=PreemptionEventType.CANCELLED,
                        scheduler_cycle=planned_cycle,
                        epoch_sequence=next_sequence,
                        reason="selected victims no longer make the target admissible",
                    )
                )
                continue

            victim_ids = set(intent.victim_task_ids)
            release_selected(victim_ids)
            # The successful simulation already includes the released units.
            remaining = releasable_remaining
            admit_with_allocation(intent.target_task_id, replacement)
            plan_events.append(
                make_preemption_event(
                    intent,
                    event_type=PreemptionEventType.EXECUTED,
                    scheduler_cycle=planned_cycle,
                    epoch_sequence=next_sequence,
                    reason="all required checkpoints committed; replacement published",
                )
            )
            waiting_target_ids.add(intent.target_task_id)

        candidate_ids: list[UUID] = []
        if allow_new_admission:
            if self.active_task_id is not None and self.active_task_id not in admitted:
                candidate_ids.append(self.active_task_id)
            candidate_ids.extend(
                task_id
                for task_id in self._ranked_dependency_satisfied_queued_task_ids(
                    cycle=planned_cycle
                )
                if task_id not in admitted
                and task_id != self.active_task_id
                and task_id not in waiting_target_ids
                and task_id not in released_victim_ids
            )

        for task_id in candidate_ids:
            task = self.tasks[task_id]
            allocation = self._allocate_task_resources(
                task,
                remaining=remaining,
                resources_by_class=resources_by_class,
            )
            if allocation is not None:
                admit_with_allocation(task_id, allocation)
                continue
            if task_id == self.active_task_id:
                raise RuntimeError(
                    "Active task resource requirements exceed the safe capacity envelope"
                )

            deficits = self._resource_class_deficits(
                task,
                remaining=remaining,
                resources_by_class=resources_by_class,
            )
            victim_candidates: list[PreemptionCandidate] = []
            reservations_by_task = {
                assignment.task_id: [
                    reservation
                    for reservation in reservations
                    if reservation.task_id == assignment.task_id
                ]
                for assignment in assignments
            }
            for current_index, assignment in enumerate(assignments):
                victim_task_id = assignment.task_id
                if (
                    victim_task_id not in preemptible_task_ids
                    or victim_task_id in protected_victim_ids
                ):
                    continue
                victim = self.tasks[victim_task_id]
                if victim.metadata.interruption_policy is InterruptionPolicy.ATOMIC:
                    continue
                if not self._strictly_outprioritizes(
                    task,
                    victim,
                    cycle=planned_cycle,
                ):
                    continue
                released_units = {
                    resource_class: sum(
                        reservation.units
                        for reservation in reservations_by_task[victim_task_id]
                        if reservation.resource_class is resource_class
                    )
                    for resource_class in deficits
                }
                if not any(released_units.values()):
                    continue
                victim_candidates.append(
                    PreemptionCandidate(
                        task_id=victim_task_id,
                        assignment_id=assignment.assignment_id,
                        base_priority=int(derive_priority(victim.metadata)),
                        checkpoint_required=(
                            victim.metadata.interruption_policy
                            is InterruptionPolicy.CHECKPOINT_ONLY
                        ),
                        current_order=(
                            assignment.created_epoch_sequence
                            * (len(self.tasks) + 1)
                            + current_index
                        ),
                        released_units=released_units,
                    )
                )

            selected = select_minimum_victims(
                deficits=deficits,
                candidates=victim_candidates,
            )
            if selected is None:
                continue

            victim_task_ids = [victim.task_id for victim in selected]
            victim_assignment_ids = [victim.assignment_id for victim in selected]
            preemption_id = deterministic_preemption_id(
                target_task_id=task.task_id,
                target_task_revision=task.revision,
                victim_assignment_ids=victim_assignment_ids,
                requested_epoch_sequence=next_sequence,
            )
            checkpoint_victim_ids = [
                victim.task_id for victim in selected if victim.checkpoint_required
            ]
            if checkpoint_victim_ids:
                intent = PendingPreemption(
                    preemption_id=preemption_id,
                    target_task_id=task.task_id,
                    target_task_revision=task.revision,
                    victim_task_ids=victim_task_ids,
                    victim_assignment_ids=victim_assignment_ids,
                    checkpoint_victim_task_ids=checkpoint_victim_ids,
                    requested_epoch_sequence=next_sequence,
                )
                carried_pending.append(intent)
                protected_victim_ids.update(victim_task_ids)
                waiting_target_ids.add(task.task_id)
                plan_events.append(
                    make_preemption_event(
                        intent,
                        event_type=PreemptionEventType.REQUESTED,
                        scheduler_cycle=planned_cycle,
                        epoch_sequence=next_sequence,
                        reason="replacement waits for selected safe checkpoints",
                    )
                )
                self._hold_available_task_capacity(
                    task,
                    remaining=remaining,
                    resources_by_class=resources_by_class,
                )
                continue

            victim_ids = set(victim_task_ids)
            release_selected(victim_ids)
            replacement = self._allocate_task_resources(
                task,
                remaining=remaining,
                resources_by_class=resources_by_class,
            )
            if replacement is None:
                raise RuntimeError(
                    "Deterministic victim selection failed to satisfy its deficits"
                )
            admit_with_allocation(task_id, replacement)
            plan_events.append(
                make_selection_preemption_event(
                    preemption_id=preemption_id,
                    target_task_id=task.task_id,
                    victim_task_ids=victim_task_ids,
                    victim_assignment_ids=victim_assignment_ids,
                    event_type=PreemptionEventType.EXECUTED,
                    scheduler_cycle=planned_cycle,
                    epoch_sequence=next_sequence,
                    reason="minimum compatible preemptible victim set released",
                )
            )

        carried_pending.sort(key=pending_preemption_sort_key)
        ranked_ids = self._ranked_dependency_satisfied_queued_task_ids(
            cycle=planned_cycle
        )
        if self.active_task_id is not None and self.active_task_id not in ranked_ids:
            ranked_ids.insert(0, self.active_task_id)
        admitted_set = set(admitted)
        unadmitted = [task_id for task_id in ranked_ids if task_id not in admitted_set]

        reservations.sort(key=resource_reservation_sort_key)
        assignment_ids = [assignment.assignment_id for assignment in assignments]
        previous_epoch_id = (
            self.current_epoch.epoch_id if self.current_epoch is not None else None
        )
        epoch_id = deterministic_epoch_id(
            sequence=next_sequence,
            previous_epoch_id=previous_epoch_id,
            scheduler_cycle=planned_cycle,
            admission_policy_version=self.admission_policy_version,
            assignment_policy_version=self.assignment_policy_version,
            assignment_ids=assignment_ids,
            reservations=reservations,
            preemption_policy_version=self.preemption_policy_version,
            pending_preemptions=carried_pending,
            preemption_event_ids=[event.event_id for event in plan_events],
            executed_preemption_ids=[
                event.preemption_id
                for event in plan_events
                if event.event_type is PreemptionEventType.EXECUTED
            ],
            cancelled_preemption_ids=[
                event.preemption_id
                for event in plan_events
                if event.event_type is PreemptionEventType.CANCELLED
            ],
            resource_observation_id=(
                resource_observation.observation_id
                if resource_observation is not None
                else None
            ),
            resource_safety_policy_version=(
                resource_observation.safety_policy_version
                if resource_observation is not None
                else None
            ),
        )
        plan = SchedulingEpochPlan(
            epoch=SchedulingEpoch(
                epoch_id=epoch_id,
                sequence=next_sequence,
                previous_epoch_id=previous_epoch_id,
                scheduler_cycle=planned_cycle,
                admission_policy_version=self.admission_policy_version,
                assignment_policy_version=self.assignment_policy_version,
                status=SchedulingEpochStatus.PLANNED,
                assignment_ids=assignment_ids,
                reservations=reservations,
                preemption_policy_version=self.preemption_policy_version,
                pending_preemptions=carried_pending,
                preemption_event_ids=[event.event_id for event in plan_events],
                executed_preemption_ids=[
                    event.preemption_id
                    for event in plan_events
                    if event.event_type is PreemptionEventType.EXECUTED
                ],
                cancelled_preemption_ids=[
                    event.preemption_id
                    for event in plan_events
                    if event.event_type is PreemptionEventType.CANCELLED
                ],
                resource_observation_id=(
                    resource_observation.observation_id
                    if resource_observation is not None
                    else None
                ),
                resource_safety_policy_version=(
                    resource_observation.safety_policy_version
                    if resource_observation is not None
                    else None
                ),
            ),
            admitted_task_ids=admitted,
            unadmitted_task_ids=unadmitted,
            assignments=assignments,
            pending_preemptions=carried_pending,
            preemption_events=plan_events,
            resource_observation=(
                resource_observation.model_copy(deep=True)
                if resource_observation is not None
                else None
            ),
        )
        self._pending_epoch_plan = plan.model_copy(deep=True)
        return plan.model_copy(deep=True)

    def _commit_pending_epoch(self, epoch_id: UUID) -> None:
        """Publish a plan in memory after the durability transaction commits."""

        plan = self._pending_epoch_plan
        if plan is None or plan.epoch.epoch_id != epoch_id:
            raise RuntimeError("No matching provisional scheduling epoch")
        committed_epoch = plan.epoch.model_copy(
            update={"status": SchedulingEpochStatus.COMMITTED},
            deep=True,
        )
        self.cycle = committed_epoch.scheduler_cycle
        self.epoch_sequence = committed_epoch.sequence
        self.current_epoch = committed_epoch
        self.admission_policy_version = committed_epoch.admission_policy_version
        self.assignment_policy_version = committed_epoch.assignment_policy_version
        if committed_epoch.preemption_policy_version is not None:
            self.preemption_policy_version = committed_epoch.preemption_policy_version
        self.admitted_task_ids = list(plan.admitted_task_ids)
        self._resource_reservations = {
            reservation.reservation_id: reservation.model_copy(deep=True)
            for reservation in committed_epoch.reservations
        }
        self._assignments = {
            assignment.assignment_id: assignment.model_copy(deep=True)
            for assignment in plan.assignments
        }
        self._pending_preemptions = {
            intent.preemption_id: intent.model_copy(deep=True)
            for intent in plan.pending_preemptions
        }
        self.preemption_state_revision += 1
        for event in plan.preemption_events:
            self._append_preemption_event(event)
        self._current_resource_observation = (
            plan.resource_observation.model_copy(deep=True)
            if plan.resource_observation is not None
            else None
        )
        self._pending_resource_observation = None
        self._pending_epoch_plan = None
        self._validate_snapshot()

    def _preemption_persistence_revision(self) -> int:
        return self._persisted_preemption_state_revision

    def _mark_preemption_state_persisted(self, revision: int) -> None:
        if revision != self.preemption_state_revision:
            raise RuntimeError("Persisted preemption revision does not match scheduler")
        self._persisted_preemption_state_revision = revision

    def _pending_transitions_for_persistence(self) -> list[TaskTransition]:
        if self._persisted_transition_count > len(self.transitions):
            raise RuntimeError("Persisted transition cursor exceeds transition history")
        return [
            transition.model_copy(deep=True)
            for transition in self.transitions[self._persisted_transition_count :]
        ]

    def _mark_transitions_persisted(self, count: int) -> None:
        if count < self._persisted_transition_count or count > len(self.transitions):
            raise RuntimeError("Invalid persisted transition cursor")
        self._persisted_transition_count = count

    def reconcile_resource_admission(self) -> ResourceAdmissionPlan:
        """Reserve a deterministic attention-ordered set that fits safely.

        The current single-focus task is preserved first because contention-
        driven interruption is deliberately deferred to a later v0.7 increment.
        Dependency-satisfied queued work is then considered in normal attention
        order. Each task's multi-resource contract is all-or-nothing.

        This retained Increment C compatibility path records admission and
        reservations only. ``plan_scheduling_epoch`` is the forward assignment
        path.
        """

        if self.resource_safety_required:
            raise RuntimeError(
                "Host-safe admission requires an observed scheduling epoch"
            )
        if self.current_epoch is not None or self._pending_epoch_plan is not None:
            raise RuntimeError(
                "Reservation-only reconciliation is unavailable after epoch scheduling begins"
            )

        remaining = {
            resource.resource_id: resource.admissible_capacity
            for resource in self.execution_resources(enabled_only=True)
        }
        resources_by_class: dict[ExecutionResourceClass, list[ExecutionResource]] = {
            resource_class: [] for resource_class in ExecutionResourceClass
        }
        for resource in self.execution_resources(enabled_only=True):
            resources_by_class[resource.resource_class].append(resource)

        candidate_ids: list[UUID] = []
        if self.active_task_id is not None:
            candidate_ids.append(self.active_task_id)
        candidate_ids.extend(
            task_id
            for task_id in self._ranked_dependency_satisfied_queued_task_ids()
            if task_id != self.active_task_id
        )

        admitted: list[UUID] = []
        unadmitted: list[UUID] = []
        reservations: list[ResourceReservation] = []
        for task_id in candidate_ids:
            allocation = self._allocate_task_resources(
                self.tasks[task_id],
                remaining=remaining,
                resources_by_class=resources_by_class,
            )
            if allocation is None:
                if task_id == self.active_task_id:
                    raise RuntimeError(
                        "Active task resource requirements exceed the safe capacity envelope"
                    )
                unadmitted.append(task_id)
                continue

            task_reservations, updated_remaining = allocation
            remaining = updated_remaining
            admitted.append(task_id)
            reservations.extend(task_reservations)

        reservations.sort(key=resource_reservation_sort_key)
        plan = ResourceAdmissionPlan(
            policy_version=RESOURCE_ADMISSION_POLICY_VERSION,
            admitted_task_ids=admitted,
            unadmitted_task_ids=unadmitted,
            reservations=reservations,
        )
        self.admission_policy_version = plan.policy_version
        self.admitted_task_ids = list(plan.admitted_task_ids)
        self._resource_reservations = {
            reservation.reservation_id: reservation.model_copy(deep=True)
            for reservation in plan.reservations
        }
        return plan.model_copy(deep=True)

    def submit(self, task: AttentionTask) -> AttentionTask:
        if task.task_id in self.tasks:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        if any(existing.created_seq == task.created_seq for existing in self.tasks.values()):
            raise ValueError(f"Duplicate created_seq: {task.created_seq}")
        if task.status is not TaskStatus.NEW:
            raise ValueError("Newly submitted tasks must be NEW")

        stored = task.model_copy(deep=True)
        self.tasks[stored.task_id] = stored
        self._transition(stored.task_id, TaskStatus.QUEUED, "task submitted")
        self._replace_task(stored.task_id, enqueued_cycle=self.cycle)
        return self.tasks[stored.task_id].model_copy(deep=True)

    def advance_cycle(self, count: int = 1) -> int:
        if count < 0:
            raise ValueError("count must be >= 0")
        if count:
            self._invalidate_scheduling_plan()
        self.cycle += count
        return self.cycle

    def queued_tasks(self) -> list[AttentionTask]:
        return [
            self.tasks[task_id].model_copy(deep=True)
            for task_id in self._ranked_queued_task_ids()
        ]

    def reconcile_focus(self) -> FocusDecision:
        """Choose what should own focus at the current deterministic cycle."""

        self._require_legacy_focus_mode()
        self.advance_cycle()
        candidate_id = self._best_queued_task_id()

        if self.active_task_id is None:
            if candidate_id is None:
                return FocusDecision(action=FocusAction.IDLE, reason="no runnable queued tasks")
            self._start(candidate_id, "highest-ranked runnable task")
            return FocusDecision(
                action=FocusAction.START,
                active_task_id=candidate_id,
                candidate_task_id=candidate_id,
                reason="started highest-ranked runnable task",
            )

        active = self.tasks[self.active_task_id]
        if candidate_id is None:
            return FocusDecision(
                action=FocusAction.CONTINUE,
                active_task_id=active.task_id,
                reason="no higher-priority runnable task",
            )

        candidate = self.tasks[candidate_id]
        if not self._strictly_outprioritizes(candidate, active):
            return FocusDecision(
                action=FocusAction.CONTINUE,
                active_task_id=active.task_id,
                candidate_task_id=candidate.task_id,
                reason="queued task does not strictly outrank active task",
            )

        policy = active.metadata.interruption_policy
        if policy is InterruptionPolicy.ATOMIC:
            return FocusDecision(
                action=FocusAction.CONTINUE,
                active_task_id=active.task_id,
                candidate_task_id=candidate.task_id,
                reason="active task is atomic",
            )

        if policy is InterruptionPolicy.CHECKPOINT_ONLY:
            self.pending_preemption_task_id = candidate.task_id
            return FocusDecision(
                action=FocusAction.WAIT_FOR_CHECKPOINT,
                active_task_id=active.task_id,
                candidate_task_id=candidate.task_id,
                reason="higher-priority task waiting for safe checkpoint",
            )

        previous_id = active.task_id
        self._requeue_active("preempted by higher-priority task")
        self._start(candidate.task_id, f"preempted {previous_id}")
        return FocusDecision(
            action=FocusAction.PREEMPT,
            active_task_id=candidate.task_id,
            candidate_task_id=candidate.task_id,
            reason=f"preempted {previous_id}",
        )

    def checkpoint_active(self) -> FocusDecision:
        """Yield at a safe boundary if a pending/higher-priority task exists."""

        self._require_legacy_focus_mode()
        if self.active_task_id is None:
            return self.reconcile_focus()

        self.advance_cycle()
        active = self.tasks[self.active_task_id]
        if active.metadata.interruption_policy is InterruptionPolicy.ATOMIC:
            return FocusDecision(
                action=FocusAction.CONTINUE,
                active_task_id=active.task_id,
                reason="atomic task ignores checkpoints for preemption",
            )

        candidate_id = self._best_queued_task_id()
        if candidate_id is None:
            self.pending_preemption_task_id = None
            return FocusDecision(
                action=FocusAction.CONTINUE,
                active_task_id=active.task_id,
                reason="checkpoint reached with no runnable queued task",
            )

        candidate = self.tasks[candidate_id]
        if not self._strictly_outprioritizes(candidate, active):
            self.pending_preemption_task_id = None
            return FocusDecision(
                action=FocusAction.CONTINUE,
                active_task_id=active.task_id,
                candidate_task_id=candidate.task_id,
                reason="checkpoint reached but queued task no longer outranks active task",
            )

        previous_id = active.task_id
        self._requeue_active("yielded at checkpoint to higher-priority task")
        self._start(candidate.task_id, f"checkpoint preemption of {previous_id}")
        return FocusDecision(
            action=FocusAction.PREEMPT,
            active_task_id=candidate.task_id,
            candidate_task_id=candidate.task_id,
            reason=f"checkpoint preemption of {previous_id}",
        )

    def complete_active(self, resumable_state: dict[str, Any] | None = None) -> AttentionTask:
        self._require_legacy_focus_mode()
        if self.active_task_id is None:
            raise RuntimeError("No active task")
        task_id = self.active_task_id
        if resumable_state is not None:
            self._replace_task(task_id, resumable_state=dict(resumable_state))
        self._transition(task_id, TaskStatus.COMPLETED, "task completed")
        self.active_task_id = None
        self.pending_preemption_task_id = None
        return self.tasks[task_id].model_copy(deep=True)

    def fail_active(self, resumable_state: dict[str, Any] | None = None) -> AttentionTask:
        self._require_legacy_focus_mode()
        if self.active_task_id is None:
            raise RuntimeError("No active task")
        task_id = self.active_task_id
        if resumable_state is not None:
            self._replace_task(task_id, resumable_state=dict(resumable_state))
        self._transition(task_id, TaskStatus.FAILED, "task failed")
        self.active_task_id = None
        self.pending_preemption_task_id = None
        return self.tasks[task_id].model_copy(deep=True)

    def set_resumable_state(self, task_id: UUID, state: dict[str, Any]) -> AttentionTask:
        if task_id not in self.tasks:
            raise KeyError(task_id)
        self._replace_task(task_id, resumable_state=dict(state))
        return self.tasks[task_id].model_copy(deep=True)

    def complete_task(
        self,
        task_id: UUID,
        resumable_state: dict[str, Any] | None = None,
    ) -> AttentionTask:
        """Complete one concurrently assigned task after its worker steps finish."""

        if task_id not in self.tasks:
            raise KeyError(task_id)
        task = self.tasks[task_id]
        if task.status is TaskStatus.COMPLETED:
            return task.model_copy(deep=True)
        if task.status in {TaskStatus.FAILED, TaskStatus.NEW}:
            raise RuntimeError(f"Task {task_id} cannot complete from {task.status.value}")
        if resumable_state is not None:
            self._replace_task(task_id, resumable_state=dict(resumable_state))
        self._transition(task_id, TaskStatus.COMPLETED, "all durable worker steps completed")
        return self.tasks[task_id].model_copy(deep=True)

    def _ranked_queued_task_ids(self, *, cycle: int | None = None) -> list[UUID]:
        runnable = [
            self.tasks[task_id]
            for task_id in self._ranked_dependency_satisfied_queued_task_ids(cycle=cycle)
            if self._resources_satisfied(self.tasks[task_id])
        ]
        return [task.task_id for task in runnable]

    def _ranked_dependency_satisfied_queued_task_ids(
        self,
        *,
        cycle: int | None = None,
    ) -> list[UUID]:
        runnable = [
            task
            for task in self.tasks.values()
            if task.status is TaskStatus.QUEUED
            and self._dependencies_satisfied(task)
        ]
        runnable.sort(key=lambda task: self._queue_key(task, cycle=cycle))
        return [task.task_id for task in runnable]

    def _best_queued_task_id(self) -> UUID | None:
        ranked = self._ranked_queued_task_ids()
        return ranked[0] if ranked else None

    def _dependencies_satisfied(self, task: AttentionTask) -> bool:
        for dependency_id in task.metadata.dependency_ids:
            dependency = self.tasks.get(dependency_id)
            if dependency is None or dependency.status is not TaskStatus.COMPLETED:
                return False
        return True

    def _resources_satisfied(self, task: AttentionTask) -> bool:
        available_by_class = {
            resource_class: 0 for resource_class in ExecutionResourceClass
        }
        for resource in self.resources.values():
            available_by_class[resource.resource_class] += resource.admissible_capacity
        return all(
            available_by_class[requirement.resource_class] >= requirement.units
            for requirement in task.metadata.quantitative_resource_requirements()
        )

    def _allocate_task_resources(
        self,
        task: AttentionTask,
        *,
        remaining: dict[str, int],
        resources_by_class: dict[ExecutionResourceClass, list[ExecutionResource]],
    ) -> tuple[list[ResourceReservation], dict[str, int]] | None:
        tentative_remaining = dict(remaining)
        reservations: list[ResourceReservation] = []
        for requirement in task.metadata.quantitative_resource_requirements():
            needed = requirement.units
            for resource in resources_by_class[requirement.resource_class]:
                available = tentative_remaining[resource.resource_id]
                reserved = min(needed, available)
                if reserved <= 0:
                    continue
                reservations.append(
                    ResourceReservation(
                        reservation_id=deterministic_reservation_id(
                            task.task_id,
                            resource.resource_id,
                        ),
                        task_id=task.task_id,
                        resource_id=resource.resource_id,
                        resource_class=resource.resource_class,
                        units=reserved,
                    )
                )
                tentative_remaining[resource.resource_id] -= reserved
                needed -= reserved
                if needed == 0:
                    break
            if needed:
                return None
        return reservations, tentative_remaining

    @staticmethod
    def _resource_class_deficits(
        task: AttentionTask,
        *,
        remaining: dict[str, int],
        resources_by_class: dict[
            ExecutionResourceClass,
            list[ExecutionResource],
        ],
    ) -> dict[ExecutionResourceClass, int]:
        deficits: dict[ExecutionResourceClass, int] = {}
        for requirement in task.metadata.quantitative_resource_requirements():
            available = sum(
                remaining[resource.resource_id]
                for resource in resources_by_class[requirement.resource_class]
            )
            missing = max(0, requirement.units - available)
            if missing:
                deficits[requirement.resource_class] = missing
        return deficits

    @staticmethod
    def _hold_available_task_capacity(
        task: AttentionTask,
        *,
        remaining: dict[str, int],
        resources_by_class: dict[
            ExecutionResourceClass,
            list[ExecutionResource],
        ],
    ) -> None:
        """Protect free capacity already needed by a checkpoint-gated target.

        The hold is provisional planning state, not a worker-visible
        reservation. It prevents later, lower-ranked work from consuming the
        free half of a replacement contract while selected victims checkpoint.
        """

        for requirement in task.metadata.quantitative_resource_requirements():
            held = 0
            for resource in resources_by_class[requirement.resource_class]:
                units = min(requirement.units - held, remaining[resource.resource_id])
                if units <= 0:
                    continue
                remaining[resource.resource_id] -= units
                held += units
                if held == requirement.units:
                    break

    def _new_assignment(
        self,
        task_id: UUID,
        *,
        reservations: list[ResourceReservation],
        created_epoch_sequence: int,
    ) -> DurableAssignment:
        task = self.tasks[task_id]
        return DurableAssignment(
            assignment_id=deterministic_assignment_id(
                task_id,
                task_revision=task.revision,
                created_epoch_sequence=created_epoch_sequence,
                reservations=reservations,
            ),
            task_id=task_id,
            task_revision=task.revision,
            created_epoch_sequence=created_epoch_sequence,
            reservation_ids=[
                reservation.reservation_id for reservation in reservations
            ],
        )

    def _service_due(self, task: AttentionTask, *, cycle: int | None = None) -> bool:
        guarantee = self.service_guarantees[task.metadata.service_class]
        if guarantee.max_wait_cycles is None:
            return False
        current_cycle = self.cycle if cycle is None else cycle
        waited = max(0, current_cycle - task.enqueued_cycle)
        return waited >= guarantee.max_wait_cycles

    def _effective_priority(
        self,
        task: AttentionTask,
        *,
        cycle: int | None = None,
    ) -> PriorityClass:
        base = derive_priority(task.metadata)
        if task.status is not TaskStatus.QUEUED:
            return base
        guarantee = self.service_guarantees[task.metadata.service_class]
        if not self._service_due(task, cycle=cycle):
            return base
        return PriorityClass(min(int(base), int(guarantee.guaranteed_priority)))

    def _queue_key(
        self,
        task: AttentionTask,
        *,
        cycle: int | None = None,
    ) -> tuple[int, int, datetime, int, str]:
        # A due guarantee wins tie-breaking within its promoted priority class.
        return (
            int(self._effective_priority(task, cycle=cycle)),
            0 if self._service_due(task, cycle=cycle) else 1,
            _normalize_deadline(task.metadata.deadline),
            task.created_seq,
            task.task_id.hex,
        )

    def _strictly_outprioritizes(
        self,
        candidate: AttentionTask,
        active: AttentionTask,
        *,
        cycle: int | None = None,
    ) -> bool:
        # Same-priority arrivals do not preempt.  This prevents deterministic
        # thrashing; FIFO/deadline tie-breakers apply when focus next becomes free.
        return int(self._effective_priority(candidate, cycle=cycle)) < int(
            derive_priority(active.metadata)
        )

    def _start(self, task_id: UUID, reason: str) -> None:
        if self.active_task_id is not None:
            raise RuntimeError("Cannot start a second active task")
        task = self.tasks[task_id]
        if task.status is not TaskStatus.QUEUED:
            raise RuntimeError(f"Task {task_id} is not queued")
        self._transition(task_id, TaskStatus.RUNNING, reason)
        self.active_task_id = task_id
        self.pending_preemption_task_id = None

    def _requeue_active(self, reason: str) -> None:
        if self.active_task_id is None:
            raise RuntimeError("No active task")
        task_id = self.active_task_id
        self._transition(task_id, TaskStatus.SUSPENDED, reason)
        self._transition(task_id, TaskStatus.QUEUED, "suspended task returned to queue")
        self._replace_task(task_id, enqueued_cycle=self.cycle)
        self.active_task_id = None
        self.pending_preemption_task_id = None

    def _transition(self, task_id: UUID, to_status: TaskStatus, reason: str) -> None:
        self._invalidate_scheduling_plan()
        task = self.tasks[task_id]
        from_status = task.status
        revision = task.revision + 1
        transition = TaskTransition(
            transition_id=uuid5(task.task_id, f"transition:{revision}:{from_status.value}:{to_status.value}"),
            task_id=task.task_id,
            revision=revision,
            scheduler_cycle=self.cycle,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
        )
        self.transitions.append(transition)
        self.tasks[task_id] = task.model_copy(
            update={"status": to_status, "revision": revision},
            deep=True,
        )

    def _replace_task(self, task_id: UUID, **updates: Any) -> None:
        self._invalidate_scheduling_plan()
        self.tasks[task_id] = self.tasks[task_id].model_copy(update=updates, deep=True)

    def _reservations_for_task(self, task_id: UUID) -> list[ResourceReservation]:
        reservations = [
            reservation.model_copy(deep=True)
            for reservation in self._resource_reservations.values()
            if reservation.task_id == task_id
        ]
        reservations.sort(key=resource_reservation_sort_key)
        return reservations

    @staticmethod
    def _consume_preserved_reservations(
        reservations: list[ResourceReservation],
        *,
        remaining: dict[str, int],
    ) -> None:
        for reservation in reservations:
            if reservation.resource_id not in remaining:
                raise RuntimeError("Committed reservation resource is unavailable")
            remaining[reservation.resource_id] -= reservation.units
            if remaining[reservation.resource_id] < 0:
                raise RuntimeError("Committed reservations exceed safe resource capacity")

    def _invalidate_scheduling_plan(self) -> None:
        self._pending_epoch_plan = None
        self._pending_resource_observation = None
        if self.current_epoch is None:
            # Increment C admission has no committed epoch boundary and is stale
            # after a scheduling mutation. Increment D committed state remains
            # worker-visible until a replacement epoch commits.
            self.admitted_task_ids = []
            self._resource_reservations = {}

    def _append_preemption_event(self, event: PreemptionEvent) -> None:
        for existing in self._preemption_events:
            if existing.event_id != event.event_id:
                continue
            if existing.model_dump(mode="json") != event.model_dump(mode="json"):
                raise RuntimeError("Conflicting deterministic preemption event")
            return
        self._preemption_events.append(event.model_copy(deep=True))

    def _require_legacy_focus_mode(self) -> None:
        if self.resource_safety_required:
            raise RuntimeError(
                "Host-safe execution requires observed scheduling epochs"
            )
        if self.current_epoch is not None or self._pending_epoch_plan is not None:
            raise RuntimeError(
                "Single-focus lifecycle operations are unavailable in epoch mode"
            )

    def _validate_snapshot(self) -> None:
        if not self.preemption_policy_version.strip():
            raise ValueError("Snapshot preemption policy version must not be empty")
        if not self.resource_safety_policy_version.strip():
            raise ValueError("Snapshot resource safety policy version must not be empty")
        event_ids = [event.event_id for event in self._preemption_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Snapshot contains duplicate preemption event ids")

        running = [task.task_id for task in self.tasks.values() if task.status is TaskStatus.RUNNING]
        if self.active_task_id is None:
            if running:
                raise ValueError("Snapshot has RUNNING task but no active_task_id")
        elif running != [self.active_task_id]:
            raise ValueError("Snapshot active_task_id does not match exactly one RUNNING task")

        if self.pending_preemption_task_id is not None:
            pending = self.tasks.get(self.pending_preemption_task_id)
            if pending is None or pending.status is not TaskStatus.QUEUED:
                raise ValueError("pending_preemption_task_id must reference a queued task")

        created = [task.created_seq for task in self.tasks.values()]
        if len(created) != len(set(created)):
            raise ValueError("Snapshot contains duplicate created_seq values")

        admitted = set(self.admitted_task_ids)
        if len(admitted) != len(self.admitted_task_ids):
            raise ValueError("Snapshot contains duplicate admitted task ids")
        for task_id in self.admitted_task_ids:
            task = self.tasks.get(task_id)
            if task is None:
                raise ValueError("admitted_task_ids must reference existing tasks")
            if task.status not in {
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            }:
                raise ValueError("Only executable or terminal-draining tasks may be admitted")
            if not self._dependencies_satisfied(task):
                raise ValueError("Admitted task dependencies must be satisfied")
        if admitted and self.active_task_id is not None and self.active_task_id not in admitted:
            raise ValueError("An authoritative admission set must include the active task")

        resource_usage: dict[str, int] = {}
        task_class_usage: dict[UUID, dict[ExecutionResourceClass, int]] = {}
        task_resource_pairs: set[tuple[UUID, str]] = set()
        for reservation in self._resource_reservations.values():
            if reservation.task_id not in admitted:
                raise ValueError("Resource reservation must belong to an admitted task")
            resource = self.resources.get(reservation.resource_id)
            if resource is None:
                raise ValueError("Resource reservation references an unknown resource")
            if resource.resource_class is not reservation.resource_class:
                raise ValueError("Resource reservation class does not match its resource")
            if reservation.reservation_id != deterministic_reservation_id(
                reservation.task_id,
                reservation.resource_id,
            ):
                raise ValueError("Resource reservation id is not deterministic")
            pair = (reservation.task_id, reservation.resource_id)
            if pair in task_resource_pairs:
                raise ValueError("Task has duplicate reservations for one resource")
            task_resource_pairs.add(pair)

            resource_usage[reservation.resource_id] = (
                resource_usage.get(reservation.resource_id, 0) + reservation.units
            )
            if resource_usage[reservation.resource_id] > resource.admissible_capacity:
                raise ValueError("Resource reservations exceed admissible capacity")

            class_usage = task_class_usage.setdefault(reservation.task_id, {})
            class_usage[reservation.resource_class] = (
                class_usage.get(reservation.resource_class, 0) + reservation.units
            )

        for task_id in self.admitted_task_ids:
            expected = {
                requirement.resource_class: requirement.units
                for requirement in self.tasks[
                    task_id
                ].metadata.quantitative_resource_requirements()
            }
            if task_class_usage.get(task_id, {}) != expected:
                raise ValueError(
                    "Admitted task reservations do not exactly satisfy its requirements"
                )

        if self.current_epoch is None:
            if self.epoch_sequence != 0:
                raise ValueError("Snapshot without a current epoch must have sequence zero")
            if self._assignments:
                raise ValueError("Snapshot assignments require a committed current epoch")
            if self._pending_preemptions:
                raise ValueError("Pending preemptions require a committed current epoch")
            if self._current_resource_observation is not None:
                raise ValueError(
                    "A current resource observation requires a committed epoch"
                )
            return

        epoch = self.current_epoch
        if epoch.status is not SchedulingEpochStatus.COMMITTED:
            raise ValueError("The current scheduling epoch must be committed")
        if epoch.sequence != self.epoch_sequence:
            raise ValueError("Current epoch sequence does not match scheduler state")
        if epoch.scheduler_cycle > self.cycle:
            raise ValueError("Current epoch cannot be newer than scheduler state")
        if epoch.admission_policy_version != self.admission_policy_version:
            raise ValueError("Current epoch admission policy does not match scheduler state")
        if epoch.assignment_policy_version != self.assignment_policy_version:
            raise ValueError("Current epoch assignment policy does not match scheduler state")
        if (
            epoch.preemption_policy_version is not None
            and epoch.preemption_policy_version != self.preemption_policy_version
        ):
            raise ValueError("Current epoch preemption policy does not match scheduler state")
        if epoch.sequence == 1 and epoch.previous_epoch_id is not None:
            raise ValueError("The first scheduling epoch cannot have a predecessor")
        if epoch.sequence > 1 and epoch.previous_epoch_id is None:
            raise ValueError("A later scheduling epoch must identify its predecessor")
        if self.resource_safety_required and epoch.resource_observation_id is None:
            raise ValueError("Host-safe epoch must reference a resource observation")
        if epoch.resource_observation_id is None:
            if self._current_resource_observation is not None:
                raise ValueError("Configured-only epoch cannot own an observation")
        else:
            observation = self._current_resource_observation
            if observation is None:
                raise ValueError("Epoch resource observation is missing from state")
            if (
                observation.observation_id != epoch.resource_observation_id
                or observation.scheduler_cycle != epoch.scheduler_cycle
                or observation.safety_policy_version
                != epoch.resource_safety_policy_version
                or observation.safety_policy_version
                != self.resource_safety_policy_version
            ):
                raise ValueError("Current resource observation does not match epoch")
            observed_by_id = observation.capacity_by_resource_id()
            for resource_id, used_units in resource_usage.items():
                capacity = observed_by_id.get(resource_id)
                if capacity is None or used_units > capacity.admission_capacity:
                    raise ValueError(
                        "Current reservations exceed observed admission capacity"
                    )

        assignment_ids = list(self._assignments)
        if set(assignment_ids) != set(epoch.assignment_ids):
            raise ValueError("Current epoch assignment ids do not match assignment state")
        assignments = [
            self._assignments[assignment_id]
            for assignment_id in epoch.assignment_ids
        ]
        if [assignment.task_id for assignment in assignments] != self.admitted_task_ids:
            raise ValueError("Current assignments must match admitted task order")

        current_reservations = self.resource_reservations()
        if [
            reservation.model_dump(mode="json")
            for reservation in epoch.reservations
        ] != [
            reservation.model_dump(mode="json")
            for reservation in current_reservations
        ]:
            raise ValueError("Current epoch reservation snapshot does not match scheduler state")

        for assignment in assignments:
            task = self.tasks.get(assignment.task_id)
            if task is None:
                raise ValueError("Assignment must reference an existing task")
            if assignment.status is not AssignmentStatus.READY:
                raise ValueError("Worker-visible assignments must be ready")
            if (
                epoch.resource_observation_id is not None
                and task.metadata.process_resource_estimate is None
            ):
                raise ValueError(
                    "Host-safe assignment requires a process resource estimate"
                )
            terminal_drain = (
                task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
                and assignment.task_revision + 1 == task.revision
            )
            if assignment.task_revision != task.revision and not terminal_drain:
                raise ValueError("Assignment task revision does not match current task")
            if assignment.created_epoch_sequence > epoch.sequence:
                raise ValueError("Assignment cannot be created after its current epoch")
            task_reservations = self._reservations_for_task(assignment.task_id)
            if set(assignment.reservation_ids) != {
                reservation.reservation_id for reservation in task_reservations
            }:
                raise ValueError(
                    "Assignment reservation ids do not match current task reservations"
                )
            if assignment.assignment_id != deterministic_assignment_id(
                assignment.task_id,
                task_revision=assignment.task_revision,
                created_epoch_sequence=assignment.created_epoch_sequence,
                reservations=task_reservations,
            ):
                raise ValueError("Assignment id is not deterministic")

        assignment_by_task = {
            assignment.task_id: assignment for assignment in assignments
        }
        pending_targets: set[UUID] = set()
        pending_victims: set[UUID] = set()
        for intent in self.pending_preemptions():
            target = self.tasks.get(intent.target_task_id)
            if target is None or target.status is not TaskStatus.QUEUED:
                raise ValueError("Pending preemption target must be a queued task")
            if not self._dependencies_satisfied(target):
                raise ValueError("Pending preemption target dependencies must be satisfied")
            if target.revision != intent.target_task_revision:
                raise ValueError("Pending preemption target revision does not match")
            if intent.target_task_id in admitted:
                raise ValueError("Pending preemption target cannot already be admitted")
            if intent.requested_epoch_sequence > self.epoch_sequence:
                raise ValueError("Pending preemption cannot originate in a future epoch")
            if intent.target_task_id in pending_targets:
                raise ValueError("A task may be the target of only one pending preemption")
            if pending_victims.intersection(intent.victim_task_ids):
                raise ValueError("A victim may belong to only one pending preemption")

            checkpoint_victims: list[UUID] = []
            for victim_task_id, victim_assignment_id in zip(
                intent.victim_task_ids,
                intent.victim_assignment_ids,
                strict=True,
            ):
                victim = self.tasks.get(victim_task_id)
                assignment = assignment_by_task.get(victim_task_id)
                if victim is None or assignment is None:
                    raise ValueError("Pending preemption victim must remain assigned")
                if assignment.assignment_id != victim_assignment_id:
                    raise ValueError("Pending preemption victim assignment does not match")
                if victim.metadata.interruption_policy is InterruptionPolicy.ATOMIC:
                    raise ValueError("Atomic work cannot be a preemption victim")
                if (
                    victim.metadata.interruption_policy
                    is InterruptionPolicy.CHECKPOINT_ONLY
                ):
                    checkpoint_victims.append(victim_task_id)
            if checkpoint_victims != intent.checkpoint_victim_task_ids:
                raise ValueError("Pending preemption checkpoint victims do not match policy")
            if intent.preemption_id != deterministic_preemption_id(
                target_task_id=intent.target_task_id,
                target_task_revision=intent.target_task_revision,
                victim_assignment_ids=intent.victim_assignment_ids,
                requested_epoch_sequence=intent.requested_epoch_sequence,
            ):
                raise ValueError("Pending preemption id is not deterministic")
            pending_targets.add(intent.target_task_id)
            pending_victims.update(intent.victim_task_ids)
        if pending_targets.intersection(pending_victims):
            raise ValueError("Pending preemption targets and victims must be disjoint")

        expected_epoch_id = deterministic_epoch_id(
            sequence=epoch.sequence,
            previous_epoch_id=epoch.previous_epoch_id,
            scheduler_cycle=epoch.scheduler_cycle,
            admission_policy_version=epoch.admission_policy_version,
            assignment_policy_version=epoch.assignment_policy_version,
            assignment_ids=epoch.assignment_ids,
            reservations=epoch.reservations,
            preemption_policy_version=epoch.preemption_policy_version,
            pending_preemptions=epoch.pending_preemptions,
            preemption_event_ids=epoch.preemption_event_ids,
            executed_preemption_ids=epoch.executed_preemption_ids,
            cancelled_preemption_ids=epoch.cancelled_preemption_ids,
            resource_observation_id=epoch.resource_observation_id,
            resource_safety_policy_version=(
                epoch.resource_safety_policy_version
            ),
        )
        if epoch.epoch_id != expected_epoch_id:
            raise ValueError("Scheduling epoch id is not deterministic")
