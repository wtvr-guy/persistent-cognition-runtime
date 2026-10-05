"""PostgreSQL durability adapter for the deterministic JIT Attention scheduler."""
from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist.attention import (
    AttentionTask,
    InterruptionPolicy,
    JITAttentionScheduler,
    SchedulerSnapshot,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    TaskStatus,
)
from prometheist.attention_resources import (
    ExecutionResource,
    ExecutionResourceClass,
    ProcessResourceEstimate,
    ResourceRequirement,
    ResourceReservation,
)
from prometheist.attention_observation import ResourceObservationSnapshot
from prometheist.db import require_row
from prometheist.attention_assignments import (
    AssignmentStatus,
    DurableAssignment,
    SchedulingEpoch,
    SchedulingEpochStatus,
)
from prometheist.attention_preemption import (
    PendingPreemption,
    PreemptionEvent,
    PreemptionEventType,
)


DEFAULT_SCHEDULER_KEY = "default"


def allocate_created_seq(conn: psycopg.Connection) -> int:
    """Allocate the authoritative intake sequence used for deterministic tie-breaking."""

    with conn.cursor() as cur:
        cur.execute("SELECT nextval('attention_task_created_seq')")
        return int(require_row(cur.fetchone(), context="attention_task_created_seq")[0])


def save_scheduler(
    conn: psycopg.Connection,
    scheduler: JITAttentionScheduler,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> None:
    """Persist scheduler state and publish a pending epoch atomically.

    A provisional epoch remains worker-invisible until its immutable epoch row,
    assignment rows, complete reservation set, and scheduler pointer all commit
    in this transaction. The in-memory scheduler is published only afterward.
    """

    snapshot = scheduler.snapshot()
    pending_transitions = scheduler._pending_transitions_for_persistence()
    persisted_transition_count = len(scheduler.transitions)
    pending = scheduler.pending_scheduling_epoch()
    preemption_events = scheduler.preemption_events()
    preemption_base_revision = scheduler._preemption_persistence_revision()
    if pending is None:
        epoch = snapshot.current_epoch
        assignments = snapshot.assignments
        state_cycle = snapshot.cycle
        epoch_sequence = snapshot.epoch_sequence
        admitted_task_ids = snapshot.admitted_task_ids
        reservations = snapshot.resource_reservations
        pending_preemptions = snapshot.pending_preemptions
        preemption_policy_version = snapshot.preemption_policy_version
        preemption_state_revision = snapshot.preemption_state_revision
        resource_observation = snapshot.current_resource_observation
    else:
        epoch = pending.epoch.model_copy(
            update={"status": SchedulingEpochStatus.COMMITTED},
            deep=True,
        )
        assignments = pending.assignments
        state_cycle = epoch.scheduler_cycle
        epoch_sequence = epoch.sequence
        admitted_task_ids = pending.admitted_task_ids
        reservations = epoch.reservations
        pending_preemptions = pending.pending_preemptions
        preemption_policy_version = (
            epoch.preemption_policy_version or snapshot.preemption_policy_version
        )
        preemption_state_revision = snapshot.preemption_state_revision + 1
        resource_observation = pending.resource_observation
        known_event_ids = {event.event_id for event in preemption_events}
        preemption_events.extend(
            event.model_copy(deep=True)
            for event in pending.preemption_events
            if event.event_id not in known_event_ids
        )
        publication_snapshot = SchedulerSnapshot(
            cycle=state_cycle,
            active_task_id=snapshot.active_task_id,
            pending_preemption_task_id=snapshot.pending_preemption_task_id,
            tasks=snapshot.tasks,
            execution_resources=snapshot.execution_resources,
            admission_policy_version=epoch.admission_policy_version,
            admitted_task_ids=admitted_task_ids,
            resource_reservations=reservations,
            epoch_sequence=epoch_sequence,
            assignment_policy_version=epoch.assignment_policy_version,
            preemption_policy_version=preemption_policy_version,
            preemption_state_revision=preemption_state_revision,
            pending_preemptions=pending_preemptions,
            current_epoch=epoch,
            assignments=assignments,
            resource_safety_required=snapshot.resource_safety_required,
            resource_safety_policy_version=(
                snapshot.resource_safety_policy_version
            ),
            current_resource_observation=resource_observation,
        )
        JITAttentionScheduler.from_snapshot(
            publication_snapshot,
            service_guarantees=scheduler.service_guarantees,
        )

    try:
        with conn.cursor() as cur:
            _lock_scheduler_generation(
                cur,
                scheduler_key=scheduler_key,
                snapshot_epoch_sequence=snapshot.epoch_sequence,
                snapshot_epoch_id=(
                    snapshot.current_epoch.epoch_id
                    if snapshot.current_epoch is not None
                    else None
                ),
                target_epoch=epoch if pending is not None else None,
                preemption_base_revision=preemption_base_revision,
                target_preemption_revision=preemption_state_revision,
                target_pending_preemptions=pending_preemptions,
            )
            for resource in snapshot.execution_resources:
                _upsert_execution_resource(cur, resource)

            for task in snapshot.tasks:
                _upsert_task(cur, task, scheduler_key=scheduler_key)

            for transition in pending_transitions:
                cur.execute(
                    """
                    INSERT INTO attention_task_transitions (
                        scheduler_key, transition_id, task_id, revision, scheduler_cycle,
                        from_status, to_status, reason
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transition_id) DO NOTHING
                    """,
                    (
                        scheduler_key,
                        transition.transition_id,
                        transition.task_id,
                        transition.revision,
                        transition.scheduler_cycle,
                        transition.from_status.value,
                        transition.to_status.value,
                        transition.reason,
                    ),
                )

            if resource_observation is not None:
                _insert_immutable_resource_observation(
                    cur,
                    scheduler_key=scheduler_key,
                    observation=resource_observation,
                )

            if epoch is not None:
                for assignment in assignments:
                    _insert_immutable_assignment(
                        cur,
                        scheduler_key=scheduler_key,
                        assignment=assignment,
                    )
                _insert_immutable_epoch(
                    cur,
                    scheduler_key=scheduler_key,
                    epoch=epoch,
                )

            cur.execute(
                """
                INSERT INTO attention_scheduler_state (
                    scheduler_key, cycle, active_task_id,
                    pending_preemption_task_id, admission_policy_version,
                    admitted_task_ids, epoch_sequence, current_epoch_id,
                    assignment_policy_version, preemption_policy_version,
                    preemption_state_revision, pending_preemptions,
                    resource_safety_required,
                    resource_safety_policy_version,
                    current_resource_observation_id
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (scheduler_key) DO UPDATE SET
                    cycle = EXCLUDED.cycle,
                    active_task_id = EXCLUDED.active_task_id,
                    pending_preemption_task_id = EXCLUDED.pending_preemption_task_id,
                    admission_policy_version = EXCLUDED.admission_policy_version,
                    admitted_task_ids = EXCLUDED.admitted_task_ids,
                    epoch_sequence = EXCLUDED.epoch_sequence,
                    current_epoch_id = EXCLUDED.current_epoch_id,
                    assignment_policy_version = EXCLUDED.assignment_policy_version,
                    preemption_policy_version = EXCLUDED.preemption_policy_version,
                    preemption_state_revision = EXCLUDED.preemption_state_revision,
                    pending_preemptions = EXCLUDED.pending_preemptions,
                    resource_safety_required = EXCLUDED.resource_safety_required,
                    resource_safety_policy_version = EXCLUDED.resource_safety_policy_version,
                    current_resource_observation_id = EXCLUDED.current_resource_observation_id,
                    updated_at = now()
                """,
                (
                    scheduler_key,
                    state_cycle,
                    snapshot.active_task_id,
                    snapshot.pending_preemption_task_id,
                    epoch.admission_policy_version
                    if epoch is not None
                    else snapshot.admission_policy_version,
                    Json([str(task_id) for task_id in admitted_task_ids]),
                    epoch_sequence,
                    epoch.epoch_id if epoch is not None else None,
                    epoch.assignment_policy_version
                    if epoch is not None
                    else snapshot.assignment_policy_version,
                    preemption_policy_version,
                    preemption_state_revision,
                    Json(
                        [
                            intent.model_dump(mode="json")
                            for intent in pending_preemptions
                        ]
                    ),
                    snapshot.resource_safety_required,
                    snapshot.resource_safety_policy_version,
                    (
                        resource_observation.observation_id
                        if resource_observation is not None
                        else None
                    ),
                ),
            )
            for event in preemption_events:
                _insert_immutable_preemption_event(
                    cur,
                    scheduler_key=scheduler_key,
                    event=event,
                )
            _replace_resource_reservations(
                cur,
                scheduler_key=scheduler_key,
                reservations=reservations,
                assignments=assignments,
                observed_admission_limits=(
                    {
                        capacity.resource_id: capacity.admission_capacity
                        for capacity in resource_observation.capacities
                    }
                    if resource_observation is not None
                    else None
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if pending is not None:
        scheduler._commit_pending_epoch(pending.epoch.epoch_id)
    scheduler._mark_transitions_persisted(persisted_transition_count)
    scheduler._mark_preemption_state_persisted(preemption_state_revision)


def load_scheduler(
    conn: psycopg.Connection,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> JITAttentionScheduler:
    """Reconstruct resource, focus, and resumable task state without LLM context."""

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                cycle, active_task_id, pending_preemption_task_id,
                admission_policy_version, admitted_task_ids,
                epoch_sequence, current_epoch_id, assignment_policy_version,
                preemption_policy_version, preemption_state_revision,
                pending_preemptions, resource_safety_required,
                resource_safety_policy_version,
                current_resource_observation_id
            FROM attention_scheduler_state
            WHERE scheduler_key = %s
            """,
            (scheduler_key,),
        )
        state = cur.fetchone()

        cur.execute(
            """
            SELECT
                resource_id, resource_class, capacity, system_headroom,
                enabled, metadata
            FROM attention_execution_resources
            ORDER BY resource_class ASC, resource_id ASC
            """
        )
        resource_rows = cur.fetchall()
        resources = [_row_to_execution_resource(row) for row in resource_rows]

        if state is None:
            return JITAttentionScheduler.from_snapshot(
                SchedulerSnapshot(cycle=0, tasks=[], execution_resources=resources)
            )

        cur.execute(
            """
            SELECT
                task_id, task_key, created_seq, parent_task_id,
                criticality, service_class, interruption_policy, deadline,
                required_capabilities, required_resource_classes,
                resource_requirements, process_resource_estimate, dependency_ids,
                status, enqueued_cycle, revision, resumable_state
            FROM attention_tasks
            WHERE scheduler_key = %s
            ORDER BY created_seq ASC, task_id ASC
            """,
            (scheduler_key,),
        )
        rows = cur.fetchall()

        cur.execute(
            """
            SELECT
                reservation_id, task_id, resource_id, resource_class, units
            FROM attention_resource_reservations
            WHERE scheduler_key = %s
            ORDER BY resource_class ASC, resource_id ASC, task_id ASC
            """,
            (scheduler_key,),
        )
        reservation_rows = cur.fetchall()

        epoch_row = None
        observation_row = None
        assignment_rows: list[dict[str, Any]] = []
        if state["current_resource_observation_id"] is not None:
            cur.execute(
                """
                SELECT snapshot
                FROM attention_resource_observations
                WHERE scheduler_key = %s AND observation_id = %s
                """,
                (scheduler_key, state["current_resource_observation_id"]),
            )
            observation_row = cur.fetchone()
            if observation_row is None:
                raise ValueError(
                    "Scheduler state references a missing resource observation"
                )
        if state["current_epoch_id"] is not None:
            cur.execute(
                """
                SELECT
                    epoch_id, epoch_sequence, previous_epoch_id,
                    scheduler_cycle, admission_policy_version,
                    assignment_policy_version, status, assignment_ids,
                    reservations, preemption_policy_version,
                    pending_preemptions, preemption_event_ids,
                    executed_preemption_ids, cancelled_preemption_ids,
                    resource_observation_id,
                    resource_safety_policy_version
                FROM attention_scheduling_epochs
                WHERE scheduler_key = %s AND epoch_id = %s
                """,
                (scheduler_key, state["current_epoch_id"]),
            )
            epoch_row = cur.fetchone()
            if epoch_row is None:
                raise ValueError(
                    "Scheduler state references a missing scheduling epoch"
                )
            cur.execute(
                """
                SELECT
                    assignment_id, task_id, task_revision,
                    created_epoch_sequence, reservation_ids, status
                FROM attention_assignments
                WHERE scheduler_key = %s
                """,
                (scheduler_key,),
            )
            assignment_rows = cur.fetchall()

    tasks = [_row_to_task(row) for row in rows]
    reservations = [
        _row_to_resource_reservation(row) for row in reservation_rows
    ]
    current_epoch = (
        _row_to_scheduling_epoch(epoch_row) if epoch_row is not None else None
    )
    current_resource_observation = (
        ResourceObservationSnapshot.model_validate(observation_row["snapshot"])
        if observation_row is not None
        else None
    )
    assignment_by_id = {
        assignment.assignment_id: assignment
        for assignment in (
            _row_to_durable_assignment(row) for row in assignment_rows
        )
    }
    assignments: list[DurableAssignment] = []
    if current_epoch is not None:
        try:
            assignments = [
                assignment_by_id[assignment_id]
                for assignment_id in current_epoch.assignment_ids
            ]
        except KeyError as exc:
            raise ValueError(
                "Scheduling epoch references a missing durable assignment"
            ) from exc
    return JITAttentionScheduler.from_snapshot(
        SchedulerSnapshot(
            cycle=int(state["cycle"]),
            active_task_id=state["active_task_id"],
            pending_preemption_task_id=state["pending_preemption_task_id"],
            tasks=tasks,
            execution_resources=resources,
            admission_policy_version=state["admission_policy_version"],
            admitted_task_ids=list(state["admitted_task_ids"] or []),
            resource_reservations=reservations,
            epoch_sequence=int(state["epoch_sequence"]),
            assignment_policy_version=state["assignment_policy_version"],
            preemption_policy_version=state["preemption_policy_version"],
            preemption_state_revision=int(state["preemption_state_revision"]),
            pending_preemptions=[
                PendingPreemption.model_validate(value)
                for value in (state["pending_preemptions"] or [])
            ],
            current_epoch=current_epoch,
            assignments=assignments,
            resource_safety_required=bool(state["resource_safety_required"]),
            resource_safety_policy_version=(
                state["resource_safety_policy_version"]
            ),
            current_resource_observation=current_resource_observation,
        )
    )


def load_worker_visible_assignments(
    conn: psycopg.Connection,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> list[DurableAssignment]:
    """Load the complete committed assignment set exposed to workers."""

    return load_scheduler(
        conn,
        scheduler_key=scheduler_key,
    ).worker_visible_assignments()


def load_transition_count(
    conn: psycopg.Connection,
    task_id: UUID | None = None,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> int:
    """Small inspection helper used by deterministic persistence tests."""

    with conn.cursor() as cur:
        if task_id is None:
            cur.execute(
                """
                SELECT count(*)
                FROM attention_task_transitions
                WHERE scheduler_key = %s
                """,
                (scheduler_key,),
            )
        else:
            cur.execute(
                """
                SELECT count(*)
                FROM attention_task_transitions
                WHERE scheduler_key = %s AND task_id = %s
                """,
                (scheduler_key, task_id),
            )
        return int(require_row(cur.fetchone(), context="task transition count")[0])


def load_preemption_event_count(
    conn: psycopg.Connection,
    preemption_id: UUID | None = None,
) -> int:
    """Return durable preemption-history size for persistence verification."""

    with conn.cursor() as cur:
        if preemption_id is None:
            cur.execute("SELECT count(*) FROM attention_preemption_events")
        else:
            cur.execute(
                """
                SELECT count(*)
                FROM attention_preemption_events
                WHERE preemption_id = %s
                """,
                (preemption_id,),
            )
        return int(require_row(cur.fetchone(), context="preemption event count")[0])


def _lock_scheduler_generation(
    cur: psycopg.Cursor[Any],
    *,
    scheduler_key: str,
    snapshot_epoch_sequence: int,
    snapshot_epoch_id: UUID | None,
    target_epoch: SchedulingEpoch | None,
    preemption_base_revision: int,
    target_preemption_revision: int,
    target_pending_preemptions: list[PendingPreemption],
) -> None:
    """Reject stale writers before they can move the current-epoch pointer."""

    if target_preemption_revision < preemption_base_revision:
        raise ValueError("Target preemption revision cannot move backward")

    cur.execute(
        """
        SELECT
            epoch_sequence, current_epoch_id,
            preemption_state_revision, pending_preemptions
        FROM attention_scheduler_state
        WHERE scheduler_key = %s
        FOR UPDATE
        """,
        (scheduler_key,),
    )
    row = cur.fetchone()
    expected = (snapshot_epoch_sequence, snapshot_epoch_id)
    if row is None:
        if expected != (0, None) or preemption_base_revision != 0:
            raise RuntimeError(
                "Cannot persist an existing epoch into missing scheduler state"
            )
        return

    state = _cursor_row_to_dict(cur, row)
    actual = (int(state["epoch_sequence"]), state["current_epoch_id"])
    allowed = {expected}
    if target_epoch is not None:
        allowed.add((target_epoch.sequence, target_epoch.epoch_id))
    if actual not in allowed:
        raise RuntimeError("Stale scheduler generation cannot overwrite current epoch")

    actual_preemption_revision = int(state["preemption_state_revision"])
    if actual_preemption_revision == preemption_base_revision:
        return
    if actual_preemption_revision != target_preemption_revision:
        raise RuntimeError("Stale preemption state cannot overwrite checkpoint progress")
    stored_pending = [
        PendingPreemption.model_validate(value).model_dump(mode="json")
        for value in (state["pending_preemptions"] or [])
    ]
    target_pending = [
        intent.model_dump(mode="json") for intent in target_pending_preemptions
    ]
    if stored_pending != target_pending:
        raise RuntimeError("Conflicting preemption state at the same revision")


def _insert_immutable_resource_observation(
    cur: psycopg.Cursor[Any],
    *,
    scheduler_key: str,
    observation: ResourceObservationSnapshot,
) -> None:
    """Persist the exact host-pressure input consumed by an epoch."""

    cur.execute(
        """
        INSERT INTO attention_resource_observations (
            scheduler_key, observation_id, scheduler_cycle,
            captured_at, valid_until, safety_policy_version,
            healthy, snapshot
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            observation.observation_id,
            observation.scheduler_cycle,
            observation.captured_at,
            observation.valid_until,
            observation.safety_policy_version,
            observation.healthy,
            Json(observation.model_dump(mode="json")),
        ),
    )
    cur.execute(
        """
        SELECT snapshot
        FROM attention_resource_observations
        WHERE scheduler_key = %s AND observation_id = %s
        """,
        (scheduler_key, observation.observation_id),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Resource observation insert did not produce a row")
    stored = ResourceObservationSnapshot.model_validate(
        _cursor_row_to_dict(cur, row)["snapshot"]
    )
    if stored.model_dump(mode="json") != observation.model_dump(mode="json"):
        raise RuntimeError("Conflicting immutable resource observation")


def _insert_immutable_assignment(
    cur: psycopg.Cursor[Any],
    *,
    scheduler_key: str,
    assignment: DurableAssignment,
) -> None:
    cur.execute(
        """
        INSERT INTO attention_assignments (
            scheduler_key, assignment_id, task_id, task_revision,
            created_epoch_sequence, reservation_ids, status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            assignment.assignment_id,
            assignment.task_id,
            assignment.task_revision,
            assignment.created_epoch_sequence,
            Json([str(value) for value in assignment.reservation_ids]),
            assignment.status.value,
        ),
    )
    cur.execute(
        """
        SELECT
            assignment_id, task_id, task_revision,
            created_epoch_sequence, reservation_ids, status
        FROM attention_assignments
        WHERE scheduler_key = %s AND assignment_id = %s
        """,
        (scheduler_key, assignment.assignment_id),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Durable assignment insert did not produce a row")
    stored = _row_to_durable_assignment(_cursor_row_to_dict(cur, row))
    if stored.model_dump(mode="json") != assignment.model_dump(mode="json"):
        raise RuntimeError("Conflicting immutable durable assignment")


def _insert_immutable_epoch(
    cur: psycopg.Cursor[Any],
    *,
    scheduler_key: str,
    epoch: SchedulingEpoch,
) -> None:
    if epoch.status is not SchedulingEpochStatus.COMMITTED:
        raise ValueError("Only committed scheduling epochs may be persisted")
    cur.execute(
        """
        INSERT INTO attention_scheduling_epochs (
            scheduler_key, epoch_id, epoch_sequence, previous_epoch_id,
            scheduler_cycle, admission_policy_version,
            assignment_policy_version, status, assignment_ids, reservations,
            preemption_policy_version, pending_preemptions,
            preemption_event_ids, executed_preemption_ids,
            cancelled_preemption_ids, resource_observation_id,
            resource_safety_policy_version
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s
        )
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            epoch.epoch_id,
            epoch.sequence,
            epoch.previous_epoch_id,
            epoch.scheduler_cycle,
            epoch.admission_policy_version,
            epoch.assignment_policy_version,
            epoch.status.value,
            Json([str(value) for value in epoch.assignment_ids]),
            Json(
                [
                    reservation.model_dump(mode="json")
                    for reservation in epoch.reservations
                ]
            ),
            epoch.preemption_policy_version,
            Json(
                [
                    intent.model_dump(mode="json")
                    for intent in epoch.pending_preemptions
                ]
            ),
            Json([str(value) for value in epoch.preemption_event_ids]),
            Json([str(value) for value in epoch.executed_preemption_ids]),
            Json([str(value) for value in epoch.cancelled_preemption_ids]),
            epoch.resource_observation_id,
            epoch.resource_safety_policy_version,
        ),
    )
    cur.execute(
        """
        SELECT
            epoch_id, epoch_sequence, previous_epoch_id,
            scheduler_cycle, admission_policy_version,
            assignment_policy_version, status, assignment_ids, reservations,
            preemption_policy_version, pending_preemptions,
            preemption_event_ids, executed_preemption_ids,
            cancelled_preemption_ids, resource_observation_id,
            resource_safety_policy_version
        FROM attention_scheduling_epochs
        WHERE scheduler_key = %s AND epoch_sequence = %s
        """,
        (scheduler_key, epoch.sequence),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Scheduling epoch insert did not produce a row")
    stored = _row_to_scheduling_epoch(_cursor_row_to_dict(cur, row))
    if stored.model_dump(mode="json") != epoch.model_dump(mode="json"):
        raise RuntimeError("Conflicting immutable scheduling epoch")


def _insert_immutable_preemption_event(
    cur: psycopg.Cursor[Any],
    *,
    scheduler_key: str,
    event: PreemptionEvent,
) -> None:
    cur.execute(
        """
        INSERT INTO attention_preemption_events (
            scheduler_key, event_id, preemption_id, event_type,
            scheduler_cycle, epoch_sequence, target_task_id,
            victim_task_ids, victim_assignment_ids,
            checkpoint_task_id, reason
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            event.event_id,
            event.preemption_id,
            event.event_type.value,
            event.scheduler_cycle,
            event.epoch_sequence,
            event.target_task_id,
            Json([str(value) for value in event.victim_task_ids]),
            Json([str(value) for value in event.victim_assignment_ids]),
            event.checkpoint_task_id,
            event.reason,
        ),
    )
    cur.execute(
        """
        SELECT
            event_id, preemption_id, event_type, scheduler_cycle,
            epoch_sequence, target_task_id, victim_task_ids,
            victim_assignment_ids, checkpoint_task_id, reason
        FROM attention_preemption_events
        WHERE scheduler_key = %s AND event_id = %s
        """,
        (scheduler_key, event.event_id),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Preemption event insert did not produce a row")
    stored = _row_to_preemption_event(_cursor_row_to_dict(cur, row))
    if stored.model_dump(mode="json") != event.model_dump(mode="json"):
        raise RuntimeError("Conflicting immutable preemption event")


def _upsert_execution_resource(
    cur: psycopg.Cursor[Any],
    resource: ExecutionResource,
) -> None:
    cur.execute(
        """
        INSERT INTO attention_execution_resources (
            resource_id, resource_class, capacity, system_headroom, enabled, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (resource_id) DO UPDATE SET
            resource_class = EXCLUDED.resource_class,
            capacity = EXCLUDED.capacity,
            system_headroom = EXCLUDED.system_headroom,
            enabled = EXCLUDED.enabled,
            metadata = EXCLUDED.metadata,
            updated_at = now()
        """,
        (
            resource.resource_id,
            resource.resource_class.value,
            resource.capacity,
            resource.system_headroom,
            resource.enabled,
            Json(resource.metadata),
        ),
    )


def _upsert_task(
    cur: psycopg.Cursor[Any],
    task: AttentionTask,
    *,
    scheduler_key: str,
) -> None:
    metadata = task.metadata
    cur.execute(
        """
        INSERT INTO attention_tasks (
            scheduler_key, task_id, task_key, created_seq, parent_task_id,
            criticality, service_class, interruption_policy, deadline,
            required_capabilities, required_resource_classes,
            resource_requirements, process_resource_estimate, dependency_ids,
            status, enqueued_cycle, revision, resumable_state
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (task_id) DO UPDATE SET
            task_key = EXCLUDED.task_key,
            created_seq = EXCLUDED.created_seq,
            parent_task_id = EXCLUDED.parent_task_id,
            criticality = EXCLUDED.criticality,
            service_class = EXCLUDED.service_class,
            interruption_policy = EXCLUDED.interruption_policy,
            deadline = EXCLUDED.deadline,
            required_capabilities = EXCLUDED.required_capabilities,
            required_resource_classes = EXCLUDED.required_resource_classes,
            resource_requirements = EXCLUDED.resource_requirements,
            process_resource_estimate = EXCLUDED.process_resource_estimate,
            dependency_ids = EXCLUDED.dependency_ids,
            status = EXCLUDED.status,
            enqueued_cycle = EXCLUDED.enqueued_cycle,
            revision = EXCLUDED.revision,
            resumable_state = EXCLUDED.resumable_state,
            updated_at = now()
        WHERE attention_tasks.scheduler_key = EXCLUDED.scheduler_key
        """,
        (
            scheduler_key,
            task.task_id,
            task.task_key,
            task.created_seq,
            task.parent_task_id,
            metadata.criticality.value,
            metadata.service_class.value,
            metadata.interruption_policy.value,
            metadata.deadline,
            Json(metadata.required_capabilities),
            Json([value.value for value in metadata.required_resource_classes]),
            Json(
                [
                    requirement.model_dump(mode="json")
                    for requirement in metadata.resource_requirements
                ]
            ),
            (
                Json(metadata.process_resource_estimate.model_dump(mode="json"))
                if metadata.process_resource_estimate is not None
                else None
            ),
            Json([str(value) for value in metadata.dependency_ids]),
            task.status.value,
            task.enqueued_cycle,
            task.revision,
            Json(task.resumable_state),
        ),
    )
    if cur.rowcount != 1:
        raise ValueError(
            f"Task {task.task_id} already belongs to another scheduler namespace"
        )


def _replace_resource_reservations(
    cur: psycopg.Cursor[Any],
    *,
    scheduler_key: str,
    reservations: list[ResourceReservation],
    assignments: list[DurableAssignment],
    observed_admission_limits: dict[str, int] | None = None,
) -> None:
    """Atomically replace one scheduler's complete authoritative reservation set."""

    # An assignment entitlement may be removed by a later epoch, but capacity
    # borrowed by a still-live Increment F worker may not disappear underneath
    # the process.  A cooperative worker must first checkpoint/release/complete;
    # expired leases are made abandoned before the replacement is evaluated.
    cur.execute(
        """
        UPDATE attention_worker_claims
        SET status = 'ABANDONED', updated_at = now()
        WHERE scheduler_key = %s
          AND status = 'ACTIVE'
          AND lease_expires_at <= now()
        """,
        (scheduler_key,),
    )
    cur.execute(
        """
        SELECT c.claim_id, c.assignment_id, s.reservation_ids
        FROM attention_worker_claims c
        JOIN attention_worker_steps s
          ON s.scheduler_key = c.scheduler_key AND s.step_id = c.step_id
        WHERE c.scheduler_key = %s AND c.status = 'ACTIVE'
        ORDER BY c.claim_id ASC
        """,
        (scheduler_key,),
    )
    active_claim_rows = [
        _cursor_row_to_dict(cur, row) for row in cur.fetchall()
    ]
    target_assignments = {
        assignment.assignment_id: assignment for assignment in assignments
    }
    target_reservations = {
        reservation.reservation_id: reservation for reservation in reservations
    }
    held_reservation_ids = {
        UUID(str(value))
        for row in active_claim_rows
        for value in (row["reservation_ids"] or [])
    }
    current_reservations: dict[UUID, ResourceReservation] = {}
    if held_reservation_ids:
        cur.execute(
            """
            SELECT reservation_id, task_id, resource_id, resource_class, units
            FROM attention_resource_reservations
            WHERE scheduler_key = %s AND reservation_id = ANY(%s)
            """,
            (
                scheduler_key,
                sorted(held_reservation_ids, key=lambda value: value.hex),
            ),
        )
        current_reservations = {
            reservation.reservation_id: reservation
            for reservation in (
                _row_to_resource_reservation(_cursor_row_to_dict(cur, value))
                for value in cur.fetchall()
            )
        }
    for row in active_claim_rows:
        assignment_id = UUID(str(row["assignment_id"]))
        held = {UUID(str(value)) for value in (row["reservation_ids"] or [])}
        target_assignment = target_assignments.get(assignment_id)
        if target_assignment is None or set(target_assignment.reservation_ids) != held:
            raise RuntimeError(
                "Cannot replace an assignment held by an active worker claim"
            )
        for reservation_id in held:
            current = current_reservations.get(reservation_id)
            target = target_reservations.get(reservation_id)
            if current is None or target is None or current != target:
                raise RuntimeError(
                    "Cannot alter a reservation held by an active worker claim"
                )

    target_usage: dict[str, int] = {}
    for reservation in reservations:
        target_usage[reservation.resource_id] = (
            target_usage.get(reservation.resource_id, 0) + reservation.units
        )
    resource_ids = sorted(target_usage)
    locked_resources: dict[str, dict[str, Any]] = {}
    if resource_ids:
        # Lock the physical pools in one global order. This makes capacity a
        # system-wide invariant across processes and scheduler keys rather than
        # an in-memory promise each scheduler can independently overbook.
        cur.execute(
            """
            SELECT resource_id, capacity, system_headroom, enabled
            FROM attention_execution_resources
            WHERE resource_id = ANY(%s)
            ORDER BY resource_id ASC
            FOR UPDATE
            """,
            (resource_ids,),
        )
        locked_resources = {
            row["resource_id"]: row
            for row in (
                _cursor_row_to_dict(cur, value) for value in cur.fetchall()
            )
        }
        if set(locked_resources) != set(resource_ids):
            raise RuntimeError("Reservation references an unknown resource pool")

    cur.execute(
        "DELETE FROM attention_resource_reservations WHERE scheduler_key = %s",
        (scheduler_key,),
    )
    existing_usage: dict[str, int] = {}
    if resource_ids:
        cur.execute(
            """
            SELECT resource_id, sum(units) AS units
            FROM attention_resource_reservations
            WHERE resource_id = ANY(%s)
            GROUP BY resource_id
            """,
            (resource_ids,),
        )
        existing_usage = {
            row["resource_id"]: int(row["units"])
            for row in (
                _cursor_row_to_dict(cur, value) for value in cur.fetchall()
            )
        }
        for resource_id, requested_units in target_usage.items():
            resource = locked_resources[resource_id]
            admissible_capacity = (
                int(resource["capacity"]) - int(resource["system_headroom"])
                if bool(resource["enabled"])
                else 0
            )
            if observed_admission_limits is not None:
                if resource_id not in observed_admission_limits:
                    raise RuntimeError(
                        "Observed admission limits are incomplete"
                    )
                admissible_capacity = min(
                    admissible_capacity,
                    observed_admission_limits[resource_id],
                )
            if (
                existing_usage.get(resource_id, 0) + requested_units
                > admissible_capacity
            ):
                raise RuntimeError(
                    f"Global reservation capacity exceeded for {resource_id!r}"
                )
    for reservation in reservations:
        cur.execute(
            """
            INSERT INTO attention_resource_reservations (
                scheduler_key, reservation_id, task_id,
                resource_id, resource_class, units
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                scheduler_key,
                reservation.reservation_id,
                reservation.task_id,
                reservation.resource_id,
                reservation.resource_class.value,
                reservation.units,
            ),
        )


def _row_to_execution_resource(row: dict[str, Any]) -> ExecutionResource:
    return ExecutionResource(
        resource_id=row["resource_id"],
        resource_class=ExecutionResourceClass(row["resource_class"]),
        capacity=int(row["capacity"]),
        system_headroom=int(row["system_headroom"]),
        enabled=bool(row["enabled"]),
        metadata=dict(row["metadata"] or {}),
    )


def _row_to_task(row: dict[str, Any]) -> AttentionTask:
    return AttentionTask(
        task_id=row["task_id"],
        task_key=row["task_key"],
        created_seq=int(row["created_seq"]),
        parent_task_id=row["parent_task_id"],
        metadata=SchedulingMetadata(
            criticality=TaskCriticality(row["criticality"]),
            service_class=ServiceClass(row["service_class"]),
            interruption_policy=InterruptionPolicy(row["interruption_policy"]),
            deadline=row["deadline"],
            required_capabilities=list(row["required_capabilities"] or []),
            required_resource_classes=[
                ExecutionResourceClass(value)
                for value in (row["required_resource_classes"] or [])
            ],
            resource_requirements=[
                ResourceRequirement.model_validate(value)
                for value in (row["resource_requirements"] or [])
            ],
            process_resource_estimate=(
                ProcessResourceEstimate.model_validate(
                    row["process_resource_estimate"]
                )
                if row["process_resource_estimate"] is not None
                else None
            ),
            dependency_ids=list(row["dependency_ids"] or []),
        ),
        status=TaskStatus(row["status"]),
        enqueued_cycle=int(row["enqueued_cycle"]),
        revision=int(row["revision"]),
        resumable_state=dict(row["resumable_state"] or {}),
    )


def _row_to_resource_reservation(row: dict[str, Any]) -> ResourceReservation:
    return ResourceReservation(
        reservation_id=row["reservation_id"],
        task_id=row["task_id"],
        resource_id=row["resource_id"],
        resource_class=ExecutionResourceClass(row["resource_class"]),
        units=int(row["units"]),
    )


def _row_to_durable_assignment(row: dict[str, Any]) -> DurableAssignment:
    return DurableAssignment(
        assignment_id=row["assignment_id"],
        task_id=row["task_id"],
        task_revision=int(row["task_revision"]),
        created_epoch_sequence=int(row["created_epoch_sequence"]),
        reservation_ids=list(row["reservation_ids"] or []),
        status=AssignmentStatus(row["status"]),
    )


def _row_to_scheduling_epoch(row: dict[str, Any]) -> SchedulingEpoch:
    return SchedulingEpoch(
        epoch_id=row["epoch_id"],
        sequence=int(row["epoch_sequence"]),
        previous_epoch_id=row["previous_epoch_id"],
        scheduler_cycle=int(row["scheduler_cycle"]),
        admission_policy_version=row["admission_policy_version"],
        assignment_policy_version=row["assignment_policy_version"],
        status=SchedulingEpochStatus(row["status"]),
        assignment_ids=list(row["assignment_ids"] or []),
        reservations=[
            ResourceReservation.model_validate(value)
            for value in (row["reservations"] or [])
        ],
        preemption_policy_version=row["preemption_policy_version"],
        pending_preemptions=[
            PendingPreemption.model_validate(value)
            for value in (row["pending_preemptions"] or [])
        ],
        preemption_event_ids=list(row["preemption_event_ids"] or []),
        executed_preemption_ids=list(row["executed_preemption_ids"] or []),
        cancelled_preemption_ids=list(row["cancelled_preemption_ids"] or []),
        resource_observation_id=row["resource_observation_id"],
        resource_safety_policy_version=row["resource_safety_policy_version"],
    )


def _row_to_preemption_event(row: dict[str, Any]) -> PreemptionEvent:
    return PreemptionEvent(
        event_id=row["event_id"],
        preemption_id=row["preemption_id"],
        event_type=PreemptionEventType(row["event_type"]),
        scheduler_cycle=int(row["scheduler_cycle"]),
        epoch_sequence=int(row["epoch_sequence"]),
        target_task_id=row["target_task_id"],
        victim_task_ids=list(row["victim_task_ids"] or []),
        victim_assignment_ids=list(row["victim_assignment_ids"] or []),
        checkpoint_task_id=row["checkpoint_task_id"],
        reason=row["reason"],
    )


def _cursor_row_to_dict(
    cur: psycopg.Cursor[Any],
    row: Any,
) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if cur.description is None:
        raise RuntimeError("Cursor has no result description")
    column_names = [description.name for description in cur.description]
    return dict(zip(column_names, row, strict=True))
