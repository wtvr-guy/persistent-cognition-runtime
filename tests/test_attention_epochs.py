from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from prometheist import attention_store, db
from prometheist.attention import (
    AttentionTask,
    JITAttentionScheduler,
    SchedulerSnapshot,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    deterministic_task_id,
)
from prometheist.attention_assignments import (
    SchedulingEpochStatus,
    deterministic_epoch_id,
)
from prometheist.attention_resources import (
    ExecutionResource,
    ExecutionResourceClass,
    ResourceRequirement,
)
from prometheist.attention_store import (
    load_scheduler,
    load_worker_visible_assignments,
    save_scheduler,
)


NAMESPACE = UUID("99999999-9999-9999-9999-999999999999")


def _task(
    key: str,
    seq: int,
    criticality: TaskCriticality = TaskCriticality.USER_REQUESTED,
    *,
    requirements: list[tuple[ExecutionResourceClass, int]] | None = None,
    deadline: datetime | None = None,
) -> AttentionTask:
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=seq,
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=ServiceClass.USER_WORK,
            deadline=deadline,
            resource_requirements=[
                ResourceRequirement(resource_class=resource_class, units=units)
                for resource_class, units in (requirements or [])
            ],
        ),
    )


def _resource(
    resource_id: str,
    resource_class: ExecutionResourceClass,
    capacity: int,
    *,
    system_headroom: int = 0,
) -> ExecutionResource:
    return ExecutionResource(
        resource_id=resource_id,
        resource_class=resource_class,
        capacity=capacity,
        system_headroom=system_headroom,
    )


def _commit_in_memory(scheduler: JITAttentionScheduler) -> None:
    plan = scheduler.pending_scheduling_epoch()
    assert plan is not None
    scheduler._commit_pending_epoch(plan.epoch.epoch_id)


def test_identical_state_produces_identical_complete_epoch_plan():
    resources = [
        _resource("cpu-z", ExecutionResourceClass.CPU_GENERAL, 2),
        _resource("cpu-a", ExecutionResourceClass.CPU_GENERAL, 1),
        _resource("llm", ExecutionResourceClass.LLM_INFERENCE, 1),
    ]
    tasks = [
        _task(
            "support",
            2,
            TaskCriticality.SUPPORTING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        ),
        _task(
            "interactive",
            1,
            TaskCriticality.USER_BLOCKING,
            requirements=[
                (ExecutionResourceClass.CPU_GENERAL, 2),
                (ExecutionResourceClass.LLM_INFERENCE, 1),
            ],
        ),
    ]

    def run(*, reverse: bool) -> dict[str, object]:
        scheduler = JITAttentionScheduler()
        for resource in reversed(resources) if reverse else resources:
            scheduler.configure_execution_resource(resource)
        for task in reversed(tasks) if reverse else tasks:
            scheduler.submit(task)
        return scheduler.plan_scheduling_epoch().model_dump(mode="json")

    assert run(reverse=False) == run(reverse=True)


def test_epoch_assignments_have_a_total_attention_order():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 3)
    )
    later = datetime(2026, 8, 26, tzinfo=timezone.utc)
    sooner = datetime(2026, 8, 25, tzinfo=timezone.utc)
    tasks = [
        scheduler.submit(
            _task(
                "later",
                1,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
                deadline=later,
            )
        ),
        scheduler.submit(
            _task(
                "sooner-first",
                2,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
                deadline=sooner,
            )
        ),
        scheduler.submit(
            _task(
                "sooner-second",
                3,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
                deadline=sooner,
            )
        ),
    ]

    plan = scheduler.plan_scheduling_epoch()

    assert plan.admitted_task_ids == [
        tasks[1].task_id,
        tasks[2].task_id,
        tasks[0].task_id,
    ]
    assert [assignment.task_id for assignment in plan.assignments] == (
        plan.admitted_task_ids
    )


def test_provisional_epoch_is_invisible_until_committed_as_a_complete_set():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
    )
    scheduler.submit(
        _task(
            "one",
            1,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.submit(
        _task(
            "two",
            2,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )

    plan = scheduler.plan_scheduling_epoch()

    assert scheduler.worker_visible_assignments() == []
    assert scheduler.snapshot().current_epoch is None
    assert scheduler.cycle == 0

    _commit_in_memory(scheduler)

    assert scheduler.current_epoch is not None
    assert scheduler.current_epoch.status is SchedulingEpochStatus.COMMITTED
    assert scheduler.cycle == plan.epoch.scheduler_cycle
    assert [
        assignment.assignment_id
        for assignment in scheduler.worker_visible_assignments()
    ] == plan.epoch.assignment_ids


def test_same_priority_arrival_preserves_assignment_and_reservation_identity():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
    )
    original = scheduler.submit(
        _task(
            "original",
            1,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            deadline=datetime(2026, 8, 27, tzinfo=timezone.utc),
        )
    )
    first_plan = scheduler.plan_scheduling_epoch()
    _commit_in_memory(scheduler)
    original_assignment = first_plan.assignments[0]
    original_reservations = list(original_assignment.reservation_ids)

    arrival = scheduler.submit(
        _task(
            "arrival",
            2,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            deadline=datetime(2026, 8, 25, tzinfo=timezone.utc),
        )
    )
    second_plan = scheduler.plan_scheduling_epoch()

    assert second_plan.admitted_task_ids == [original.task_id, arrival.task_id]
    assert second_plan.assignments[0].assignment_id == (
        original_assignment.assignment_id
    )
    assert second_plan.assignments[0].reservation_ids == original_reservations
    assert second_plan.assignments[0].created_epoch_sequence == 1


def test_higher_priority_arrival_displaces_compatible_preemptible_work_in_increment_e():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("llm", ExecutionResourceClass.LLM_INFERENCE, 1)
    )
    existing = scheduler.submit(
        _task(
            "existing",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.LLM_INFERENCE, 1)],
        )
    )
    first_plan = scheduler.plan_scheduling_epoch()
    _commit_in_memory(scheduler)
    urgent = scheduler.submit(
        _task(
            "urgent",
            2,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.LLM_INFERENCE, 1)],
        )
    )

    next_plan = scheduler.plan_scheduling_epoch()

    assert next_plan.admitted_task_ids == [urgent.task_id]
    assert next_plan.unadmitted_task_ids == [existing.task_id]
    assert next_plan.assignments[0].assignment_id != (
        first_plan.assignments[0].assignment_id
    )
    assert next_plan.epoch.executed_preemption_ids


def test_discrete_capacity_and_headroom_are_never_oversubscribed():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("llm", ExecutionResourceClass.LLM_INFERENCE, 1)
    )
    scheduler.configure_execution_resource(
        _resource(
            "cpu",
            ExecutionResourceClass.CPU_GENERAL,
            4,
            system_headroom=1,
        )
    )
    for seq in range(1, 4):
        scheduler.submit(
            _task(
                f"task-{seq}",
                seq,
                requirements=[
                    (ExecutionResourceClass.LLM_INFERENCE, 1),
                    (ExecutionResourceClass.CPU_GENERAL, 2),
                ],
            )
        )

    plan = scheduler.plan_scheduling_epoch()
    usage = {
        resource_id: sum(
            reservation.units
            for reservation in plan.epoch.reservations
            if reservation.resource_id == resource_id
        )
        for resource_id in ("llm", "cpu")
    }

    assert len(plan.admitted_task_ids) == 1
    assert usage == {"llm": 1, "cpu": 2}


def test_committed_epoch_snapshot_round_trip_is_exact_and_tamper_evident():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
    )
    scheduler.submit(
        _task(
            "durable",
            1,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit_in_memory(scheduler)
    snapshot = scheduler.snapshot()

    restored = JITAttentionScheduler.from_snapshot(snapshot)
    assert restored.snapshot().model_dump(mode="json") == snapshot.model_dump(
        mode="json"
    )

    tampered = SchedulerSnapshot.model_validate(snapshot.model_dump(mode="python"))
    tampered.assignments[0].task_revision += 1
    with pytest.raises(ValueError, match="task revision"):
        JITAttentionScheduler.from_snapshot(tampered)


def test_increment_d_epoch_identity_remains_valid_after_preemption_migration():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
    )
    scheduler.submit(
        _task(
            "legacy-durable",
            1,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit_in_memory(scheduler)
    snapshot = scheduler.snapshot()
    assert snapshot.current_epoch is not None
    epoch = snapshot.current_epoch
    legacy_id = deterministic_epoch_id(
        sequence=epoch.sequence,
        previous_epoch_id=epoch.previous_epoch_id,
        scheduler_cycle=epoch.scheduler_cycle,
        admission_policy_version=epoch.admission_policy_version,
        assignment_policy_version=epoch.assignment_policy_version,
        assignment_ids=epoch.assignment_ids,
        reservations=epoch.reservations,
    )
    snapshot.current_epoch = epoch.model_copy(
        update={
            "epoch_id": legacy_id,
            "preemption_policy_version": None,
            "pending_preemptions": [],
            "preemption_event_ids": [],
            "executed_preemption_ids": [],
            "cancelled_preemption_ids": [],
        },
        deep=True,
    )

    restored = JITAttentionScheduler.from_snapshot(snapshot)

    assert restored.current_epoch is not None
    assert restored.current_epoch.epoch_id == legacy_id
    assert restored.current_epoch.preemption_policy_version is None


def test_failed_postgres_epoch_transaction_exposes_no_partial_state(monkeypatch):
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        scheduler.configure_execution_resource(
            _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
        )
        scheduler.submit(
            _task(
                "rollback",
                1,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        scheduler.plan_scheduling_epoch()

        def fail_before_publication(*args, **kwargs):
            raise RuntimeError("injected persistence failure")

        monkeypatch.setattr(
            attention_store,
            "_replace_resource_reservations",
            fail_before_publication,
        )
        with pytest.raises(RuntimeError, match="injected persistence failure"):
            save_scheduler(conn, scheduler)

        assert scheduler.worker_visible_assignments() == []
        assert scheduler.pending_scheduling_epoch() is not None
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM attention_scheduling_epochs")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM attention_assignments")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM attention_scheduler_state")
            assert cur.fetchone()[0] == 0
    finally:
        conn.close()


def test_committed_epoch_round_trips_through_postgres_without_partial_visibility():
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        scheduler.configure_execution_resource(
            _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
        )
        first = scheduler.submit(
            _task(
                "first",
                1,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        scheduler.plan_scheduling_epoch()

        assert load_worker_visible_assignments(conn) == []
        save_scheduler(conn, scheduler)
        first_visible = scheduler.worker_visible_assignments()
        assert [assignment.task_id for assignment in first_visible] == [first.task_id]

        second = scheduler.submit(
            _task(
                "second",
                2,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        scheduler.plan_scheduling_epoch()
        assert [
            assignment.task_id
            for assignment in load_worker_visible_assignments(conn)
        ] == [first.task_id]

        save_scheduler(conn, scheduler)
        expected = scheduler.snapshot().model_dump(mode="json")
        save_scheduler(conn, scheduler)
    finally:
        conn.close()

    restarted_conn = db.get_connection()
    try:
        restored = load_scheduler(restarted_conn)
        assert restored.snapshot().model_dump(mode="json") == expected
        assert [
            assignment.task_id
            for assignment in load_worker_visible_assignments(restarted_conn)
        ] == [first.task_id, second.task_id]
    finally:
        restarted_conn.close()


def test_stale_postgres_scheduler_cannot_overwrite_a_committed_epoch():
    first_conn = db.get_connection()
    stale_conn = db.get_connection()
    try:
        first = load_scheduler(first_conn)
        stale = load_scheduler(stale_conn)
        for scheduler in (first, stale):
            scheduler.configure_execution_resource(
                _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
            )

        first.submit(
            _task(
                "winner",
                1,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        first.plan_scheduling_epoch()
        save_scheduler(first_conn, first)

        stale.submit(
            _task(
                "stale",
                1,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        stale.plan_scheduling_epoch()
        with pytest.raises(RuntimeError, match="Stale scheduler generation"):
            save_scheduler(stale_conn, stale)

        assert [
            assignment.task_id
            for assignment in load_worker_visible_assignments(first_conn)
        ] == [first.worker_visible_assignments()[0].task_id]
    finally:
        stale_conn.close()
        first_conn.close()
