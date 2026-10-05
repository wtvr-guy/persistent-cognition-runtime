from __future__ import annotations

from uuid import UUID

import pytest

from prometheist import db
from prometheist.attention import (
    AttentionTask,
    JITAttentionScheduler,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    deterministic_task_id,
)
from prometheist.attention_resources import (
    ExecutionResource,
    ExecutionResourceClass,
    ResourceRequirement,
)
from prometheist.attention_store import load_scheduler, save_scheduler


NAMESPACE = UUID("88888888-8888-8888-8888-888888888888")


def _task(
    key: str,
    seq: int,
    criticality: TaskCriticality,
    *,
    requirements: list[tuple[ExecutionResourceClass, int]] | None = None,
    required_classes: list[ExecutionResourceClass] | None = None,
) -> AttentionTask:
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=seq,
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=ServiceClass.USER_WORK,
            required_resource_classes=required_classes or [],
            resource_requirements=[
                ResourceRequirement(resource_class=resource_class, units=units)
                for resource_class, units in (requirements or [])
            ],
        ),
    )


def _configure(
    scheduler: JITAttentionScheduler,
    resources: list[ExecutionResource],
) -> None:
    for resource in resources:
        scheduler.configure_execution_resource(resource)


def test_execution_resource_rejects_headroom_above_capacity():
    with pytest.raises(ValueError, match="must not exceed capacity"):
        ExecutionResource(
            resource_id="unsafe",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=1,
            system_headroom=2,
        )


def test_identical_state_produces_identical_admission_independent_of_configuration_order():
    resources = [
        ExecutionResource(
            resource_id="network",
            resource_class=ExecutionResourceClass.NETWORK_IO,
            capacity=2,
        ),
        ExecutionResource(
            resource_id="cpu-z",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=2,
        ),
        ExecutionResource(
            resource_id="cpu-a",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=1,
        ),
    ]

    def run(configured: list[ExecutionResource]):
        scheduler = JITAttentionScheduler()
        _configure(scheduler, configured)
        scheduler.submit(
            _task(
                "interactive",
                1,
                TaskCriticality.USER_BLOCKING,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
            )
        )
        scheduler.submit(
            _task(
                "support",
                2,
                TaskCriticality.SUPPORTING,
                requirements=[
                    (ExecutionResourceClass.CPU_GENERAL, 1),
                    (ExecutionResourceClass.NETWORK_IO, 1),
                ],
            )
        )
        return (
            scheduler.reconcile_resource_admission().model_dump(mode="json"),
            scheduler.snapshot().model_dump(mode="json"),
        )

    assert run(resources) == run(list(reversed(resources)))


def test_priority_does_not_serialize_tasks_that_fit_concurrently():
    scheduler = JITAttentionScheduler()
    _configure(
        scheduler,
        [
            ExecutionResource(
                resource_id="cpu",
                resource_class=ExecutionResourceClass.CPU_GENERAL,
                capacity=4,
            )
        ],
    )
    high = scheduler.submit(
        _task(
            "high",
            1,
            TaskCriticality.USER_BLOCKING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )
    low = scheduler.submit(
        _task(
            "low",
            2,
            TaskCriticality.SUPPORTING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )

    plan = scheduler.reconcile_resource_admission()

    assert plan.admitted_task_ids == [high.task_id, low.task_id]
    assert plan.unadmitted_task_ids == []
    assert sum(item.units for item in plan.reservations) == 4


def test_headroom_is_never_allocatable_to_ordinary_work():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="cpu",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=4,
            system_headroom=2,
        )
    )
    task = scheduler.submit(
        _task(
            "too-large-for-safe-envelope",
            1,
            TaskCriticality.USER_REQUESTED,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 3)],
        )
    )

    plan = scheduler.reconcile_resource_admission()

    assert plan.admitted_task_ids == []
    assert plan.unadmitted_task_ids == [task.task_id]
    assert plan.reservations == []


def test_admission_never_oversubscribes_safe_capacity():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="cpu",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=4,
            system_headroom=1,
        )
    )
    high = scheduler.submit(
        _task(
            "high",
            1,
            TaskCriticality.USER_BLOCKING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )
    low = scheduler.submit(
        _task(
            "low",
            2,
            TaskCriticality.SUPPORTING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )

    plan = scheduler.reconcile_resource_admission()

    assert plan.admitted_task_ids == [high.task_id]
    assert plan.unadmitted_task_ids == [low.task_id]
    assert sum(item.units for item in plan.reservations) <= 3


def test_failed_multi_resource_candidate_does_not_consume_partial_capacity():
    scheduler = JITAttentionScheduler()
    _configure(
        scheduler,
        [
            ExecutionResource(
                resource_id="cpu",
                resource_class=ExecutionResourceClass.CPU_GENERAL,
                capacity=2,
            ),
            ExecutionResource(
                resource_id="database",
                resource_class=ExecutionResourceClass.DATABASE,
                capacity=1,
            ),
        ],
    )
    impossible = scheduler.submit(
        _task(
            "impossible",
            1,
            TaskCriticality.USER_BLOCKING,
            requirements=[
                (ExecutionResourceClass.CPU_GENERAL, 2),
                (ExecutionResourceClass.DATABASE, 2),
            ],
        )
    )
    feasible = scheduler.submit(
        _task(
            "feasible",
            2,
            TaskCriticality.USER_REQUESTED,
            requirements=[
                (ExecutionResourceClass.CPU_GENERAL, 2),
                (ExecutionResourceClass.DATABASE, 1),
            ],
        )
    )

    plan = scheduler.reconcile_resource_admission()

    assert plan.admitted_task_ids == [feasible.task_id]
    assert plan.unadmitted_task_ids == [impossible.task_id]
    assert {item.task_id for item in plan.reservations} == {feasible.task_id}


def test_requirement_splits_across_ordered_fungible_pools_deterministically():
    scheduler = JITAttentionScheduler()
    _configure(
        scheduler,
        [
            ExecutionResource(
                resource_id="cpu-z",
                resource_class=ExecutionResourceClass.CPU_GENERAL,
                capacity=2,
            ),
            ExecutionResource(
                resource_id="cpu-a",
                resource_class=ExecutionResourceClass.CPU_GENERAL,
                capacity=1,
            ),
        ],
    )
    task = scheduler.submit(
        _task(
            "split",
            1,
            TaskCriticality.USER_REQUESTED,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 3)],
        )
    )

    plan = scheduler.reconcile_resource_admission()

    assert plan.admitted_task_ids == [task.task_id]
    assert [
        (item.resource_id, item.units) for item in plan.reservations
    ] == [("cpu-a", 1), ("cpu-z", 2)]


def test_increment_b_class_requirement_defaults_to_one_unit():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="llm",
            resource_class=ExecutionResourceClass.LLM_INFERENCE,
            capacity=1,
        )
    )
    task = scheduler.submit(
        _task(
            "legacy-contract",
            1,
            TaskCriticality.USER_REQUESTED,
            required_classes=[ExecutionResourceClass.LLM_INFERENCE],
        )
    )

    plan = scheduler.reconcile_resource_admission()

    assert plan.admitted_task_ids == [task.task_id]
    assert len(plan.reservations) == 1
    assert plan.reservations[0].units == 1


def test_running_focus_is_preserved_until_contention_preemption_exists():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="cpu",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=2,
        )
    )
    active = scheduler.submit(
        _task(
            "active-low",
            1,
            TaskCriticality.SUPPORTING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )
    scheduler.reconcile_focus()
    urgent = scheduler.submit(
        _task(
            "urgent",
            2,
            TaskCriticality.USER_BLOCKING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )

    plan = scheduler.reconcile_resource_admission()

    assert plan.admitted_task_ids == [active.task_id]
    assert plan.unadmitted_task_ids == [urgent.task_id]


def test_scheduling_mutation_invalidates_stale_admission_state():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="cpu",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=2,
        )
    )
    scheduler.submit(
        _task(
            "first",
            1,
            TaskCriticality.USER_REQUESTED,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.reconcile_resource_admission()
    assert scheduler.admitted_task_ids
    assert scheduler.resource_reservations()

    scheduler.submit(
        _task(
            "new-arrival",
            2,
            TaskCriticality.USER_BLOCKING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )

    assert scheduler.admitted_task_ids == []
    assert scheduler.resource_reservations() == []


def test_admission_snapshot_round_trip_is_exact():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="cpu",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=3,
            system_headroom=1,
        )
    )
    scheduler.submit(
        _task(
            "durable",
            1,
            TaskCriticality.USER_REQUESTED,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )
    scheduler.reconcile_resource_admission()

    restored = JITAttentionScheduler.from_snapshot(scheduler.snapshot())

    assert restored.snapshot().model_dump(mode="json") == scheduler.snapshot().model_dump(
        mode="json"
    )


def test_tampered_snapshot_cannot_oversubscribe_a_resource():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="cpu",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=2,
        )
    )
    scheduler.submit(
        _task(
            "tamper-target",
            1,
            TaskCriticality.USER_REQUESTED,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )
    scheduler.reconcile_resource_admission()
    snapshot = scheduler.snapshot()
    snapshot.resource_reservations[0].units = 3

    with pytest.raises(ValueError, match="exceed admissible capacity"):
        JITAttentionScheduler.from_snapshot(snapshot)


def test_resource_admission_round_trips_through_postgres():
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        _configure(
            scheduler,
            [
                ExecutionResource(
                    resource_id="cpu",
                    resource_class=ExecutionResourceClass.CPU_GENERAL,
                    capacity=4,
                    system_headroom=1,
                ),
                ExecutionResource(
                    resource_id="database",
                    resource_class=ExecutionResourceClass.DATABASE,
                    capacity=1,
                ),
            ],
        )
        scheduler.submit(
            _task(
                "postgres-durable",
                1,
                TaskCriticality.USER_REQUESTED,
                requirements=[
                    (ExecutionResourceClass.CPU_GENERAL, 3),
                    (ExecutionResourceClass.DATABASE, 1),
                ],
            )
        )
        scheduler.reconcile_resource_admission()
        expected = scheduler.snapshot().model_dump(mode="json")
        save_scheduler(conn, scheduler)
    finally:
        conn.close()

    restarted_conn = db.get_connection()
    try:
        restored = load_scheduler(restarted_conn)
        assert restored.snapshot().model_dump(mode="json") == expected

        restored.advance_cycle()
        save_scheduler(restarted_conn, restored)
    finally:
        restarted_conn.close()

    final_conn = db.get_connection()
    try:
        final = load_scheduler(final_conn)
        assert final.admitted_task_ids == []
        assert final.resource_reservations() == []
    finally:
        final_conn.close()
