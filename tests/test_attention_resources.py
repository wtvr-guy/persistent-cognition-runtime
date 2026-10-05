from __future__ import annotations

from uuid import UUID

from prometheist import db
from prometheist.attention import (
    AttentionTask,
    FocusAction,
    JITAttentionScheduler,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    TaskStatus,
    deterministic_task_id,
)
from prometheist.attention_resources import ExecutionResource, ExecutionResourceClass
from prometheist.attention_store import allocate_created_seq, load_scheduler, save_scheduler


NAMESPACE = UUID("77777777-7777-7777-7777-777777777777")


def _task(
    key: str,
    seq: int,
    *,
    required_resources: list[ExecutionResourceClass] | None = None,
) -> AttentionTask:
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=seq,
        metadata=SchedulingMetadata(
            criticality=TaskCriticality.USER_REQUESTED,
            service_class=ServiceClass.USER_WORK,
            required_resource_classes=required_resources or [],
        ),
    )


def test_execution_resource_order_is_deterministic_independent_of_configuration_order():
    expected = ["cpu-a", "cpu-z", "llm-a", "network-z"]

    first = JITAttentionScheduler()
    second = JITAttentionScheduler()
    resources = [
        ExecutionResource(
            resource_id="network-z",
            resource_class=ExecutionResourceClass.NETWORK_IO,
            capacity=2,
        ),
        ExecutionResource(
            resource_id="cpu-z",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=4,
        ),
        ExecutionResource(
            resource_id="llm-a",
            resource_class=ExecutionResourceClass.LLM_INFERENCE,
            capacity=1,
        ),
        ExecutionResource(
            resource_id="cpu-a",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=1,
        ),
    ]

    for resource in resources:
        first.configure_execution_resource(resource)
    for resource in reversed(resources):
        second.configure_execution_resource(resource)

    assert [resource.resource_id for resource in first.execution_resources()] == expected
    assert [resource.resource_id for resource in second.execution_resources()] == expected
    assert first.snapshot().model_dump(mode="json") == second.snapshot().model_dump(mode="json")


def test_task_requiring_unsupported_resource_remains_queued():
    scheduler = JITAttentionScheduler()
    task = scheduler.submit(
        _task(
            "needs-llm",
            1,
            required_resources=[ExecutionResourceClass.LLM_INFERENCE],
        )
    )

    decision = scheduler.reconcile_focus()

    assert decision.action is FocusAction.IDLE
    assert scheduler.active_task_id is None
    assert scheduler.tasks[task.task_id].status is TaskStatus.QUEUED
    assert scheduler.queued_tasks() == []


def test_disabled_resource_does_not_make_task_runnable():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="local-llm",
            resource_class=ExecutionResourceClass.LLM_INFERENCE,
            capacity=1,
            enabled=False,
        )
    )
    task = scheduler.submit(
        _task(
            "disabled-llm-task",
            1,
            required_resources=[ExecutionResourceClass.LLM_INFERENCE],
        )
    )

    decision = scheduler.reconcile_focus()

    assert decision.action is FocusAction.IDLE
    assert scheduler.tasks[task.task_id].status is TaskStatus.QUEUED
    assert scheduler.execution_resources(enabled_only=True) == []


def test_task_becomes_runnable_when_all_required_resource_classes_are_enabled():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="cpu",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=2,
        )
    )
    task = scheduler.submit(
        _task(
            "cpu-plus-db",
            1,
            required_resources=[
                ExecutionResourceClass.CPU_GENERAL,
                ExecutionResourceClass.DATABASE,
            ],
        )
    )

    assert scheduler.reconcile_focus().action is FocusAction.IDLE
    assert scheduler.tasks[task.task_id].status is TaskStatus.QUEUED

    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="postgres",
            resource_class=ExecutionResourceClass.DATABASE,
            capacity=1,
        )
    )
    decision = scheduler.reconcile_focus()

    assert decision.action is FocusAction.START
    assert decision.active_task_id == task.task_id


def test_resource_identity_cannot_silently_change_class():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        ExecutionResource(
            resource_id="shared-id",
            resource_class=ExecutionResourceClass.CPU_GENERAL,
            capacity=1,
        )
    )

    try:
        scheduler.configure_execution_resource(
            ExecutionResource(
                resource_id="shared-id",
                resource_class=ExecutionResourceClass.NETWORK_IO,
                capacity=1,
            )
        )
    except ValueError as exc:
        assert "cannot change class" in str(exc)
    else:
        raise AssertionError("resource identity changed class without rejection")


def test_execution_resources_and_task_requirements_round_trip_through_postgres():
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        scheduler.configure_execution_resource(
            ExecutionResource(
                resource_id="cpu-general",
                resource_class=ExecutionResourceClass.CPU_GENERAL,
                capacity=3,
                system_headroom=1,
                metadata={"host": "local"},
            )
        )
        scheduler.configure_execution_resource(
            ExecutionResource(
                resource_id="local-llm",
                resource_class=ExecutionResourceClass.LLM_INFERENCE,
                capacity=1,
                enabled=False,
                metadata={"backend": "ollama"},
            )
        )
        task = scheduler.submit(
            _task(
                "durable-resource-task",
                allocate_created_seq(conn),
                required_resources=[ExecutionResourceClass.LLM_INFERENCE],
            )
        )
        save_scheduler(conn, scheduler)
    finally:
        conn.close()

    restarted_conn = db.get_connection()
    try:
        restored = load_scheduler(restarted_conn)
        resources = restored.execution_resources()

        assert [resource.resource_id for resource in resources] == [
            "cpu-general",
            "local-llm",
        ]
        assert resources[0].capacity == 3
        assert resources[0].system_headroom == 1
        assert resources[0].admissible_capacity == 2
        assert resources[0].metadata == {"host": "local"}
        assert resources[1].enabled is False
        assert resources[1].metadata == {"backend": "ollama"}
        assert restored.tasks[task.task_id].metadata.required_resource_classes == [
            ExecutionResourceClass.LLM_INFERENCE
        ]

        decision = restored.reconcile_focus()
        assert decision.action is FocusAction.IDLE
        assert restored.tasks[task.task_id].status is TaskStatus.QUEUED
    finally:
        restarted_conn.close()
