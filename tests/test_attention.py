from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from prometheist.attention import (
    AttentionTask,
    FocusAction,
    InterruptionPolicy,
    JITAttentionScheduler,
    PriorityClass,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    TaskStatus,
    derive_priority,
    deterministic_task_id,
)


NAMESPACE = UUID("11111111-1111-1111-1111-111111111111")


def _task(
    key: str,
    seq: int,
    criticality: TaskCriticality,
    *,
    service_class: ServiceClass = ServiceClass.USER_WORK,
    interruption_policy: InterruptionPolicy = InterruptionPolicy.PREEMPTIBLE,
    deadline: datetime | None = None,
    dependencies: list[UUID] | None = None,
) -> AttentionTask:
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=seq,
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=service_class,
            interruption_policy=interruption_policy,
            deadline=deadline,
            dependency_ids=dependencies or [],
        ),
    )


def test_priority_is_derived_only_from_structured_criticality():
    expected = {
        TaskCriticality.EMERGENCY: PriorityClass.P0,
        TaskCriticality.USER_BLOCKING: PriorityClass.P1,
        TaskCriticality.USER_REQUESTED: PriorityClass.P2,
        TaskCriticality.SUPPORTING: PriorityClass.P3,
        TaskCriticality.MAINTENANCE: PriorityClass.P4,
        TaskCriticality.OPPORTUNISTIC: PriorityClass.P5,
    }
    for criticality, priority in expected.items():
        metadata = SchedulingMetadata(
            criticality=criticality,
            service_class=ServiceClass.USER_WORK,
        )
        assert derive_priority(metadata) is priority


def test_deterministic_task_id_is_stable():
    first = deterministic_task_id(NAMESPACE, "compile-release")
    second = deterministic_task_id(NAMESPACE, "compile-release")
    other = deterministic_task_id(NAMESPACE, "deploy-release")
    assert first == second
    assert first != other


def test_submit_rejects_prequeued_task_without_recording_a_noop_transition():
    scheduler = JITAttentionScheduler()
    prequeued = _task("already-queued", 1, TaskCriticality.USER_REQUESTED).model_copy(
        update={"status": TaskStatus.QUEUED}
    )

    with pytest.raises(ValueError, match="must be NEW"):
        scheduler.submit(prequeued)

    assert scheduler.tasks == {}
    assert scheduler.transitions == []


def test_checkpoint_without_active_task_advances_exactly_one_cycle():
    scheduler = JITAttentionScheduler()

    decision = scheduler.checkpoint_active()

    assert decision.action is FocusAction.IDLE
    assert scheduler.cycle == 1


def test_concurrently_assigned_task_can_complete_without_legacy_active_focus():
    scheduler = JITAttentionScheduler()
    submitted = scheduler.submit(
        _task("concurrent-completion", 1, TaskCriticality.USER_REQUESTED)
    )

    completed = scheduler.complete_task(
        submitted.task_id,
        {"result_ref": "worker-result:1"},
    )

    assert scheduler.active_task_id is None
    assert completed.status is TaskStatus.COMPLETED
    assert completed.resumable_state == {"result_ref": "worker-result:1"}
    assert scheduler.complete_task(completed.task_id) == completed


def test_queue_order_is_priority_then_deadline_then_creation_sequence():
    scheduler = JITAttentionScheduler()
    later_deadline = datetime(2026, 8, 25, tzinfo=timezone.utc)
    sooner_deadline = datetime(2026, 8, 24, tzinfo=timezone.utc)

    scheduler.submit(
        _task("p3", 1, TaskCriticality.SUPPORTING, deadline=sooner_deadline)
    )
    scheduler.submit(
        _task("p2-late", 2, TaskCriticality.USER_REQUESTED, deadline=later_deadline)
    )
    scheduler.submit(
        _task("p2-soon", 3, TaskCriticality.USER_REQUESTED, deadline=sooner_deadline)
    )
    scheduler.submit(
        _task("p2-soon-later-seq", 4, TaskCriticality.USER_REQUESTED, deadline=sooner_deadline)
    )

    assert [task.task_key for task in scheduler.queued_tasks()] == [
        "p2-soon",
        "p2-soon-later-seq",
        "p2-late",
        "p3",
    ]


def test_preemptible_task_is_requeued_when_higher_priority_work_arrives():
    scheduler = JITAttentionScheduler()
    low = scheduler.submit(_task("low", 1, TaskCriticality.USER_REQUESTED))

    first = scheduler.reconcile_focus()
    assert first.action is FocusAction.START
    assert first.active_task_id == low.task_id

    high = scheduler.submit(_task("high", 2, TaskCriticality.USER_BLOCKING))
    second = scheduler.reconcile_focus()

    assert second.action is FocusAction.PREEMPT
    assert scheduler.active_task_id == high.task_id
    assert scheduler.tasks[low.task_id].status is TaskStatus.QUEUED
    assert [transition.to_status for transition in scheduler.transitions[-3:]] == [
        TaskStatus.SUSPENDED,
        TaskStatus.QUEUED,
        TaskStatus.RUNNING,
    ]


def test_same_priority_arrival_never_preempts_active_task():
    scheduler = JITAttentionScheduler()
    later_deadline = datetime(2026, 8, 25, tzinfo=timezone.utc)
    sooner_deadline = datetime(2026, 8, 24, tzinfo=timezone.utc)
    active = scheduler.submit(
        _task(
            "active-same-priority",
            1,
            TaskCriticality.USER_REQUESTED,
            deadline=later_deadline,
        )
    )
    scheduler.reconcile_focus()
    queued = scheduler.submit(
        _task(
            "queued-same-priority",
            2,
            TaskCriticality.USER_REQUESTED,
            deadline=sooner_deadline,
        )
    )

    decision = scheduler.reconcile_focus()

    assert decision.action is FocusAction.CONTINUE
    assert scheduler.active_task_id == active.task_id
    assert scheduler.tasks[queued.task_id].status is TaskStatus.QUEUED


def test_checkpoint_only_task_defers_preemption_until_checkpoint():
    scheduler = JITAttentionScheduler()
    low = scheduler.submit(
        _task(
            "checkpointed",
            1,
            TaskCriticality.USER_REQUESTED,
            interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
        )
    )
    scheduler.reconcile_focus()
    high = scheduler.submit(_task("urgent", 2, TaskCriticality.USER_BLOCKING))

    deferred = scheduler.reconcile_focus()
    assert deferred.action is FocusAction.WAIT_FOR_CHECKPOINT
    assert scheduler.active_task_id == low.task_id
    assert scheduler.pending_preemption_task_id == high.task_id

    yielded = scheduler.checkpoint_active()
    assert yielded.action is FocusAction.PREEMPT
    assert scheduler.active_task_id == high.task_id
    assert scheduler.tasks[low.task_id].status is TaskStatus.QUEUED


def test_atomic_task_cannot_be_preempted_even_by_p0():
    scheduler = JITAttentionScheduler()
    atomic = scheduler.submit(
        _task(
            "atomic-write",
            1,
            TaskCriticality.MAINTENANCE,
            interruption_policy=InterruptionPolicy.ATOMIC,
        )
    )
    scheduler.reconcile_focus()
    emergency = scheduler.submit(
        _task(
            "integrity-emergency",
            2,
            TaskCriticality.EMERGENCY,
            service_class=ServiceClass.INTERACTIVE,
        )
    )

    decision = scheduler.reconcile_focus()
    assert decision.action is FocusAction.CONTINUE
    assert scheduler.active_task_id == atomic.task_id
    assert scheduler.tasks[emergency.task_id].status is TaskStatus.QUEUED

    scheduler.complete_active()
    next_decision = scheduler.reconcile_focus()
    assert next_decision.action is FocusAction.START
    assert next_decision.active_task_id == emergency.task_id


def test_service_guarantee_promotes_long_waiting_maintenance_deterministically():
    scheduler = JITAttentionScheduler()
    maintenance = scheduler.submit(
        _task(
            "maintenance",
            1,
            TaskCriticality.MAINTENANCE,
            service_class=ServiceClass.MAINTENANCE,
        )
    )
    scheduler.submit(
        _task(
            "support",
            2,
            TaskCriticality.SUPPORTING,
            service_class=ServiceClass.SUPPORT,
        )
    )

    assert [task.task_key for task in scheduler.queued_tasks()] == ["support", "maintenance"]

    scheduler.advance_cycle(32)

    assert scheduler.queued_tasks()[0].task_id == maintenance.task_id


def test_unmet_dependency_is_not_runnable_until_dependency_completes():
    scheduler = JITAttentionScheduler()
    prerequisite = scheduler.submit(_task("prerequisite", 1, TaskCriticality.SUPPORTING))
    dependent = scheduler.submit(
        _task(
            "dependent",
            2,
            TaskCriticality.EMERGENCY,
            dependencies=[prerequisite.task_id],
        )
    )

    first = scheduler.reconcile_focus()
    assert first.active_task_id == prerequisite.task_id
    assert scheduler.tasks[dependent.task_id].status is TaskStatus.QUEUED

    scheduler.complete_active()
    second = scheduler.reconcile_focus()
    assert second.active_task_id == dependent.task_id


def test_snapshot_round_trip_preserves_focus_queue_and_resumable_state():
    scheduler = JITAttentionScheduler()
    active = scheduler.submit(_task("active", 1, TaskCriticality.USER_REQUESTED))
    queued = scheduler.submit(_task("queued", 2, TaskCriticality.SUPPORTING))
    scheduler.reconcile_focus()
    scheduler.set_resumable_state(active.task_id, {"step": 3, "artifact": "abc"})

    restored = JITAttentionScheduler.from_snapshot(scheduler.snapshot())

    assert restored.cycle == scheduler.cycle
    assert restored.active_task_id == active.task_id
    assert restored.tasks[active.task_id].resumable_state == {"step": 3, "artifact": "abc"}
    assert restored.tasks[queued.task_id].status is TaskStatus.QUEUED
    assert [task.task_id for task in restored.queued_tasks()] == [queued.task_id]


def test_identical_replay_produces_identical_focus_and_transition_ids():
    def run_scenario():
        scheduler = JITAttentionScheduler()
        scheduler.submit(_task("replay-low", 1, TaskCriticality.USER_REQUESTED))
        scheduler.reconcile_focus()
        scheduler.submit(_task("replay-high", 2, TaskCriticality.USER_BLOCKING))
        decision = scheduler.reconcile_focus()
        return (
            scheduler.snapshot().model_dump(mode="json"),
            [transition.model_dump(mode="json") for transition in scheduler.transitions],
            decision.model_dump(mode="json"),
        )

    assert run_scenario() == run_scenario()
