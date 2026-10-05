from __future__ import annotations

from uuid import UUID

import pytest

from prometheist import attention_store, db
from prometheist.attention import (
    AttentionTask,
    InterruptionPolicy,
    JITAttentionScheduler,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    deterministic_task_id,
)
from prometheist.attention_preemption import PreemptionEventType
from prometheist.attention_resources import (
    ExecutionResource,
    ExecutionResourceClass,
    ResourceRequirement,
)
from prometheist.attention_store import (
    load_preemption_event_count,
    load_scheduler,
    save_scheduler,
)


NAMESPACE = UUID("77777777-7777-7777-7777-777777777777")


def _task(
    key: str,
    seq: int,
    criticality: TaskCriticality,
    *,
    requirements: list[tuple[ExecutionResourceClass, int]],
    interruption_policy: InterruptionPolicy = InterruptionPolicy.PREEMPTIBLE,
) -> AttentionTask:
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=seq,
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=ServiceClass.USER_WORK,
            interruption_policy=interruption_policy,
            resource_requirements=[
                ResourceRequirement(resource_class=resource_class, units=units)
                for resource_class, units in requirements
            ],
        ),
    )


def _resource(
    resource_id: str,
    resource_class: ExecutionResourceClass,
    capacity: int,
) -> ExecutionResource:
    return ExecutionResource(
        resource_id=resource_id,
        resource_class=resource_class,
        capacity=capacity,
    )


def _commit(scheduler: JITAttentionScheduler) -> None:
    plan = scheduler.pending_scheduling_epoch()
    assert plan is not None
    scheduler._commit_pending_epoch(plan.epoch.epoch_id)


def test_safe_concurrent_capacity_never_causes_preemption():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
    )
    existing = scheduler.submit(
        _task(
            "existing",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    urgent = scheduler.submit(
        _task(
            "urgent",
            2,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )

    plan = scheduler.plan_scheduling_epoch()

    assert plan.admitted_task_ids == [existing.task_id, urgent.task_id]
    assert plan.preemption_events == []


def test_only_work_holding_a_deficient_resource_is_eligible_to_yield():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
    )
    scheduler.configure_execution_resource(
        _resource("db", ExecutionResourceClass.DATABASE, 1)
    )
    cpu_task = scheduler.submit(
        _task(
            "cpu-low",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    db_task = scheduler.submit(
        _task(
            "db-low",
            2,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.DATABASE, 1)],
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    urgent = scheduler.submit(
        _task(
            "cpu-urgent",
            3,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )

    plan = scheduler.plan_scheduling_epoch()

    assert plan.admitted_task_ids == [db_task.task_id, urgent.task_id]
    assert plan.unadmitted_task_ids == [cpu_task.task_id]
    assert plan.preemption_events[0].victim_task_ids == [cpu_task.task_id]


def test_atomic_and_same_priority_assignments_are_not_preempted():
    for policy, existing_criticality in (
        (InterruptionPolicy.ATOMIC, TaskCriticality.MAINTENANCE),
        (InterruptionPolicy.PREEMPTIBLE, TaskCriticality.USER_REQUESTED),
    ):
        scheduler = JITAttentionScheduler()
        scheduler.configure_execution_resource(
            _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
        )
        existing = scheduler.submit(
            _task(
                f"existing-{policy.value}",
                1,
                existing_criticality,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
                interruption_policy=policy,
            )
        )
        scheduler.plan_scheduling_epoch()
        _commit(scheduler)
        candidate_criticality = (
            TaskCriticality.EMERGENCY
            if policy is InterruptionPolicy.ATOMIC
            else TaskCriticality.USER_REQUESTED
        )
        candidate = scheduler.submit(
            _task(
                f"candidate-{policy.value}",
                2,
                candidate_criticality,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )

        plan = scheduler.plan_scheduling_epoch()

        assert plan.admitted_task_ids == [existing.task_id]
        assert plan.unadmitted_task_ids == [candidate.task_id]
        assert plan.preemption_events == []


def test_victim_selection_minimizes_count_before_tie_breaking():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 4)
    )
    large = scheduler.submit(
        _task(
            "large",
            1,
            TaskCriticality.SUPPORTING,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )
    small_one = scheduler.submit(
        _task(
            "small-one",
            2,
            TaskCriticality.OPPORTUNISTIC,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    small_two = scheduler.submit(
        _task(
            "small-two",
            3,
            TaskCriticality.OPPORTUNISTIC,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    urgent = scheduler.submit(
        _task(
            "urgent-two",
            4,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )

    plan = scheduler.plan_scheduling_epoch()

    assert plan.preemption_events[0].victim_task_ids == [large.task_id]
    assert set(plan.admitted_task_ids) == {
        small_one.task_id,
        small_two.task_id,
        urgent.task_id,
    }


def test_equal_victims_release_the_newest_assignment_deterministically():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
    )
    established = scheduler.submit(
        _task(
            "established",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    newer = scheduler.submit(
        _task(
            "newer",
            2,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    urgent = scheduler.submit(
        _task(
            "urgent",
            3,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )

    plan = scheduler.plan_scheduling_epoch()

    assert plan.preemption_events[0].victim_task_ids == [newer.task_id]
    assert plan.admitted_task_ids == [established.task_id, urgent.task_id]


def test_victim_selection_is_independent_of_task_submission_order():
    tasks = [
        _task(
            "first-low",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        ),
        _task(
            "second-low",
            2,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        ),
    ]

    def run(*, reverse: bool) -> dict[str, object]:
        scheduler = JITAttentionScheduler()
        scheduler.configure_execution_resource(
            _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
        )
        for task in reversed(tasks) if reverse else tasks:
            scheduler.submit(task)
        scheduler.plan_scheduling_epoch()
        _commit(scheduler)
        scheduler.submit(
            _task(
                "deterministic-urgent",
                3,
                TaskCriticality.EMERGENCY,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        return scheduler.plan_scheduling_epoch().model_dump(mode="json")

    assert run(reverse=False) == run(reverse=True)


def test_checkpoint_only_intent_retains_capacity_until_acknowledged_epoch():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
    )
    checkpointed = scheduler.submit(
        _task(
            "checkpointed",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    original_assignment_id = scheduler.worker_visible_assignments()[0].assignment_id
    urgent = scheduler.submit(
        _task(
            "urgent",
            2,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )

    waiting = scheduler.plan_scheduling_epoch()

    assert waiting.admitted_task_ids == [checkpointed.task_id]
    assert waiting.unadmitted_task_ids == [urgent.task_id]
    assert waiting.pending_preemptions[0].target_task_id == urgent.task_id
    assert waiting.preemption_events[0].event_type is PreemptionEventType.REQUESTED
    _commit(scheduler)

    acknowledged = scheduler.checkpoint_pending_preemption(
        checkpointed.task_id,
        resumable_state={"step": 4},
    )
    assert acknowledged.ready_to_execute
    assert scheduler.tasks[checkpointed.task_id].resumable_state == {"step": 4}
    assert scheduler.worker_visible_assignments()[0].assignment_id == (
        original_assignment_id
    )

    replacement = scheduler.plan_scheduling_epoch()

    assert replacement.admitted_task_ids == [urgent.task_id]
    assert replacement.unadmitted_task_ids == [checkpointed.task_id]
    assert replacement.pending_preemptions == []
    assert replacement.preemption_events[0].event_type is PreemptionEventType.EXECUTED


def test_mixed_checkpoint_selection_releases_nothing_partially():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
    )
    immediate = scheduler.submit(
        _task(
            "immediate",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    checkpointed = scheduler.submit(
        _task(
            "checkpointed",
            2,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    urgent = scheduler.submit(
        _task(
            "urgent",
            3,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )

    waiting = scheduler.plan_scheduling_epoch()

    assert waiting.admitted_task_ids == [immediate.task_id, checkpointed.task_id]
    assert waiting.unadmitted_task_ids == [urgent.task_id]
    assert set(waiting.pending_preemptions[0].victim_task_ids) == {
        immediate.task_id,
        checkpointed.task_id,
    }
    _commit(scheduler)
    scheduler.checkpoint_pending_preemption(checkpointed.task_id)

    replacement = scheduler.plan_scheduling_epoch()

    assert replacement.admitted_task_ids == [urgent.task_id]
    assert set(replacement.unadmitted_task_ids) == {
        immediate.task_id,
        checkpointed.task_id,
    }


def test_capacity_growth_cancels_checkpoint_intent_without_disturbing_victim():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
    )
    checkpointed = scheduler.submit(
        _task(
            "checkpointed",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    urgent = scheduler.submit(
        _task(
            "urgent",
            2,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)

    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
    )
    plan = scheduler.plan_scheduling_epoch()

    assert plan.admitted_task_ids == [checkpointed.task_id, urgent.task_id]
    assert plan.pending_preemptions == []
    assert plan.preemption_events[0].event_type is PreemptionEventType.CANCELLED


def test_pending_target_holds_its_free_capacity_against_lower_ranked_work():
    scheduler = JITAttentionScheduler()
    scheduler.configure_execution_resource(
        _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 3)
    )
    checkpointed = scheduler.submit(
        _task(
            "checkpointed-hold",
            1,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
        )
    )
    atomic = scheduler.submit(
        _task(
            "atomic-neighbor",
            2,
            TaskCriticality.MAINTENANCE,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            interruption_policy=InterruptionPolicy.ATOMIC,
        )
    )
    scheduler.plan_scheduling_epoch()
    _commit(scheduler)
    urgent = scheduler.submit(
        _task(
            "urgent-hold",
            3,
            TaskCriticality.EMERGENCY,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
        )
    )
    lower = scheduler.submit(
        _task(
            "lower-contender",
            4,
            TaskCriticality.OPPORTUNISTIC,
            requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
        )
    )

    waiting = scheduler.plan_scheduling_epoch()

    assert waiting.admitted_task_ids == [checkpointed.task_id, atomic.task_id]
    assert waiting.unadmitted_task_ids == [urgent.task_id, lower.task_id]
    _commit(scheduler)
    scheduler.checkpoint_pending_preemption(checkpointed.task_id)

    replacement = scheduler.plan_scheduling_epoch()

    assert replacement.admitted_task_ids == [atomic.task_id, urgent.task_id]
    assert lower.task_id in replacement.unadmitted_task_ids


def test_postgres_checkpoint_intent_and_causal_events_survive_process_restart():
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        scheduler.configure_execution_resource(
            _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
        )
        checkpointed = scheduler.submit(
            _task(
                "durable-checkpoint",
                1,
                TaskCriticality.MAINTENANCE,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
                interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
            )
        )
        scheduler.plan_scheduling_epoch()
        save_scheduler(conn, scheduler)
        urgent = scheduler.submit(
            _task(
                "durable-urgent",
                2,
                TaskCriticality.EMERGENCY,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        scheduler.plan_scheduling_epoch()
        save_scheduler(conn, scheduler)
        preemption_id = scheduler.pending_preemptions()[0].preemption_id
        assert load_preemption_event_count(conn, preemption_id) == 1
    finally:
        conn.close()

    restarted_conn = db.get_connection()
    try:
        restored = load_scheduler(restarted_conn)
        assert restored.pending_preemptions()[0].target_task_id == urgent.task_id
        assert restored.worker_visible_assignments()[0].task_id == checkpointed.task_id
        restored.checkpoint_pending_preemption(
            checkpointed.task_id,
            resumable_state={"cursor": "safe-boundary"},
        )
        save_scheduler(restarted_conn, restored)
        save_scheduler(restarted_conn, restored)
        assert load_preemption_event_count(restarted_conn, preemption_id) == 2
    finally:
        restarted_conn.close()

    final_conn = db.get_connection()
    try:
        final = load_scheduler(final_conn)
        assert final.tasks[checkpointed.task_id].resumable_state == {
            "cursor": "safe-boundary"
        }
        final.plan_scheduling_epoch()
        save_scheduler(final_conn, final)
        assert final.pending_preemptions() == []
        assert final.worker_visible_assignments()[0].task_id == urgent.task_id
        assert load_preemption_event_count(final_conn, preemption_id) == 3
    finally:
        final_conn.close()


def test_postgres_concurrent_checkpoint_writers_cannot_erase_each_others_progress():
    setup_conn = db.get_connection()
    try:
        scheduler = load_scheduler(setup_conn)
        scheduler.configure_execution_resource(
            _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 2)
        )
        first_victim = scheduler.submit(
            _task(
                "checkpoint-one",
                1,
                TaskCriticality.MAINTENANCE,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
                interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
            )
        )
        second_victim = scheduler.submit(
            _task(
                "checkpoint-two",
                2,
                TaskCriticality.MAINTENANCE,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
                interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
            )
        )
        scheduler.plan_scheduling_epoch()
        save_scheduler(setup_conn, scheduler)
        scheduler.submit(
            _task(
                "needs-both",
                3,
                TaskCriticality.EMERGENCY,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 2)],
            )
        )
        scheduler.plan_scheduling_epoch()
        save_scheduler(setup_conn, scheduler)
    finally:
        setup_conn.close()

    first_conn = db.get_connection()
    stale_conn = db.get_connection()
    try:
        first = load_scheduler(first_conn)
        stale = load_scheduler(stale_conn)
        first.checkpoint_pending_preemption(first_victim.task_id)
        save_scheduler(first_conn, first)

        stale.checkpoint_pending_preemption(second_victim.task_id)
        with pytest.raises(RuntimeError, match="Conflicting preemption state"):
            save_scheduler(stale_conn, stale)
    finally:
        first_conn.close()
        stale_conn.close()

    verify_conn = db.get_connection()
    try:
        verified = load_scheduler(verify_conn).pending_preemptions()[0]
        assert verified.checkpointed_victim_task_ids == [first_victim.task_id]
    finally:
        verify_conn.close()


def test_failed_postgres_preemption_epoch_publishes_neither_event_nor_victim_release(
    monkeypatch,
):
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        scheduler.configure_execution_resource(
            _resource("cpu", ExecutionResourceClass.CPU_GENERAL, 1)
        )
        existing = scheduler.submit(
            _task(
                "rollback-existing",
                1,
                TaskCriticality.MAINTENANCE,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        scheduler.plan_scheduling_epoch()
        save_scheduler(conn, scheduler)
        scheduler.submit(
            _task(
                "rollback-urgent",
                2,
                TaskCriticality.EMERGENCY,
                requirements=[(ExecutionResourceClass.CPU_GENERAL, 1)],
            )
        )
        scheduler.plan_scheduling_epoch()

        def fail_before_publication(*args, **kwargs):
            raise RuntimeError("injected preemption publication failure")

        monkeypatch.setattr(
            attention_store,
            "_replace_resource_reservations",
            fail_before_publication,
        )
        with pytest.raises(RuntimeError, match="injected preemption"):
            save_scheduler(conn, scheduler)

        assert scheduler.worker_visible_assignments()[0].task_id == existing.task_id
        assert scheduler.pending_scheduling_epoch() is not None
        assert load_preemption_event_count(conn) == 0
    finally:
        conn.close()

    verify_conn = db.get_connection()
    try:
        visible = load_scheduler(verify_conn).worker_visible_assignments()
        assert [assignment.task_id for assignment in visible] == [existing.task_id]
    finally:
        verify_conn.close()
