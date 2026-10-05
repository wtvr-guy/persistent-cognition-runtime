from __future__ import annotations

from uuid import UUID

from prometheist import db
from prometheist.attention import (
    AttentionTask,
    FocusAction,
    InterruptionPolicy,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    TaskStatus,
    deterministic_task_id,
)
from prometheist.attention_store import (
    allocate_created_seq,
    load_scheduler,
    load_transition_count,
    save_scheduler,
)


NAMESPACE = UUID("22222222-2222-2222-2222-222222222222")


def _new_task(
    conn,
    key: str,
    criticality: TaskCriticality,
    *,
    interruption_policy=InterruptionPolicy.PREEMPTIBLE,
):
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=allocate_created_seq(conn),
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=ServiceClass.USER_WORK,
            interruption_policy=interruption_policy,
        ),
    )


def test_attention_state_survives_fresh_connection_restart():
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        active = scheduler.submit(_new_task(conn, "active", TaskCriticality.USER_REQUESTED))
        queued = scheduler.submit(_new_task(conn, "queued", TaskCriticality.SUPPORTING))
        scheduler.reconcile_focus()
        scheduler.set_resumable_state(active.task_id, {"step": 2, "output": "intermediate"})
        save_scheduler(conn, scheduler)
    finally:
        conn.close()

    restarted_conn = db.get_connection()
    try:
        restored = load_scheduler(restarted_conn)
        assert restored.active_task_id == active.task_id
        assert restored.tasks[active.task_id].status is TaskStatus.RUNNING
        assert restored.tasks[active.task_id].resumable_state == {
            "step": 2,
            "output": "intermediate",
        }
        assert restored.tasks[queued.task_id].status is TaskStatus.QUEUED

        restored.complete_active({"step": 3, "output": "done"})
        decision = restored.reconcile_focus()
        assert decision.action is FocusAction.START
        assert decision.active_task_id == queued.task_id
        save_scheduler(restarted_conn, restored)
    finally:
        restarted_conn.close()

    final_conn = db.get_connection()
    try:
        final = load_scheduler(final_conn)
        assert final.tasks[active.task_id].status is TaskStatus.COMPLETED
        assert final.active_task_id == queued.task_id
    finally:
        final_conn.close()


def test_checkpoint_preemption_survives_restart():
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        checkpointed = scheduler.submit(
            _new_task(
                conn,
                "checkpointed",
                TaskCriticality.USER_REQUESTED,
                interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
            )
        )
        scheduler.reconcile_focus()
        urgent = scheduler.submit(_new_task(conn, "urgent", TaskCriticality.USER_BLOCKING))
        deferred = scheduler.reconcile_focus()
        assert deferred.action is FocusAction.WAIT_FOR_CHECKPOINT
        save_scheduler(conn, scheduler)
    finally:
        conn.close()

    restarted_conn = db.get_connection()
    try:
        restored = load_scheduler(restarted_conn)
        assert restored.active_task_id == checkpointed.task_id
        assert restored.pending_preemption_task_id == urgent.task_id

        yielded = restored.checkpoint_active()
        assert yielded.action is FocusAction.PREEMPT
        assert restored.active_task_id == urgent.task_id
        assert restored.tasks[checkpointed.task_id].status is TaskStatus.QUEUED
    finally:
        restarted_conn.close()


def test_transition_persistence_is_idempotent():
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        task = scheduler.submit(_new_task(conn, "idempotent", TaskCriticality.USER_REQUESTED))
        scheduler.reconcile_focus()
        expected_count = len(scheduler.transitions)

        save_scheduler(conn, scheduler)
        assert load_transition_count(conn, task.task_id) == expected_count
        assert scheduler._pending_transitions_for_persistence() == []

        save_scheduler(conn, scheduler)
        assert load_transition_count(conn, task.task_id) == expected_count
        assert scheduler._pending_transitions_for_persistence() == []
    finally:
        conn.close()


def test_scheduler_namespaces_reconstruct_only_their_own_tasks_and_transitions():
    conn = db.get_connection()
    try:
        first = load_scheduler(conn, scheduler_key="first")
        first_task = first.submit(
            _new_task(conn, "first-task", TaskCriticality.USER_REQUESTED)
        )
        save_scheduler(conn, first, scheduler_key="first")

        second = load_scheduler(conn, scheduler_key="second")
        assert second.tasks == {}
        second_task = second.submit(
            _new_task(conn, "second-task", TaskCriticality.SUPPORTING)
        )
        save_scheduler(conn, second, scheduler_key="second")

        restored_first = load_scheduler(conn, scheduler_key="first")
        restored_second = load_scheduler(conn, scheduler_key="second")

        assert set(restored_first.tasks) == {first_task.task_id}
        assert set(restored_second.tasks) == {second_task.task_id}
        assert load_transition_count(conn, scheduler_key="first") == 1
        assert load_transition_count(conn, scheduler_key="second") == 1
    finally:
        conn.close()
