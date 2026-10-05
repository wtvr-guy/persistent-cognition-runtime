"""Subprocess helper for deterministic JIT Attention crash/restart tests."""
from __future__ import annotations

import json
import sys
import time
from uuid import UUID

from prometheist import db
from prometheist.attention import (
    AttentionTask,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    deterministic_task_id,
)
from prometheist.attention_store import allocate_created_seq, load_scheduler, save_scheduler


NAMESPACE = UUID("33333333-3333-3333-3333-333333333333")


def _task(conn, key: str, criticality: TaskCriticality) -> AttentionTask:
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=allocate_created_seq(conn),
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=ServiceClass.USER_WORK,
        ),
    )


def checkpoint_and_wait(run_key: str) -> None:
    """Persist an unfinished task, signal the parent, then remain killable."""

    active_key = f"{run_key}:active"
    queued_key = f"{run_key}:queued"
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        active = scheduler.submit(_task(conn, active_key, TaskCriticality.USER_REQUESTED))
        queued = scheduler.submit(_task(conn, queued_key, TaskCriticality.SUPPORTING))
        scheduler.reconcile_focus()
        if scheduler.active_task_id != active.task_id:
            raise RuntimeError("unexpected active task before checkpoint")
        scheduler.set_resumable_state(
            active.task_id,
            {"step": 1, "artifact": "persisted-before-kill"},
        )
        save_scheduler(conn, scheduler)
        print(
            json.dumps(
                {
                    "active_task_id": str(active.task_id),
                    "queued_task_id": str(queued.task_id),
                    "cycle": scheduler.cycle,
                }
            ),
            flush=True,
        )

        # The parent test deliberately kills this process after reading the
        # checkpoint line.  There is no graceful shutdown path by design.
        while True:
            time.sleep(60)
    finally:
        conn.close()


def resume(run_key: str) -> None:
    """Load durable state in a fresh process and continue from the checkpoint."""

    active_id = deterministic_task_id(NAMESPACE, f"{run_key}:active")
    queued_id = deterministic_task_id(NAMESPACE, f"{run_key}:queued")
    conn = db.get_connection()
    try:
        scheduler = load_scheduler(conn)
        if scheduler.active_task_id != active_id:
            raise RuntimeError("restart did not recover the previously active task")
        recovered_state = dict(scheduler.tasks[active_id].resumable_state)

        scheduler.complete_active(
            {"step": 2, "artifact": "completed-after-restart"}
        )
        decision = scheduler.reconcile_focus()
        if decision.active_task_id != queued_id:
            raise RuntimeError("restart did not continue with the queued task")
        save_scheduler(conn, scheduler)

        print(
            json.dumps(
                {
                    "recovered_active_task_id": str(active_id),
                    "recovered_state": recovered_state,
                    "next_active_task_id": str(scheduler.active_task_id),
                    "completed_status": scheduler.tasks[active_id].status.value,
                    "cycle": scheduler.cycle,
                }
            ),
            flush=True,
        )
    finally:
        conn.close()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m tests._attention_process <checkpoint-wait|resume> <run-key>")
    command, run_key = sys.argv[1], sys.argv[2]
    if command == "checkpoint-wait":
        checkpoint_and_wait(run_key)
    elif command == "resume":
        resume(run_key)
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
