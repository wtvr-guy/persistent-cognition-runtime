"""Fresh-process actors for the integrated v0.7 Increment H acceptance."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid5

from prometheist import db, event_store
from prometheist.attention_observation import HostResourceMetrics, LocalResourceAdmissionController
from prometheist.attention_store import load_scheduler, save_scheduler
from prometheist.models import EventType
from prometheist.worker_protocol import WorkerEffectPolicy
from prometheist.worker_store import (
    checkpoint_worker_claim,
    complete_worker_claim,
    guarded_claim_worker_step,
    register_worker_step,
)


BASE_TIME = datetime(2026, 8, 26, 22, 0, tzinfo=timezone.utc)
EFFECT_SOURCE = "v0.7-h.fake-idempotent-effect"


class FixedProbe:
    def capture(self) -> HostResourceMetrics:
        return HostResourceMetrics(
            platform="test",
            logical_cpu_count=8,
            cpu_utilization_percent=10,
            load_1m=0,
            memory_total_mib=16_384,
            memory_available_mib=12_000,
        )


def _record_effect(
    conn,
    *,
    step_id: UUID,
    idempotency_key: str,
    conversation_id: UUID,
    correlation_id: UUID,
):
    return event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.SYSTEM_EVENT,
        source=EFFECT_SOURCE,
        payload={"step_id": str(step_id), "idempotency_key": idempotency_key},
        event_id=uuid5(step_id, "fake-idempotent-effect"),
    )


def checkpoint_and_wait(step_id: UUID) -> None:
    with db.get_connection() as conn:
        attempt = guarded_claim_worker_step(
            conn,
            step_id=step_id,
            worker_id="checkpoint-worker-before-destruction",
            probe=FixedProbe(),
            clock=lambda: BASE_TIME,
            lease_seconds=30,
        )
        if attempt.envelope is None:
            raise RuntimeError(attempt.observation.reason)
        checkpoint = checkpoint_worker_claim(
            conn,
            claim_id=attempt.envelope.claim.claim_id,
            worker_id=attempt.envelope.claim.worker_id,
            state={"cursor": 17, "artifact": "checkpoint-before-destruction"},
            output_refs=["artifact:checkpoint-17"],
            clock=lambda: BASE_TIME + timedelta(seconds=1),
        )
        print(
            json.dumps(
                {
                    "claim_id": str(attempt.envelope.claim.claim_id),
                    "checkpoint_id": str(checkpoint.checkpoint_id),
                    "checkpoint_revision": checkpoint.revision,
                }
            ),
            flush=True,
        )
        while True:
            time.sleep(60)


def effect_and_wait(
    step_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    worker_id: str,
) -> None:
    with db.get_connection() as conn:
        attempt = guarded_claim_worker_step(
            conn,
            step_id=step_id,
            worker_id=worker_id,
            probe=FixedProbe(),
            clock=lambda: BASE_TIME,
            lease_seconds=5,
        )
        if attempt.envelope is None:
            raise RuntimeError(attempt.observation.reason)
        effect = _record_effect(
            conn,
            step_id=step_id,
            idempotency_key=attempt.envelope.step.idempotency_key,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
        )
        print(
            json.dumps(
                {
                    "claim_id": str(attempt.envelope.claim.claim_id),
                    "effect_event_id": str(effect.event_id),
                    "idempotency_key": attempt.envelope.step.idempotency_key,
                }
            ),
            flush=True,
        )
        while True:
            time.sleep(60)


def _recover_effect_step(
    conn,
    *,
    step_id: UUID,
    task_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    worker_id: str,
) -> int:
    attempt = guarded_claim_worker_step(
        conn,
        step_id=step_id,
        worker_id=worker_id,
        probe=FixedProbe(),
        clock=lambda: BASE_TIME + timedelta(seconds=10),
        lease_seconds=30,
    )
    if attempt.envelope is None:
        raise RuntimeError(attempt.observation.reason)
    _record_effect(
        conn,
        step_id=step_id,
        idempotency_key=attempt.envelope.step.idempotency_key,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
    )
    complete_worker_claim(
        conn,
        claim_id=attempt.envelope.claim.claim_id,
        worker_id=worker_id,
        output={"task_id": str(task_id), "recovered": True},
        output_refs=[f"event:{uuid5(step_id, 'fake-idempotent-effect')}"],
        clock=lambda: BASE_TIME + timedelta(seconds=11),
    )
    return attempt.envelope.claim.attempt


def resume_workload(payload: dict[str, str]) -> None:
    task_ids = {
        name: UUID(payload[f"{name}_task_id"])
        for name in ("checkpoint", "first", "second", "urgent")
    }
    old_step_ids = {
        name: UUID(payload[f"{name}_step_id"])
        for name in ("first", "second")
    }
    conversation_id = UUID(payload["conversation_id"])
    correlation_id = UUID(payload["correlation_id"])

    with db.get_connection() as conn:
        scheduler = load_scheduler(conn)
        assigned_before = {
            assignment.task_id for assignment in scheduler.worker_visible_assignments()
        }
        expected_before = {
            task_ids["first"],
            task_ids["second"],
            task_ids["urgent"],
        }
        if assigned_before != expected_before:
            raise RuntimeError("restart reconstructed the wrong assignment set")
        if scheduler.tasks[task_ids["checkpoint"]].resumable_state != {
            "cursor": 17,
            "artifact": "checkpoint-before-destruction",
        }:
            raise RuntimeError("task checkpoint was not reconstructed")

        attempts = {
            name: _recover_effect_step(
                conn,
                step_id=old_step_ids[name],
                task_id=task_ids[name],
                conversation_id=conversation_id,
                correlation_id=correlation_id,
                worker_id=f"fresh-{name}-worker",
            )
            for name in ("first", "second")
        }

        urgent_assignment = next(
            assignment
            for assignment in scheduler.worker_visible_assignments()
            if assignment.task_id == task_ids["urgent"]
        )
        urgent_step = register_worker_step(
            conn,
            assignment_id=urgent_assignment.assignment_id,
            step_key="work",
            capability="acceptance.fake-effect",
            effect_policy=WorkerEffectPolicy.NO_EXTERNAL_EFFECT,
        )
        urgent_claim = guarded_claim_worker_step(
            conn,
            step_id=urgent_step.step_id,
            worker_id="fresh-urgent-worker",
            probe=FixedProbe(),
            clock=lambda: BASE_TIME + timedelta(seconds=10),
        )
        if urgent_claim.envelope is None:
            raise RuntimeError(urgent_claim.observation.reason)
        complete_worker_claim(
            conn,
            claim_id=urgent_claim.envelope.claim.claim_id,
            worker_id="fresh-urgent-worker",
            output={"urgent": "completed-after-restart"},
            clock=lambda: BASE_TIME + timedelta(seconds=11),
        )

        for name in ("first", "second", "urgent"):
            scheduler.complete_task(task_ids[name], {"completed_after_restart": True})
        LocalResourceAdmissionController(
            scheduler,
            probe=FixedProbe(),
            clock=lambda: BASE_TIME + timedelta(seconds=12),
            host_id="increment-h-test-host",
        ).plan_scheduling_epoch()
        save_scheduler(conn, scheduler)

        checkpoint_assignment = next(
            assignment
            for assignment in scheduler.worker_visible_assignments()
            if assignment.task_id == task_ids["checkpoint"]
        )
        resumed_step = register_worker_step(
            conn,
            assignment_id=checkpoint_assignment.assignment_id,
            step_key="work",
            capability="acceptance.fake-effect",
            effect_policy=WorkerEffectPolicy.NO_EXTERNAL_EFFECT,
        )
        resumed_claim = guarded_claim_worker_step(
            conn,
            step_id=resumed_step.step_id,
            worker_id="fresh-checkpoint-worker",
            probe=FixedProbe(),
            clock=lambda: BASE_TIME + timedelta(seconds=13),
        )
        if resumed_claim.envelope is None:
            raise RuntimeError(resumed_claim.observation.reason)
        complete_worker_claim(
            conn,
            claim_id=resumed_claim.envelope.claim.claim_id,
            worker_id="fresh-checkpoint-worker",
            output={
                "resumed_cursor": scheduler.tasks[
                    task_ids["checkpoint"]
                ].resumable_state["cursor"]
            },
            clock=lambda: BASE_TIME + timedelta(seconds=14),
        )
        scheduler.complete_task(
            task_ids["checkpoint"],
            {"cursor": 18, "completed_after_restart": True},
        )
        LocalResourceAdmissionController(
            scheduler,
            probe=FixedProbe(),
            clock=lambda: BASE_TIME + timedelta(seconds=15),
            host_id="increment-h-test-host",
        ).plan_scheduling_epoch()
        save_scheduler(conn, scheduler)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM events WHERE source = %s",
                (EFFECT_SOURCE,),
            )
            effect_count = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT event_type
                FROM attention_preemption_events
                ORDER BY CASE event_type
                    WHEN 'REQUESTED' THEN 1
                    WHEN 'CHECKPOINT_ACKNOWLEDGED' THEN 2
                    WHEN 'EXECUTED' THEN 3
                    ELSE 4
                END
                """
            )
            preemption_events = [row[0] for row in cur.fetchall()]
        final = load_scheduler(conn)
        print(
            json.dumps(
                {
                    "recovery_attempts": attempts,
                    "effect_count": effect_count,
                    "preemption_events": preemption_events,
                    "final_assignments": len(final.worker_visible_assignments()),
                    "final_statuses": {
                        name: final.tasks[task_id].status.value
                        for name, task_id in task_ids.items()
                    },
                    "resumed_checkpoint_step_id": str(resumed_step.step_id),
                    "old_checkpoint_step_id": payload["checkpoint_step_id"],
                },
                sort_keys=True,
            )
        )


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "checkpoint-wait" and len(sys.argv) == 3:
        checkpoint_and_wait(UUID(sys.argv[2]))
    elif command == "effect-wait" and len(sys.argv) == 6:
        effect_and_wait(
            UUID(sys.argv[2]),
            UUID(sys.argv[3]),
            UUID(sys.argv[4]),
            sys.argv[5],
        )
    elif command == "resume" and len(sys.argv) == 3:
        resume_workload(json.loads(sys.argv[2]))
    else:
        raise SystemExit(
            "usage: _v07_restart_process "
            "<checkpoint-wait STEP|effect-wait STEP CONVERSATION CORRELATION WORKER|resume JSON>"
        )


if __name__ == "__main__":
    main()
