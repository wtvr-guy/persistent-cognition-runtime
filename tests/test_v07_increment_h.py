from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import timedelta
from uuid import UUID

from prometheist import db, event_store
from prometheist.attention import (
    AttentionTask,
    InterruptionPolicy,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    deterministic_task_id,
)
from prometheist.attention_observation import HostResourceMetrics, LocalResourceAdmissionController
from prometheist.attention_resources import ProcessResourceEstimate, ResourceEstimateSource
from prometheist.attention_store import allocate_created_seq, load_scheduler, save_scheduler
from prometheist.worker_protocol import WorkerEffectPolicy
from prometheist.worker_store import register_worker_step
from tests._v07_restart_process import BASE_TIME


NAMESPACE = UUID("cf94642e-b134-4323-81b7-9c1e81ee65da")


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


def _task(
    conn,
    run_key: str,
    name: str,
    *,
    criticality: TaskCriticality,
    service_class: ServiceClass,
    cpu_units: int,
    memory_mib: int,
    llm_slots: int = 0,
    interruption_policy: InterruptionPolicy = InterruptionPolicy.PREEMPTIBLE,
) -> AttentionTask:
    key = f"increment-h:{run_key}:{name}"
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=allocate_created_seq(conn),
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=service_class,
            interruption_policy=interruption_policy,
            required_capabilities=["acceptance.fake-effect"],
            process_resource_estimate=ProcessResourceEstimate(
                cpu_units=cpu_units,
                memory_mib=memory_mib,
                llm_slots=llm_slots,
                source=ResourceEstimateSource.DECLARED,
                basis="Increment H deterministic fake worker",
            ),
        ),
    )


def _read_ready(process: subprocess.Popen[str]) -> dict:
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if not line:
        error = process.stderr.read() if process.stderr is not None else ""
        raise AssertionError(f"worker process produced no readiness record: {error}")
    return json.loads(line)


def test_forced_concurrent_restart_recovers_checkpoint_and_exactly_once_effects():
    run_key = uuid.uuid4().hex
    conversation_id = uuid.uuid4()
    correlation_id = uuid.uuid4()
    workers: list[subprocess.Popen[str]] = []

    with db.get_connection() as conn:
        event_store.start_conversation(conn, conversation_id)
        scheduler = load_scheduler(conn)
        checkpoint_task = scheduler.submit(
            _task(
                conn,
                run_key,
                "checkpoint",
                criticality=TaskCriticality.MAINTENANCE,
                service_class=ServiceClass.BACKGROUND,
                cpu_units=1,
                memory_mib=4096,
                llm_slots=1,
                interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
            )
        )
        first_task = scheduler.submit(
            _task(
                conn,
                run_key,
                "first",
                criticality=TaskCriticality.SUPPORTING,
                service_class=ServiceClass.SUPPORT,
                cpu_units=1,
                memory_mib=512,
            )
        )
        second_task = scheduler.submit(
            _task(
                conn,
                run_key,
                "second",
                criticality=TaskCriticality.USER_REQUESTED,
                service_class=ServiceClass.USER_WORK,
                cpu_units=2,
                memory_mib=1024,
            )
        )
        LocalResourceAdmissionController(
            scheduler,
            probe=FixedProbe(),
            clock=lambda: BASE_TIME,
            host_id="increment-h-test-host",
        ).plan_scheduling_epoch()
        save_scheduler(conn, scheduler)
        assert {assignment.task_id for assignment in scheduler.worker_visible_assignments()} == {
            checkpoint_task.task_id,
            first_task.task_id,
            second_task.task_id,
        }

        steps = {}
        for name, task, effect_policy in (
            ("checkpoint", checkpoint_task, WorkerEffectPolicy.NO_EXTERNAL_EFFECT),
            ("first", first_task, WorkerEffectPolicy.IDEMPOTENT_WITH_KEY),
            ("second", second_task, WorkerEffectPolicy.IDEMPOTENT_WITH_KEY),
        ):
            assignment = next(
                value
                for value in scheduler.worker_visible_assignments()
                if value.task_id == task.task_id
            )
            steps[name] = register_worker_step(
                conn,
                assignment_id=assignment.assignment_id,
                step_key="work",
                capability="acceptance.fake-effect",
                effect_policy=effect_policy,
            )

        checkpoint_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests._v07_restart_process",
                "checkpoint-wait",
                str(steps["checkpoint"].step_id),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        workers.append(checkpoint_process)
        checkpoint_record = _read_ready(checkpoint_process)
        assert checkpoint_record["checkpoint_revision"] == 1

        for name in ("first", "second"):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests._v07_restart_process",
                    "effect-wait",
                    str(steps[name].step_id),
                    str(conversation_id),
                    str(correlation_id),
                    f"destroyed-{name}-worker",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            workers.append(process)
            effect_record = _read_ready(process)
            assert effect_record["idempotency_key"] == steps[name].idempotency_key

        urgent_task = scheduler.submit(
            _task(
                conn,
                run_key,
                "urgent",
                criticality=TaskCriticality.EMERGENCY,
                service_class=ServiceClass.INTERACTIVE,
                cpu_units=1,
                memory_mib=4096,
                llm_slots=1,
            )
        )
        waiting = LocalResourceAdmissionController(
            scheduler,
            probe=FixedProbe(),
            clock=lambda: BASE_TIME + timedelta(seconds=2),
            host_id="increment-h-test-host",
        ).plan_scheduling_epoch()
        assert waiting.admitted_task_ids == [
            second_task.task_id,
            first_task.task_id,
            checkpoint_task.task_id,
        ]
        assert waiting.pending_preemptions[0].victim_task_ids == [
            checkpoint_task.task_id
        ]
        assert waiting.pending_preemptions[0].target_task_id == urgent_task.task_id
        save_scheduler(conn, scheduler)

        scheduler.checkpoint_pending_preemption(
            checkpoint_task.task_id,
            resumable_state={
                "cursor": 17,
                "artifact": "checkpoint-before-destruction",
            },
        )
        replacement = LocalResourceAdmissionController(
            scheduler,
            probe=FixedProbe(),
            clock=lambda: BASE_TIME + timedelta(seconds=3),
            host_id="increment-h-test-host",
        ).plan_scheduling_epoch()
        assert set(replacement.admitted_task_ids) == {
            first_task.task_id,
            second_task.task_id,
            urgent_task.task_id,
        }
        assert replacement.unadmitted_task_ids == [checkpoint_task.task_id]
        save_scheduler(conn, scheduler)

        payload = {
            "conversation_id": str(conversation_id),
            "correlation_id": str(correlation_id),
            "checkpoint_task_id": str(checkpoint_task.task_id),
            "first_task_id": str(first_task.task_id),
            "second_task_id": str(second_task.task_id),
            "urgent_task_id": str(urgent_task.task_id),
            "checkpoint_step_id": str(steps["checkpoint"].step_id),
            "first_step_id": str(steps["first"].step_id),
            "second_step_id": str(steps["second"].step_id),
        }

    try:
        for process in workers:
            process.kill()
        for process in workers:
            process.wait(timeout=10)
            assert process.returncode != 0

        resumed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests._v07_restart_process",
                "resume",
                json.dumps(payload, sort_keys=True),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert resumed.returncode == 0, resumed.stderr
        result = json.loads(resumed.stdout.strip())

        assert result["recovery_attempts"] == {"first": 2, "second": 2}
        assert result["effect_count"] == 2
        assert result["preemption_events"] == [
            "REQUESTED",
            "CHECKPOINT_ACKNOWLEDGED",
            "EXECUTED",
        ]
        assert result["final_assignments"] == 0
        assert set(result["final_statuses"].values()) == {"COMPLETED"}
        assert result["resumed_checkpoint_step_id"] != result["old_checkpoint_step_id"]
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
