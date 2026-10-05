from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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
from prometheist.attention_observation import (
    HostResourceMetrics,
    LocalResourceAdmissionController,
    ResourceSafetyPolicy,
)
from prometheist.attention_resources import (
    ExecutionResourceClass,
    ProcessResourceEstimate,
    ResourceEstimateSource,
)
from prometheist.attention_store import save_scheduler
from prometheist.worker_protocol import (
    WorkerClaimDecision,
    WorkerClaimStatus,
    WorkerEffectPolicy,
    WorkerStep,
    deterministic_worker_claim_id,
    deterministic_worker_idempotency_key,
    deterministic_worker_step_id,
)
from prometheist.worker_runtime import GuardedWorkerLauncher, WorkerLaunchDenied
from prometheist.worker_store import (
    WorkerProtocolError,
    checkpoint_worker_claim,
    complete_worker_claim,
    guarded_claim_worker_step,
    heartbeat_worker_claim,
    load_claim_observations,
    load_worker_claim,
    load_worker_claim_envelope,
    load_worker_result,
    register_worker_step,
    release_worker_claim,
)


NAMESPACE = UUID("6cb00f48-c41e-4850-8894-205638ab2c29")
CAPTURED_AT = datetime(2026, 8, 26, 16, 0, tzinfo=timezone.utc)


class FixedProbe:
    def __init__(self, *, cpu_percent: int = 10) -> None:
        self.cpu_percent = cpu_percent

    def capture(self) -> HostResourceMetrics:
        return HostResourceMetrics(
            platform="test",
            logical_cpu_count=8,
            cpu_utilization_percent=self.cpu_percent,
            load_1m=0,
            memory_total_mib=16_384,
            memory_available_mib=12_000,
        )


class MutableClock:
    def __init__(self, value: datetime = CAPTURED_AT) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _task(
    key: str,
    seq: int,
    *,
    llm: bool = False,
    criticality: TaskCriticality = TaskCriticality.USER_REQUESTED,
) -> AttentionTask:
    estimate = ProcessResourceEstimate(
        cpu_units=1,
        memory_mib=512,
        llm_slots=1 if llm else 0,
        source=ResourceEstimateSource.DECLARED,
        basis="bounded test worker",
    )
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=seq,
        metadata=SchedulingMetadata(
            criticality=criticality,
            service_class=ServiceClass.USER_WORK,
            required_capabilities=["test.echo"],
            required_resource_classes=(
                [ExecutionResourceClass.LLM_INFERENCE] if llm else []
            ),
            process_resource_estimate=estimate,
        ),
    )


def _prepare_step(
    conn,
    *,
    effect_policy: WorkerEffectPolicy = WorkerEffectPolicy.NO_EXTERNAL_EFFECT,
    llm: bool = False,
):
    scheduler = JITAttentionScheduler.for_local_host()
    scheduler.submit(_task("worker-task", 1, llm=llm))
    controller = LocalResourceAdmissionController(
        scheduler,
        probe=FixedProbe(),
        clock=lambda: CAPTURED_AT,
        host_id="test-host",
    )
    controller.plan_scheduling_epoch()
    save_scheduler(conn, scheduler)
    assignment = scheduler.worker_visible_assignments()[0]
    step = register_worker_step(
        conn,
        assignment_id=assignment.assignment_id,
        step_key="bounded-step-0",
        capability="test.echo",
        input_refs=["event:input-1"],
        effect_policy=effect_policy,
    )
    return scheduler, step


def test_worker_step_identity_and_effect_key_are_stable_and_agent_neutral():
    conn = db.get_connection()
    try:
        _, step = _prepare_step(
            conn,
            effect_policy=WorkerEffectPolicy.IDEMPOTENT_WITH_KEY,
        )
        repeated = register_worker_step(
            conn,
            assignment_id=step.assignment_id,
            step_key=step.step_key,
            capability=step.capability,
            input_refs=step.input_refs,
            effect_policy=step.effect_policy,
        )

        assert repeated == step
        assert step.step_id == deterministic_worker_step_id(
            step.assignment_id,
            step.step_key,
        )
        assert step.idempotency_key == deterministic_worker_idempotency_key(
            step.step_id
        )
        assert "agent" not in WorkerStep.model_fields
        with pytest.raises(WorkerProtocolError, match="not declared"):
            register_worker_step(
                conn,
                assignment_id=step.assignment_id,
                step_key="invented-step",
                capability="worker.invented",
            )
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM attention_worker_steps")
            assert int(cur.fetchone()[0]) == 1
    finally:
        conn.close()


def test_claim_time_reobservation_denies_pressure_that_changed_after_assignment():
    conn = db.get_connection()
    try:
        _, step = _prepare_step(conn)

        attempt = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(cpu_percent=95),
            clock=lambda: CAPTURED_AT + timedelta(seconds=1),
        )

        assert attempt.envelope is None
        assert attempt.observation.decision is WorkerClaimDecision.DENIED
        assert "host-cpu" in attempt.observation.reason
        assert load_claim_observations(conn, step.step_id) == [
            attempt.observation
        ]
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM attention_worker_claims")
            assert int(cur.fetchone()[0]) == 0
    finally:
        conn.close()


def test_claim_uses_exact_policy_committed_with_the_epoch():
    conn = db.get_connection()
    try:
        _, step = _prepare_step(conn)

        attempt = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            policy=ResourceSafetyPolicy(max_cpu_pressure_percent=99),
            clock=lambda: CAPTURED_AT,
        )

        assert attempt.envelope is None
        assert "exact committed policy" in attempt.observation.reason
        assert attempt.observation.resource_snapshot.policy == ResourceSafetyPolicy()
    finally:
        conn.close()


def test_stale_claim_observation_is_persisted_as_a_denial():
    conn = db.get_connection()
    timestamps = iter(
        [
            CAPTURED_AT,
            CAPTURED_AT + timedelta(seconds=6),
        ]
    )
    try:
        _, step = _prepare_step(conn)

        attempt = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            clock=lambda: next(timestamps),
        )

        assert attempt.envelope is None
        assert "expired before publication" in attempt.observation.reason
        assert attempt.observation.evaluated_at > attempt.observation.valid_until
        assert load_claim_observations(conn, step.step_id) == [attempt.observation]
    finally:
        conn.close()


def test_live_claim_exclusion_and_idempotent_abandoned_recovery():
    conn = db.get_connection()
    clock = MutableClock()
    try:
        _, step = _prepare_step(
            conn,
            effect_policy=WorkerEffectPolicy.IDEMPOTENT_WITH_KEY,
        )
        first = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            clock=clock,
            lease_seconds=5,
        )
        assert first.envelope is not None
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TEMP TABLE fake_idempotent_effects (
                    idempotency_key TEXT PRIMARY KEY,
                    calls INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                """
                INSERT INTO fake_idempotent_effects (idempotency_key, calls)
                VALUES (%s, 1)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (first.envelope.step.idempotency_key,),
            )
        conn.commit()

        blocked = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-b",
            probe=FixedProbe(),
            clock=clock,
            lease_seconds=5,
        )
        assert blocked.envelope is None
        assert "live claim" in blocked.observation.reason

        clock.advance(6)
        recovered = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-b",
            probe=FixedProbe(),
            clock=clock,
            lease_seconds=5,
        )
        assert recovered.envelope is not None
        assert recovered.envelope.claim.attempt == 2
        assert recovered.envelope.step.idempotency_key == (
            first.envelope.step.idempotency_key
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fake_idempotent_effects (idempotency_key, calls)
                VALUES (%s, 1)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (recovered.envelope.step.idempotency_key,),
            )
            cur.execute("SELECT count(*), sum(calls) FROM fake_idempotent_effects")
            assert cur.fetchone() == (1, 1)
        conn.commit()
        assert load_worker_claim(
            conn,
            first.envelope.claim.claim_id,
        ).status is WorkerClaimStatus.ABANDONED
    finally:
        conn.close()


def test_heartbeat_extends_a_live_claim_before_recovery_is_allowed():
    conn = db.get_connection()
    clock = MutableClock()
    try:
        _, step = _prepare_step(conn)
        first = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            clock=clock,
            lease_seconds=5,
        )
        assert first.envelope is not None

        clock.advance(3)
        heartbeat = heartbeat_worker_claim(
            conn,
            claim_id=first.envelope.claim.claim_id,
            worker_id="worker-a",
            clock=clock,
            lease_seconds=5,
        )
        assert heartbeat.lease_expires_at == CAPTURED_AT + timedelta(seconds=8)

        clock.advance(3)
        blocked = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-b",
            probe=FixedProbe(),
            clock=clock,
        )
        assert blocked.envelope is None
        assert "live claim" in blocked.observation.reason

        clock.advance(3)
        recovered = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-b",
            probe=FixedProbe(),
            clock=clock,
        )
        assert recovered.envelope is not None
        assert recovered.envelope.claim.attempt == 2
    finally:
        conn.close()


def test_fresh_worker_resumes_from_latest_committed_checkpoint():
    conn = db.get_connection()
    clock = MutableClock()
    try:
        _, step = _prepare_step(conn)
        first = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            clock=clock,
        )
        assert first.envelope is not None
        clock.advance(1)
        checkpoint = checkpoint_worker_claim(
            conn,
            claim_id=first.envelope.claim.claim_id,
            worker_id="worker-a",
            state={"record_offset": 41},
            output_refs=["artifact:partial-41"],
            clock=clock,
        )

        clock.advance(1)
        resumed = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="fresh-worker",
            probe=FixedProbe(),
            clock=clock,
        )

        assert resumed.envelope is not None
        assert resumed.envelope.claim.attempt == 2
        assert resumed.envelope.claim.checkpoint_revision == 1
        assert resumed.envelope.checkpoint == checkpoint
        assert resumed.envelope.checkpoint.state == {"record_offset": 41}
        loaded = load_worker_claim_envelope(
            conn,
            resumed.envelope.claim.claim_id,
            worker_id="fresh-worker",
            clock=clock,
        )
        assert loaded == resumed.envelope
    finally:
        conn.close()


def test_terminal_result_is_idempotent_and_prevents_completed_work_retry():
    conn = db.get_connection()
    clock = MutableClock()
    try:
        _, step = _prepare_step(
            conn,
            effect_policy=WorkerEffectPolicy.IDEMPOTENT_WITH_KEY,
        )
        attempt = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            clock=clock,
        )
        assert attempt.envelope is not None
        clock.advance(1)
        result = complete_worker_claim(
            conn,
            claim_id=attempt.envelope.claim.claim_id,
            worker_id="worker-a",
            output={"answer": 42},
            output_refs=["event:result-42"],
            clock=clock,
        )
        clock.advance(1)
        repeated = complete_worker_claim(
            conn,
            claim_id=attempt.envelope.claim.claim_id,
            worker_id="worker-a",
            output={"answer": 42},
            output_refs=["event:result-42"],
            clock=clock,
        )
        assert repeated == result
        assert load_worker_result(conn, step.step_id) == result

        denied = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-b",
            probe=FixedProbe(),
            clock=clock,
        )
        assert denied.envelope is None
        assert "terminal result" in denied.observation.reason
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM attention_worker_results")
            assert int(cur.fetchone()[0]) == 1
    finally:
        conn.close()


def test_at_most_once_abandonment_fails_closed_instead_of_guessing():
    conn = db.get_connection()
    clock = MutableClock()
    try:
        _, step = _prepare_step(
            conn,
            effect_policy=WorkerEffectPolicy.AT_MOST_ONCE,
        )
        first = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            clock=clock,
            lease_seconds=1,
        )
        assert first.envelope is not None
        clock.advance(2)

        denied = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-b",
            probe=FixedProbe(),
            clock=clock,
        )

        assert denied.envelope is None
        assert "requires explicit reconciliation" in denied.observation.reason
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM attention_worker_claims")
            assert int(cur.fetchone()[0]) == 1
    finally:
        conn.close()


def test_active_worker_claim_prevents_scheduler_from_releasing_its_capacity():
    conn = db.get_connection()
    clock = MutableClock(datetime.now(timezone.utc))
    try:
        scheduler, step = _prepare_step(conn, llm=True)
        attempt = guarded_claim_worker_step(
            conn,
            step_id=step.step_id,
            worker_id="worker-a",
            probe=FixedProbe(),
            clock=clock,
        )
        assert attempt.envelope is not None

        urgent = _task(
            "urgent",
            2,
            llm=True,
            criticality=TaskCriticality.USER_BLOCKING,
        )
        scheduler.submit(urgent)
        controller = LocalResourceAdmissionController(
            scheduler,
            probe=FixedProbe(),
            clock=lambda: CAPTURED_AT + timedelta(seconds=1),
            host_id="test-host",
        )
        plan = controller.plan_scheduling_epoch()
        assert plan.admitted_task_ids == [urgent.task_id]
        with pytest.raises(
            RuntimeError,
            match="active worker claim",
        ):
            save_scheduler(conn, scheduler)

        release_worker_claim(
            conn,
            claim_id=attempt.envelope.claim.claim_id,
            worker_id="worker-a",
            clock=clock,
        )
        save_scheduler(conn, scheduler)
        assert scheduler.worker_visible_assignments()[0].task_id == urgent.task_id
    finally:
        conn.close()


def test_forced_process_loss_recovers_same_step_from_postgres():
    conn = db.get_connection()
    try:
        _, step = _prepare_step(
            conn,
            effect_policy=WorkerEffectPolicy.IDEMPOTENT_WITH_KEY,
        )
    finally:
        conn.close()

    process = subprocess.run(
        [sys.executable, "-m", "tests._worker_process", str(step.step_id)],
        check=True,
        capture_output=True,
        text=True,
    )
    destroyed = json.loads(process.stdout.strip())

    recovered_conn = db.get_connection()
    try:
        recovered = guarded_claim_worker_step(
            recovered_conn,
            step_id=step.step_id,
            worker_id="fresh-process",
            probe=FixedProbe(),
            clock=lambda: CAPTURED_AT + timedelta(seconds=2),
        )
        assert recovered.envelope is not None
        assert recovered.envelope.claim.attempt == 2
        assert recovered.envelope.step.idempotency_key == destroyed["idempotency_key"]
    finally:
        recovered_conn.close()


def test_guarded_launcher_never_spawns_when_claim_is_denied():
    conn = db.get_connection()
    try:
        _, step = _prepare_step(conn)
    finally:
        conn.close()
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_process(command, *, env, cwd, shell):
        assert shell is False
        calls.append((list(command), dict(env)))
        return object()

    launcher = GuardedWorkerLauncher(
        db.get_connection,
        probe=FixedProbe(),
        clock=lambda: CAPTURED_AT,
        process_factory=fake_process,
    )
    launched = launcher.launch(
        step_id=step.step_id,
        worker_id="worker-a",
        command=[sys.executable, "-c", "pass"],
        env={},
    )
    assert len(calls) == 1
    assert calls[0][1]["PROMETHEIST_WORKER_CLAIM_ID"] == str(
        launched.envelope.claim.claim_id
    )
    assert calls[0][1]["PROMETHEIST_WORKER_ID"] == "worker-a"
    assert calls[0][1]["PROMETHEIST_WORKER_SCHEDULER_KEY"] == "default"

    with pytest.raises(WorkerLaunchDenied, match="live claim"):
        launcher.launch(
            step_id=step.step_id,
            worker_id="worker-b",
            command=[sys.executable, "-c", "pass"],
            env={},
        )
    assert len(calls) == 1


def test_guarded_launcher_releases_claim_when_process_spawn_fails():
    conn = db.get_connection()
    try:
        _, step = _prepare_step(conn)
    finally:
        conn.close()

    def failed_process(*_args, **_kwargs):
        raise OSError("synthetic spawn failure")

    launcher = GuardedWorkerLauncher(
        db.get_connection,
        probe=FixedProbe(),
        clock=lambda: CAPTURED_AT,
        process_factory=failed_process,
    )
    with pytest.raises(OSError, match="synthetic spawn failure"):
        launcher.launch(
            step_id=step.step_id,
            worker_id="worker-a",
            command=[sys.executable, "-c", "pass"],
            env={},
        )

    verification_conn = db.get_connection()
    try:
        claim = load_worker_claim(
            verification_conn,
            deterministic_worker_claim_id(step.step_id, 1),
        )
        assert claim.status is WorkerClaimStatus.RELEASED
    finally:
        verification_conn.close()
