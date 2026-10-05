from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from prometheist import attention_store, db
from prometheist.attention import (
    AttentionTask,
    JITAttentionScheduler,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
    deterministic_task_id,
)
from prometheist.attention_observation import (
    HOST_CPU_RESOURCE_ID,
    HOST_MEMORY_RESOURCE_ID,
    LOCAL_LLM_RESOURCE_ID,
    HostResourceMetrics,
    LocalResourceAdmissionController,
    ResourceObservationError,
    ResourceSafetyPolicy,
    build_resource_observation,
    discover_local_execution_resources,
)
from prometheist.attention_resources import (
    ExecutionResourceClass,
    ProcessResourceEstimate,
    ResourceEstimateSource,
)
from prometheist.attention_store import load_scheduler, save_scheduler


NAMESPACE = UUID("23ffacb4-6996-44ea-8d63-c9ed7cd326f5")
CAPTURED_AT = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


class FixedProbe:
    def __init__(self, metrics: HostResourceMetrics) -> None:
        self.metrics = metrics

    def capture(self) -> HostResourceMetrics:
        return self.metrics.model_copy(deep=True)


class FailingProbe:
    def capture(self) -> HostResourceMetrics:
        raise OSError("counter unavailable")


def _metrics(
    *,
    cpu_percent: int = 10,
    memory_available_mib: int = 12_000,
) -> HostResourceMetrics:
    return HostResourceMetrics(
        platform="test",
        logical_cpu_count=8,
        cpu_utilization_percent=cpu_percent,
        load_1m=0,
        memory_total_mib=16_384,
        memory_available_mib=memory_available_mib,
    )


def _task(
    key: str,
    seq: int,
    *,
    llm: bool = False,
    estimate: ProcessResourceEstimate | None = None,
) -> AttentionTask:
    required = (
        [ExecutionResourceClass.LLM_INFERENCE]
        if llm
        else []
    )
    return AttentionTask(
        task_id=deterministic_task_id(NAMESPACE, key),
        task_key=key,
        created_seq=seq,
        metadata=SchedulingMetadata(
            criticality=TaskCriticality.USER_REQUESTED,
            service_class=ServiceClass.USER_WORK,
            required_resource_classes=required,
            process_resource_estimate=estimate,
        ),
    )


def _controller(
    scheduler: JITAttentionScheduler,
    probe: object,
    *,
    policy: ResourceSafetyPolicy | None = None,
) -> LocalResourceAdmissionController:
    return LocalResourceAdmissionController(
        scheduler,
        probe=probe,  # type: ignore[arg-type]
        policy=policy,
        clock=lambda: CAPTURED_AT,
        host_id="test-host",
    )


def test_local_discovery_defaults_to_one_llm_and_reserves_host_headroom():
    policy = ResourceSafetyPolicy()
    resources = discover_local_execution_resources(_metrics(), policy=policy)
    by_id = {resource.resource_id: resource for resource in resources}

    assert by_id[LOCAL_LLM_RESOURCE_ID].capacity == 1
    assert by_id[HOST_CPU_RESOURCE_ID].system_headroom >= 1
    assert by_id[HOST_MEMORY_RESOURCE_ID].system_headroom >= 1_024


def test_host_safe_scheduler_refuses_to_plan_without_an_observation():
    scheduler = JITAttentionScheduler.for_local_host()
    scheduler.submit(_task("missing-observation", 1))

    with pytest.raises(ResourceObservationError, match="fresh resource observation"):
        scheduler.plan_scheduling_epoch()


def test_unestimated_work_gets_a_persisted_conservative_guess():
    scheduler = JITAttentionScheduler.for_local_host()
    task = scheduler.submit(_task("ordinary", 1))

    plan = _controller(scheduler, FixedProbe(_metrics())).plan_scheduling_epoch()

    estimate = scheduler.tasks[task.task_id].metadata.process_resource_estimate
    assert estimate is not None
    assert estimate.source is ResourceEstimateSource.CONSERVATIVE_DEFAULT
    assert estimate.cpu_units == 1
    assert estimate.memory_mib == 512
    assert {
        reservation.resource_class: reservation.units
        for reservation in plan.epoch.reservations
    } == {
        ExecutionResourceClass.CPU_GENERAL: 1,
        ExecutionResourceClass.MEMORY_RAM: 512,
    }


def test_llm_guess_is_larger_and_default_capacity_admits_only_one_llm():
    scheduler = JITAttentionScheduler.for_local_host()
    first = scheduler.submit(_task("first-llm", 1, llm=True))
    second = scheduler.submit(_task("second-llm", 2, llm=True))

    plan = _controller(scheduler, FixedProbe(_metrics())).plan_scheduling_epoch()

    first_estimate = scheduler.tasks[
        first.task_id
    ].metadata.process_resource_estimate
    assert first_estimate is not None
    assert first_estimate.memory_mib == 4_096
    assert first_estimate.llm_slots == 1
    assert plan.admitted_task_ids == [first.task_id]
    assert plan.unadmitted_task_ids == [second.task_id]


def test_profiled_estimate_is_preserved_instead_of_replaced_by_guess():
    estimate = ProcessResourceEstimate(
        cpu_units=2,
        memory_mib=768,
        source=ResourceEstimateSource.PROFILED,
        basis="peak of 20 representative runs",
    )
    scheduler = JITAttentionScheduler.for_local_host()
    task = scheduler.submit(_task("profiled", 1, estimate=estimate))
    revision_before_refresh = scheduler.tasks[task.task_id].revision

    _controller(scheduler, FixedProbe(_metrics())).refresh()

    assert (
        scheduler.tasks[task.task_id].metadata.process_resource_estimate
        == estimate
    )
    assert scheduler.tasks[task.task_id].revision == revision_before_refresh


def test_high_cpu_pressure_green_lights_no_new_work():
    scheduler = JITAttentionScheduler.for_local_host()
    task = scheduler.submit(_task("cpu-blocked", 1))

    plan = _controller(
        scheduler,
        FixedProbe(_metrics(cpu_percent=90)),
    ).plan_scheduling_epoch()

    assert plan.admitted_task_ids == []
    assert plan.unadmitted_task_ids == [task.task_id]
    observation = plan.resource_observation
    assert observation is not None
    cpu = observation.capacity_by_resource_id()[HOST_CPU_RESOURCE_ID]
    assert cpu.admission_capacity == 0


def test_low_available_ram_green_lights_no_new_work():
    scheduler = JITAttentionScheduler.for_local_host()
    task = scheduler.submit(_task("ram-blocked", 1))

    plan = _controller(
        scheduler,
        FixedProbe(_metrics(memory_available_mib=1_200)),
    ).plan_scheduling_epoch()

    assert plan.admitted_task_ids == []
    assert plan.unadmitted_task_ids == [task.task_id]
    observation = plan.resource_observation
    assert observation is not None
    memory = observation.capacity_by_resource_id()[HOST_MEMORY_RESOURCE_ID]
    assert memory.admission_capacity == 0


def test_probe_failure_is_persistable_fail_closed_input():
    scheduler = JITAttentionScheduler.for_local_host()
    task = scheduler.submit(_task("probe-failure", 1))

    plan = _controller(scheduler, FailingProbe()).plan_scheduling_epoch()

    assert plan.admitted_task_ids == []
    assert plan.unadmitted_task_ids == [task.task_id]
    observation = plan.resource_observation
    assert observation is not None
    assert observation.healthy is False
    assert observation.metrics is None
    assert observation.probe_errors == ["OSError: counter unavailable"]


def test_stale_observation_is_rejected_before_policy_evaluation():
    policy = ResourceSafetyPolicy(observation_max_age_seconds=5)
    scheduler = JITAttentionScheduler.for_local_host(
        resource_safety_policy_version=policy.policy_version
    )
    resources = discover_local_execution_resources(_metrics(), policy=policy)
    for resource in resources:
        scheduler.configure_execution_resource(resource)
    observation = build_resource_observation(
        scheduler_cycle=1,
        captured_at=CAPTURED_AT,
        resources=scheduler.execution_resources(),
        reservations=[],
        policy=policy,
        metrics=_metrics(),
    )

    with pytest.raises(ResourceObservationError, match="stale"):
        scheduler.attach_resource_observation(
            observation,
            evaluated_at=CAPTURED_AT + timedelta(seconds=6),
        )


def test_identical_state_observation_and_policy_produce_identical_plans():
    def plan() -> dict[str, object]:
        scheduler = JITAttentionScheduler.for_local_host()
        scheduler.submit(_task("deterministic", 1, llm=True))
        return _controller(
            scheduler,
            FixedProbe(_metrics()),
        ).plan_scheduling_epoch().model_dump(mode="json")

    assert plan() == plan()


def test_observation_survives_in_memory_commit_and_snapshot_round_trip():
    scheduler = JITAttentionScheduler.for_local_host()
    scheduler.submit(_task("round-trip", 1))
    plan = _controller(
        scheduler,
        FixedProbe(_metrics()),
    ).plan_scheduling_epoch()
    scheduler._commit_pending_epoch(plan.epoch.epoch_id)

    snapshot = scheduler.snapshot()
    restored = JITAttentionScheduler.from_snapshot(snapshot)

    assert restored.snapshot().model_dump(mode="json") == snapshot.model_dump(
        mode="json"
    )
    assert restored.current_epoch is not None
    assert (
        restored.current_epoch.resource_observation_id
        == snapshot.current_resource_observation.observation_id  # type: ignore[union-attr]
    )


def test_resource_observation_and_estimate_round_trip_through_postgres():
    conn = db.get_connection()
    try:
        scheduler = JITAttentionScheduler.for_local_host()
        task = scheduler.submit(_task("postgres", 1, llm=True))
        plan = _controller(
            scheduler,
            FixedProbe(_metrics()),
        ).plan_scheduling_epoch()
        save_scheduler(conn, scheduler)
    finally:
        conn.close()

    restarted_conn = db.get_connection()
    try:
        restored = load_scheduler(restarted_conn)
        observation = restored.current_resource_observation()

        assert observation is not None
        assert plan.resource_observation is not None
        assert (
            observation.observation_id
            == plan.resource_observation.observation_id
        )
        assert restored.current_epoch is not None
        assert restored.current_epoch.resource_observation_id == observation.observation_id
        estimate = restored.tasks[task.task_id].metadata.process_resource_estimate
        assert estimate is not None
        assert estimate.source is ResourceEstimateSource.CONSERVATIVE_DEFAULT
        assert estimate.memory_mib == 4_096
        with restarted_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM attention_resource_observations")
            assert int(cur.fetchone()[0]) == 1
    finally:
        restarted_conn.close()


def test_failed_postgres_host_safe_epoch_rolls_back_observation_and_assignments(
    monkeypatch: pytest.MonkeyPatch,
):
    conn = db.get_connection()
    try:
        scheduler = JITAttentionScheduler.for_local_host()
        scheduler.submit(_task("rollback", 1))
        plan = _controller(
            scheduler,
            FixedProbe(_metrics()),
        ).plan_scheduling_epoch()

        def fail_epoch_insert(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected epoch failure")

        monkeypatch.setattr(
            attention_store,
            "_insert_immutable_epoch",
            fail_epoch_insert,
        )
        with pytest.raises(RuntimeError, match="injected epoch failure"):
            save_scheduler(conn, scheduler)

        with conn.cursor() as cur:
            for table in (
                "attention_resource_observations",
                "attention_assignments",
                "attention_scheduling_epochs",
                "attention_scheduler_state",
            ):
                cur.execute(f"SELECT count(*) FROM {table}")
                assert int(cur.fetchone()[0]) == 0
        assert scheduler.pending_scheduling_epoch() == plan
        assert scheduler.worker_visible_assignments() == []
    finally:
        conn.close()


def test_postgres_enforces_one_llm_globally_across_scheduler_keys():
    first_conn = db.get_connection()
    second_conn = db.get_connection()
    try:
        first = JITAttentionScheduler.for_local_host()
        first.submit(_task("global-llm-one", 1, llm=True))
        _controller(first, FixedProbe(_metrics())).plan_scheduling_epoch()
        save_scheduler(first_conn, first, scheduler_key="first")

        second = JITAttentionScheduler.for_local_host()
        second.submit(_task("global-llm-two", 2, llm=True))
        _controller(second, FixedProbe(_metrics())).plan_scheduling_epoch()
        with pytest.raises(
            RuntimeError,
            match="Global reservation capacity exceeded for 'local-llm'",
        ):
            save_scheduler(second_conn, second, scheduler_key="second")

        with second_conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)
                FROM attention_scheduler_state
                WHERE scheduler_key = 'second'
                """
            )
            assert int(cur.fetchone()[0]) == 0
            cur.execute(
                """
                SELECT sum(units)
                FROM attention_resource_reservations
                WHERE resource_id = 'local-llm'
                """
            )
            assert int(cur.fetchone()[0]) == 1
    finally:
        first_conn.close()
        second_conn.close()
