from __future__ import annotations

from datetime import datetime, timezone
import uuid

from prometheist.admission_diagnostics import build_resource_admission_diagnostics
from prometheist.attention import (
    AttentionTask,
    InterruptionPolicy,
    JITAttentionScheduler,
    SchedulingMetadata,
    ServiceClass,
    TaskCriticality,
)
from prometheist.attention_observation import (
    HostResourceMetrics,
    LocalResourceAdmissionController,
)
from prometheist.attention_resources import ProcessResourceEstimate, ResourceEstimateSource


NOW = datetime(2026, 8, 27, 8, 0, tzinfo=timezone.utc)


class LowMemoryProbe:
    def capture(self) -> HostResourceMetrics:
        return HostResourceMetrics(
            platform="test",
            logical_cpu_count=8,
            cpu_utilization_percent=0,
            load_1m=0,
            memory_total_mib=16_384,
            memory_available_mib=6_000,
        )


def test_admission_diagnostics_expose_memory_deficit_without_changing_policy():
    scheduler = JITAttentionScheduler.for_local_host()
    task = AttentionTask(
        task_id=uuid.uuid4(),
        task_key="diagnostic-test",
        created_seq=1,
        metadata=SchedulingMetadata(
            criticality=TaskCriticality.USER_BLOCKING,
            service_class=ServiceClass.INTERACTIVE,
            interruption_policy=InterruptionPolicy.CHECKPOINT_ONLY,
            process_resource_estimate=ProcessResourceEstimate(
                cpu_units=1,
                memory_mib=4096,
                llm_slots=1,
                source=ResourceEstimateSource.CONSERVATIVE_DEFAULT,
                basis="test",
            ),
        ),
    )
    scheduler.submit(task)
    controller = LocalResourceAdmissionController(
        scheduler,
        probe=LowMemoryProbe(),
        clock=lambda: NOW,
    )

    controller.plan_scheduling_epoch()
    diagnostics = build_resource_admission_diagnostics(scheduler)

    observation = diagnostics["resource_observation"]
    assert observation["host_metrics"]["memory_available_mib"] == 6_000
    memory_capacity = next(
        item
        for item in observation["capacities"]
        if item["resource_class"] == "MEMORY_RAM"
    )
    assert memory_capacity["available_for_new_work"] == 3_488

    task_diagnostic = diagnostics["unassigned_tasks"][0]
    assert task_diagnostic["requirements_by_resource_class"] == {
        "CPU_GENERAL": 1,
        "MEMORY_RAM": 4096,
        "LLM_INFERENCE": 1,
    }
    assert task_diagnostic["deficits_by_resource_class"] == {
        "CPU_GENERAL": 0,
        "LLM_INFERENCE": 0,
        "MEMORY_RAM": 608,
    }
