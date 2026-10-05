"""Structured evidence for resource-admission failures.

Diagnostics are intentionally observational.  They explain the authoritative
resource envelope that the deterministic scheduler consumed without changing
admission policy or retrying denied work.
"""
from __future__ import annotations

from typing import Any

from prometheist.attention import JITAttentionScheduler, TaskStatus


RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX = (
    "PROMETHEIST_RESOURCE_ADMISSION_DIAGNOSTICS="
)


def build_resource_admission_diagnostics(
    scheduler: JITAttentionScheduler,
) -> dict[str, Any]:
    """Describe why unfinished work could not fit the observed safe envelope."""

    observation = (
        scheduler.current_resource_observation()
        or scheduler.pending_resource_observation()
    )
    assigned_task_ids = {
        assignment.task_id for assignment in scheduler.worker_visible_assignments()
    }
    unassigned_tasks = sorted(
        (
            task
            for task in scheduler.tasks.values()
            if task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}
            and task.task_id not in assigned_task_ids
        ),
        key=lambda task: (task.created_seq, task.task_id.hex),
    )

    if observation is None:
        return {
            "scheduler_cycle": scheduler.cycle,
            "epoch_sequence": scheduler.epoch_sequence,
            "resource_observation": None,
            "unassigned_tasks": [
                {
                    "task_id": str(task.task_id),
                    "task_key": task.task_key,
                    "status": task.status.value,
                }
                for task in unassigned_tasks
            ],
        }

    available_by_class: dict[str, int] = {}
    capacities: list[dict[str, Any]] = []
    for capacity in observation.capacities:
        class_name = capacity.resource_class.value
        available_by_class[class_name] = (
            available_by_class.get(class_name, 0)
            + capacity.available_for_new_work
        )
        row = capacity.model_dump(mode="json")
        row["available_for_new_work"] = capacity.available_for_new_work
        capacities.append(row)

    task_rows: list[dict[str, Any]] = []
    for task in unassigned_tasks:
        requirements = task.metadata.quantitative_resource_requirements()
        required_by_class = {
            requirement.resource_class.value: requirement.units
            for requirement in requirements
        }
        deficits = {
            class_name: max(
                0,
                required_units - available_by_class.get(class_name, 0),
            )
            for class_name, required_units in required_by_class.items()
        }
        task_rows.append(
            {
                "task_id": str(task.task_id),
                "task_key": task.task_key,
                "status": task.status.value,
                "process_resource_estimate": (
                    task.metadata.process_resource_estimate.model_dump(mode="json")
                    if task.metadata.process_resource_estimate is not None
                    else None
                ),
                "requirements_by_resource_class": required_by_class,
                "available_for_new_work_by_resource_class": dict(
                    sorted(available_by_class.items())
                ),
                "deficits_by_resource_class": dict(sorted(deficits.items())),
            }
        )

    return {
        "scheduler_cycle": scheduler.cycle,
        "epoch_sequence": scheduler.epoch_sequence,
        "resource_observation": {
            "observation_id": str(observation.observation_id),
            "captured_at": observation.captured_at.isoformat(),
            "valid_until": observation.valid_until.isoformat(),
            "healthy": observation.healthy,
            "probe_errors": list(observation.probe_errors),
            "host_metrics": (
                observation.metrics.model_dump(mode="json")
                if observation.metrics is not None
                else None
            ),
            "policy": observation.policy.model_dump(mode="json"),
            "capacities": capacities,
        },
        "unassigned_tasks": task_rows,
    }
