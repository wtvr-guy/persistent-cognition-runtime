from __future__ import annotations

import uuid

import pytest

from prometheist import db
from prometheist.attention_observation import HostResourceMetrics
from prometheist.attention_store import load_scheduler
from prometheist.percept_response_runtime import begin_percept
from prometheist.native_policy import native_resource_safety_policy
from prometheist.ollama_runtime import OllamaRuntimeState


class ConstrainedHostProbe:
    def capture(self) -> HostResourceMetrics:
        return HostResourceMetrics(
            platform="Windows",
            logical_cpu_count=8,
            cpu_utilization_percent=0,
            load_1m=None,
            memory_total_mib=16_118,
            memory_available_mib=4_146,
        )


def _runtime_state(*, resident: bool) -> OllamaRuntimeState:
    return OllamaRuntimeState(
        model="qwen3:4b",
        probe_ok=True,
        resident=resident,
        reported_name="qwen3:4b" if resident else None,
        size_bytes=3_184_001_022 if resident else None,
        size_vram_bytes=0 if resident else None,
    )


def test_resident_model_reduces_interaction_admission_to_incremental_worker_ram():
    conn = db.get_connection()
    try:
        interaction = begin_percept(
            conn,
            "warm admission",
            uuid.uuid4(),
            probe=ConstrainedHostProbe(),
            policy=native_resource_safety_policy(),
            ollama_runtime_state=_runtime_state(resident=True),
        )
        scheduler = load_scheduler(conn)
        task = scheduler.tasks[interaction.task_id]
        estimate = task.metadata.process_resource_estimate

        assert estimate is not None
        assert estimate.memory_mib == 512
        assert "warm-resident" in estimate.basis
        assert task.resumable_state["resource_admission"]["memory_mib"] == 512
        assert task.resumable_state["resource_admission"]["ollama_runtime_state"][
            "resident"
        ] is True
        assert scheduler.worker_visible_assignments()[0].task_id == interaction.task_id
    finally:
        conn.close()


def test_same_host_still_denies_full_cold_model_load():
    conn = db.get_connection()
    try:
        with pytest.raises(
            RuntimeError,
            match="interaction was not safely admitted to one assignment",
        ):
            begin_percept(
                conn,
                "cold admission",
                uuid.uuid4(),
                probe=ConstrainedHostProbe(),
                policy=native_resource_safety_policy(),
                ollama_runtime_state=_runtime_state(resident=False),
            )

        scheduler = load_scheduler(conn)
        task = next(iter(scheduler.tasks.values()))
        estimate = task.metadata.process_resource_estimate
        assert estimate is not None
        assert estimate.memory_mib == 3_072
        assert "cold-nonresident" in estimate.basis
        assert scheduler.worker_visible_assignments() == []
    finally:
        conn.close()
