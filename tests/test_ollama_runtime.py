from __future__ import annotations

from datetime import datetime, timezone

import pytest

import httpx

from persistent_cognition.attention_observation import (
    HOST_MEMORY_RESOURCE_ID,
    HostResourceMetrics,
    build_resource_observation,
    discover_local_execution_resources,
)
from persistent_cognition.native_policy import native_resource_safety_policy
from persistent_cognition.ollama_runtime import OllamaClaimHostResourceProbe, OllamaRuntimeProbe


_MIB = 1024 * 1024


def _client(payload: dict, *, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(status_code, json=payload, request=request)

    return httpx.Client(
        base_url="https://ollama.test",
        transport=httpx.MockTransport(handler),
    )


class FixedHostProbe:
    def __init__(self, available_mib: int = 3_877) -> None:
        self.available_mib = available_mib

    def capture(self) -> HostResourceMetrics:
        return HostResourceMetrics(
            platform="test",
            logical_cpu_count=8,
            cpu_utilization_percent=10,
            load_1m=0,
            memory_total_mib=16_118,
            memory_available_mib=self.available_mib,
        )


def _warm_runtime_probe() -> OllamaRuntimeProbe:
    return OllamaRuntimeProbe(
        model="qwen3:4b",
        client=_client(
            {
                "models": [
                    {
                        "name": "qwen3:4b",
                        "model": "qwen3:4b",
                        "size": 3_000 * _MIB,
                        "size_vram": 0,
                        "expires_at": "2026-08-27T08:30:00-07:00",
                    }
                ]
            }
        ),
    )


def _cold_runtime_probe() -> OllamaRuntimeProbe:
    return OllamaRuntimeProbe(
        model="qwen3:4b",
        client=_client({"models": [{"name": "other:latest", "size": 1}]}),
    )


def test_running_model_is_detected_without_loading_it():
    state = _warm_runtime_probe().capture()

    assert state.probe_ok is True
    assert state.resident is True
    assert state.reported_name == "qwen3:4b"
    assert state.system_memory_mib == 3_000
    assert state.reusable_memory_credit_mib(native_resource_safety_policy()) == 2_560
    assert state.incremental_process_memory_mib(native_resource_safety_policy()) == 512
    assert state.residency_label == "warm-resident"


def test_running_model_credit_excludes_reported_vram_and_stays_bounded():
    probe = OllamaRuntimeProbe(
        model="qwen3:4b",
        client=_client(
            {
                "models": [
                    {
                        "name": "qwen3:4b",
                        "size": 3_000 * _MIB,
                        "size_vram": 2_000 * _MIB,
                    }
                ]
            }
        ),
    )

    state = probe.capture()

    assert state.system_memory_mib == 1_000
    assert state.reusable_memory_credit_mib(native_resource_safety_policy()) == 1_000
    assert state.incremental_process_memory_mib(native_resource_safety_policy()) == 512


def test_nonresident_or_failed_runtime_probe_falls_back_to_full_cold_requirement():
    policy = native_resource_safety_policy()
    cold = _cold_runtime_probe().capture()
    failed = OllamaRuntimeProbe(
        model="qwen3:4b",
        client=_client({}, status_code=503),
    ).capture()

    assert cold.probe_ok is True
    assert cold.resident is False
    assert cold.reusable_memory_credit_mib(policy) == 0
    assert cold.incremental_process_memory_mib(policy) == 3_072
    assert cold.residency_label == "cold-nonresident"
    assert failed.probe_ok is False
    assert failed.resident is False
    assert failed.reusable_memory_credit_mib(policy) == 0
    assert failed.incremental_process_memory_mib(policy) == 3_072
    assert failed.residency_label == "unverified-cold-fallback"


def test_cold_reservation_that_becomes_warm_gets_bounded_claim_credit():
    policy = native_resource_safety_policy()
    probe = OllamaClaimHostResourceProbe(
        base_probe=FixedHostProbe(),
        runtime_probe=_warm_runtime_probe(),
        policy=policy,
        scheduled_memory_mib=3_072,
    )

    effective = probe.capture()

    assert probe.last_physical_metrics is not None
    assert probe.last_physical_metrics.memory_available_mib == 3_877
    assert probe.last_runtime_state is not None
    assert probe.last_runtime_state.resident is True
    assert probe.last_effective_required_memory_mib == 512
    assert probe.last_memory_credit_mib == 2_560
    assert probe.last_memory_debit_mib == 0
    assert effective.memory_available_mib == 6_437


def test_warm_reservation_that_remains_warm_uses_physical_ram_without_credit():
    policy = native_resource_safety_policy()
    probe = OllamaClaimHostResourceProbe(
        base_probe=FixedHostProbe(available_mib=4_146),
        runtime_probe=_warm_runtime_probe(),
        policy=policy,
        scheduled_memory_mib=512,
    )

    effective = probe.capture()

    assert probe.last_effective_required_memory_mib == 512
    assert probe.last_memory_credit_mib == 0
    assert probe.last_memory_debit_mib == 0
    assert effective.memory_available_mib == 4_146


def test_warm_reservation_that_turns_cold_is_debited_before_claim_gate():
    policy = native_resource_safety_policy()
    probe = OllamaClaimHostResourceProbe(
        base_probe=FixedHostProbe(available_mib=4_146),
        runtime_probe=_cold_runtime_probe(),
        policy=policy,
        scheduled_memory_mib=512,
    )

    effective = probe.capture()

    assert probe.last_effective_required_memory_mib == 3_072
    assert probe.last_memory_credit_mib == 0
    assert probe.last_memory_debit_mib == 2_695
    assert effective.memory_available_mib == 1_451


def _memory_admission_capacity(
    probe: OllamaClaimHostResourceProbe,
) -> int:
    policy = native_resource_safety_policy()
    metrics = probe.capture()
    observation = build_resource_observation(
        scheduler_cycle=1,
        captured_at=datetime.now(timezone.utc),
        resources=discover_local_execution_resources(metrics, policy=policy),
        reservations=[],
        policy=policy,
        metrics=metrics,
    )
    return observation.capacity_by_resource_id()[HOST_MEMORY_RESOURCE_ID].admission_capacity


def test_logged_cold_memory_snapshot_fits_with_fixed_one_gib_headroom():
    # October 8 acceptance log: 4,744 MiB free, 3,072 MiB cold-load claim.
    policy = native_resource_safety_policy()
    probe = OllamaClaimHostResourceProbe(
        base_probe=FixedHostProbe(available_mib=4_744),
        runtime_probe=_cold_runtime_probe(),
        policy=policy,
        scheduled_memory_mib=3_072,
    )
    assert _memory_admission_capacity(probe) == 3_534
    assert probe.last_effective_required_memory_mib == 3_072


def test_verified_warm_model_uses_only_incremental_worker_budget():
    policy = native_resource_safety_policy()
    probe = OllamaClaimHostResourceProbe(
        base_probe=FixedHostProbe(available_mib=2_150),
        runtime_probe=_warm_runtime_probe(),
        policy=policy,
        scheduled_memory_mib=512,
    )
    assert _memory_admission_capacity(probe) >= 512
    assert probe.last_effective_required_memory_mib == 512
    assert probe.last_memory_credit_mib == 0


def test_unverified_model_cannot_borrow_warm_memory_budget():
    policy = native_resource_safety_policy()
    failed_runtime = OllamaRuntimeProbe(
        model="qwen3:4b",
        client=_client({}, status_code=503),
    )
    probe = OllamaClaimHostResourceProbe(
        base_probe=FixedHostProbe(available_mib=2_150),
        runtime_probe=failed_runtime,
        policy=policy,
        scheduled_memory_mib=3_072,
    )
    assert _memory_admission_capacity(probe) < 3_072
    assert probe.last_runtime_state is not None
    assert probe.last_runtime_state.probe_ok is False
    assert probe.last_effective_required_memory_mib == 3_072


pytestmark = pytest.mark.usefixtures("consented_mock_ollama")
