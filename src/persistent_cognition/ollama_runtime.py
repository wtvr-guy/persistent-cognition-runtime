"""Target-specific Ollama residency observations for local interaction admission.

The operating-system memory probe reports resident model pages as unavailable.
Persistent Cognition therefore distinguishes a cold local-LLM launch from a warm launch
that reuses the configured model already resident in Ollama. Scheduler admission
uses the state-dependent incremental requirement; guarded worker claims recheck
residency and conservatively compensate if the state changed after scheduling.
"""
from __future__ import annotations

import math
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field, model_validator

from persistent_cognition.attention_observation import (
    HostResourceMetrics,
    HostResourceProbe,
    ResourceSafetyPolicy,
    SystemHostResourceProbe,
)


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
_MIB = 1024 * 1024


def configured_ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def configured_ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def _canonical_model_name(value: str) -> str:
    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("Ollama model name must not be empty")
    leaf = normalized.rsplit("/", 1)[-1]
    if ":" not in leaf:
        normalized += ":latest"
    return normalized


class OllamaRuntimeState(BaseModel):
    """One bounded observation of whether the configured model is resident."""

    model: str = Field(min_length=1)
    probe_ok: bool
    resident: bool
    reported_name: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    size_vram_bytes: int | None = Field(default=None, ge=0)
    expires_at: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "OllamaRuntimeState":
        if self.resident and not self.probe_ok:
            raise ValueError("an unverified Ollama probe cannot claim model residency")
        if self.probe_ok and self.error is not None:
            raise ValueError("a successful Ollama probe cannot carry an error")
        if not self.probe_ok and not self.error:
            raise ValueError("a failed Ollama probe must carry an error")
        return self

    @property
    def system_memory_mib(self) -> int:
        """Conservative resident bytes attributable to system RAM, in MiB."""

        if not self.resident or self.size_bytes is None:
            return 0
        vram = self.size_vram_bytes or 0
        system_bytes = max(0, self.size_bytes - vram)
        return system_bytes // _MIB

    def reusable_memory_credit_mib(self, policy: ResourceSafetyPolicy) -> int:
        """Return the bounded cold-load RAM already represented by resident pages."""

        if not self.probe_ok or not self.resident:
            return 0
        max_credit = max(
            0,
            policy.default_llm_process_memory_mib
            - policy.default_process_memory_mib,
        )
        return min(max_credit, self.system_memory_mib)

    def incremental_process_memory_mib(self, policy: ResourceSafetyPolicy) -> int:
        """RAM that starting this interaction can add to current host pressure.

        A verified resident match needs only ordinary worker/process overhead.
        Every nonresident or failed observation falls back to the full cold local
        LLM budget. This is intentionally fail-closed.
        """

        if self.probe_ok and self.resident:
            return policy.default_process_memory_mib
        return policy.default_llm_process_memory_mib

    @property
    def residency_label(self) -> str:
        if not self.probe_ok:
            return "unverified-cold-fallback"
        return "warm-resident" if self.resident else "cold-nonresident"


class OllamaRuntimeProbe:
    """Read Ollama's running-model endpoint without loading a model."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 2.0,
        client: httpx.Client | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url or configured_ollama_base_url()
        self.model = model or configured_ollama_model()
        _canonical_model_name(self.model)
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            trust_env=False,
        )

    def capture(self) -> OllamaRuntimeState:
        try:
            from persistent_cognition.network_consent import NetworkPurpose, require_destination
            require_destination(str(getattr(self._client, "base_url", self.base_url)), NetworkPurpose.MODEL)
            response = self._client.get("/api/ps")
            response.raise_for_status()
            body = response.json()
            models = body.get("models", [])
            if not isinstance(models, list):
                raise ValueError("Ollama /api/ps response has no model list")
        except Exception as exc:
            return OllamaRuntimeState(
                model=self.model,
                probe_ok=False,
                resident=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        target = _canonical_model_name(self.model)
        for raw in models:
            if not isinstance(raw, dict):
                continue
            names = [raw.get("name"), raw.get("model")]
            if not any(
                isinstance(name, str) and _canonical_model_name(name) == target
                for name in names
            ):
                continue
            return OllamaRuntimeState(
                model=self.model,
                probe_ok=True,
                resident=True,
                reported_name=str(raw.get("name") or raw.get("model") or self.model),
                size_bytes=_optional_nonnegative_int(raw.get("size")),
                size_vram_bytes=_optional_nonnegative_int(raw.get("size_vram")),
                expires_at=(
                    str(raw["expires_at"])
                    if raw.get("expires_at") is not None
                    else None
                ),
            )

        return OllamaRuntimeState(
            model=self.model,
            probe_ok=True,
            resident=False,
        )


class OllamaClaimHostResourceProbe:
    """Reconcile scheduling-time and claim-time Ollama residency safely.

    The task's durable reservation records the incremental RAM required when it
    was admitted. Immediately before each worker launch, this probe rechecks the
    configured model. If the model became warm after a cold reservation, a
    bounded credit prevents double-counting. If a warm model unloaded after a
    small reservation, an uncertainty-adjusted debit forces the claim gate to
    prove enough physical RAM exists for a cold load before launch.
    """

    def __init__(
        self,
        *,
        base_probe: HostResourceProbe | None = None,
        runtime_probe: OllamaRuntimeProbe | None = None,
        policy: ResourceSafetyPolicy | None = None,
        scheduled_memory_mib: int | None = None,
    ) -> None:
        self.base_probe = base_probe or SystemHostResourceProbe()
        self.runtime_probe = runtime_probe or OllamaRuntimeProbe()
        self.policy = policy or ResourceSafetyPolicy()
        self.scheduled_memory_mib = (
            scheduled_memory_mib
            if scheduled_memory_mib is not None
            else self.policy.default_llm_process_memory_mib
        )
        if self.scheduled_memory_mib < 1:
            raise ValueError("scheduled_memory_mib must be positive")
        self.last_physical_metrics: HostResourceMetrics | None = None
        self.last_runtime_state: OllamaRuntimeState | None = None
        self.last_effective_required_memory_mib = self.scheduled_memory_mib
        self.last_memory_credit_mib = 0
        self.last_memory_debit_mib = 0

    def capture(self) -> HostResourceMetrics:
        physical = self.base_probe.capture()
        runtime = self.runtime_probe.capture()
        required = runtime.incremental_process_memory_mib(self.policy)
        delta = self.scheduled_memory_mib - required

        self.last_physical_metrics = physical.model_copy(deep=True)
        self.last_runtime_state = runtime.model_copy(deep=True)
        self.last_effective_required_memory_mib = required
        self.last_memory_credit_mib = max(0, delta)
        self.last_memory_debit_mib = 0

        if delta == 0:
            return physical

        if delta > 0:
            # A cold reservation is larger than the now-warm incremental need.
            # Under-crediting is conservative because uncertainty is applied
            # again by the normal host-safety calculation.
            return physical.model_copy(
                update={
                    "memory_available_mib": min(
                        physical.memory_total_mib,
                        physical.memory_available_mib + delta,
                    )
                },
                deep=True,
            )

        # A task admitted while warm now needs a cold load. The normal claim
        # gate will compare against the smaller durable reservation, so reduce
        # the observed free-RAM input enough that, after uncertainty is applied,
        # the smaller comparison remains at least as strict as the cold need.
        missing = -delta
        retained_percent = 100 - self.policy.uncertainty_headroom_percent
        debit = (
            math.ceil(missing * 100 / retained_percent)
            if retained_percent > 0
            else missing
        )
        self.last_memory_debit_mib = debit
        return physical.model_copy(
            update={
                "memory_available_mib": max(
                    0,
                    physical.memory_available_mib - debit,
                )
            },
            deep=True,
        )


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("Ollama memory sizes must be non-negative")
    return parsed
