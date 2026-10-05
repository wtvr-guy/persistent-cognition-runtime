"""Authoritative host-resource observations for safe JIT Attention admission.

The operating-system probe is intentionally outside the deterministic
scheduler. It produces one immutable, freshness-bounded snapshot; policy then
consumes only that recorded input. Probe failure produces a zero-new-work
envelope instead of falling back to optimistic configured capacity.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, field_validator, model_validator

from prometheist.attention_resources import (
    ExecutionResource,
    ExecutionResourceClass,
    ResourceReservation,
    execution_resource_sort_key,
)


RESOURCE_SAFETY_POLICY_VERSION = "v0.7-resource-observation-v1"
HOST_CPU_RESOURCE_ID = "host-cpu"
HOST_MEMORY_RESOURCE_ID = "host-ram-mib"
LOCAL_LLM_RESOURCE_ID = "local-llm"
_OBSERVATION_NAMESPACE = UUID("64b7f260-e79d-4f81-b732-45ee4e4ae4a8")


class ResourceObservationError(RuntimeError):
    """Raised when the host cannot provide a required safety measurement."""


class CapacityMeasurement(str, Enum):
    HOST = "HOST"
    CONFIGURED = "CONFIGURED"
    FAIL_CLOSED = "FAIL_CLOSED"


class ResourceSafetyPolicy(BaseModel):
    """Conservative, versioned policy that turns measurements into capacity."""

    policy_version: str = Field(
        default=RESOURCE_SAFETY_POLICY_VERSION,
        min_length=1,
    )
    observation_max_age_seconds: int = Field(default=5, ge=1)
    cpu_system_headroom_percent: int = Field(default=10, ge=0, lt=100)
    max_cpu_pressure_percent: int = Field(default=85, ge=1, le=100)
    memory_system_headroom_percent: int = Field(default=10, ge=0, lt=100)
    memory_system_headroom_min_mib: int = Field(default=1024, ge=0)
    uncertainty_headroom_percent: int = Field(default=20, ge=0, lt=100)
    llm_concurrency_limit: int = Field(default=1, ge=1)
    default_process_cpu_units: int = Field(default=1, ge=1)
    default_process_memory_mib: int = Field(default=512, ge=1)
    default_llm_process_memory_mib: int = Field(default=4096, ge=1)

    def cpu_system_headroom_units(self, logical_cpu_count: int) -> int:
        if logical_cpu_count <= 1 or self.cpu_system_headroom_percent == 0:
            return 0
        return max(
            1,
            math.ceil(
                logical_cpu_count * self.cpu_system_headroom_percent / 100
            ),
        )

    def memory_system_headroom_mib(self, total_mib: int) -> int:
        proportional = math.ceil(
            total_mib * self.memory_system_headroom_percent / 100
        )
        return min(
            total_mib,
            max(self.memory_system_headroom_min_mib, proportional),
        )


class HostResourceMetrics(BaseModel):
    """One bounded CPU/RAM sample, normalized for deterministic policy."""

    platform: str = Field(min_length=1)
    logical_cpu_count: int = Field(ge=1)
    cpu_utilization_percent: int = Field(ge=0, le=100)
    load_1m: float | None = Field(default=None, ge=0)
    memory_total_mib: int = Field(ge=1)
    memory_available_mib: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_available_memory(self) -> "HostResourceMetrics":
        if self.memory_available_mib > self.memory_total_mib:
            raise ValueError("available memory must not exceed total memory")
        return self


class ObservedResourceCapacity(BaseModel):
    """The complete per-pool envelope consumed by one scheduling epoch."""

    resource_id: str = Field(min_length=1)
    resource_class: ExecutionResourceClass
    measurement: CapacityMeasurement
    configured_capacity: int = Field(ge=1)
    configured_headroom_units: int = Field(ge=0)
    committed_units: int = Field(ge=0)
    observed_available_units: int = Field(ge=0)
    uncertainty_headroom_units: int = Field(ge=0)
    admission_capacity: int = Field(ge=0)

    @field_validator("resource_id")
    @classmethod
    def normalize_resource_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("resource_id must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_capacity_envelope(self) -> "ObservedResourceCapacity":
        if self.configured_headroom_units > self.configured_capacity:
            raise ValueError("configured headroom cannot exceed capacity")
        if self.committed_units > self.admission_capacity:
            raise ValueError("observation cannot invalidate committed reservations")
        if self.admission_capacity > (
            self.configured_capacity - self.configured_headroom_units
        ):
            raise ValueError("observation cannot exceed configured safe capacity")
        return self

    @property
    def available_for_new_work(self) -> int:
        return self.admission_capacity - self.committed_units


class ResourceObservationSnapshot(BaseModel):
    """Immutable, replayable host-pressure input to one scheduler cycle."""

    observation_id: UUID
    scheduler_cycle: int = Field(ge=1)
    captured_at: datetime
    valid_until: datetime
    safety_policy_version: str = Field(min_length=1)
    policy: ResourceSafetyPolicy
    host_id: str = Field(default="local", min_length=1)
    healthy: bool
    probe_errors: list[str] = Field(default_factory=list)
    metrics: HostResourceMetrics | None = None
    capacities: list[ObservedResourceCapacity] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ResourceObservationSnapshot":
        captured = _as_utc(self.captured_at)
        valid_until = _as_utc(self.valid_until)
        if valid_until <= captured:
            raise ValueError("resource observation must have a positive lifetime")
        self.captured_at = captured
        self.valid_until = valid_until
        if self.healthy != (self.metrics is not None and not self.probe_errors):
            raise ValueError("observation health must match metrics and probe errors")
        if self.safety_policy_version != self.policy.policy_version:
            raise ValueError("observation safety policy version does not match policy")
        resource_ids = [capacity.resource_id for capacity in self.capacities]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource observation contains duplicate resource ids")
        self.capacities = sorted(
            self.capacities,
            key=lambda value: (
                list(ExecutionResourceClass).index(value.resource_class),
                value.resource_id,
            ),
        )
        expected = deterministic_resource_observation_id(self)
        if self.observation_id != expected:
            raise ValueError("resource observation id is not deterministic")
        return self

    def assert_fresh(self, *, at: datetime) -> None:
        evaluated_at = _as_utc(at)
        if evaluated_at < self.captured_at or evaluated_at > self.valid_until:
            raise ResourceObservationError("resource observation is stale")

    def capacity_by_resource_id(self) -> dict[str, ObservedResourceCapacity]:
        return {
            capacity.resource_id: capacity.model_copy(deep=True)
            for capacity in self.capacities
        }


class HostResourceProbe(Protocol):
    def capture(self) -> HostResourceMetrics:
        """Capture one current host-pressure sample."""
        ...


class ReservationCreditHostResourceProbe:
    """Credit unused task-level RAM reservation for one smaller worker stage.

    The Attention Fabric currently reserves process RAM at task-assignment
    granularity. A multi-stage task may therefore reserve a cold LLM budget even
    while executing a deterministic stage. This probe preserves the durable task
    reservation while adjusting only the fresh host observation used for one
    claim. The credit is bounded to the difference between the durable
    reservation and this stage's declared process requirement.

    Because normal uncertainty/headroom policy is applied after this adjusted
    observation, the credit remains conservative. This class cannot represent a
    stage whose actual requirement exceeds the durable reservation.
    """

    def __init__(
        self,
        *,
        base_probe: HostResourceProbe | None = None,
        scheduled_memory_mib: int,
        required_memory_mib: int,
    ) -> None:
        if scheduled_memory_mib < 1:
            raise ValueError("scheduled_memory_mib must be positive")
        if required_memory_mib < 1:
            raise ValueError("required_memory_mib must be positive")
        if required_memory_mib > scheduled_memory_mib:
            raise ValueError(
                "required_memory_mib cannot exceed scheduled_memory_mib"
            )
        self.base_probe = base_probe or SystemHostResourceProbe()
        self.scheduled_memory_mib = scheduled_memory_mib
        self.required_memory_mib = required_memory_mib
        self.last_physical_metrics: HostResourceMetrics | None = None
        self.last_memory_credit_mib = (
            scheduled_memory_mib - required_memory_mib
        )

    def capture(self) -> HostResourceMetrics:
        physical = self.base_probe.capture()
        self.last_physical_metrics = physical.model_copy(deep=True)
        return physical.model_copy(
            update={
                "memory_available_mib": min(
                    physical.memory_total_mib,
                    physical.memory_available_mib + self.last_memory_credit_mib,
                )
            },
            deep=True,
        )


class SystemHostResourceProbe:
    """Small standard-library CPU/RAM probe for Linux, Windows, and macOS."""

    def __init__(
        self,
        *,
        sample_interval_seconds: float = 0.1,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.sample_interval_seconds = sample_interval_seconds
        self._sleep = sleep

    def capture(self) -> HostResourceMetrics:
        system = platform.system()
        logical_cpu_count = os.cpu_count() or 1
        if system == "Linux":
            cpu_percent = self._sample_linux_cpu_percent()
            total_mib, available_mib = self._linux_memory_mib()
        elif system == "Windows":
            cpu_percent = self._sample_windows_cpu_percent()
            total_mib, available_mib = self._windows_memory_mib()
        elif system == "Darwin":
            cpu_percent = self._darwin_cpu_pressure_percent(logical_cpu_count)
            total_mib, available_mib = self._darwin_memory_mib()
        else:
            raise ResourceObservationError(
                f"Unsupported host platform for resource discovery: {system!r}"
            )

        # getloadavg is absent on Windows, so probe for it rather than relying on
        # an AttributeError from a platform-specific attribute.
        getloadavg = getattr(os, "getloadavg", None)
        try:
            load_1m = round(float(getloadavg()[0]), 2) if getloadavg is not None else None
        except OSError:
            load_1m = None
        return HostResourceMetrics(
            platform=system,
            logical_cpu_count=logical_cpu_count,
            cpu_utilization_percent=cpu_percent,
            load_1m=load_1m,
            memory_total_mib=total_mib,
            memory_available_mib=available_mib,
        )

    def _sample_linux_cpu_percent(self) -> int:
        first = self._linux_cpu_times()
        self._sleep(self.sample_interval_seconds)
        second = self._linux_cpu_times()
        return _cpu_percent_from_counters(first, second)

    @staticmethod
    def _linux_cpu_times() -> tuple[int, int]:
        try:
            line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
            fields = [int(value) for value in line.split()[1:]]
        except (OSError, IndexError, ValueError) as exc:
            raise ResourceObservationError("Unable to read Linux CPU counters") from exc
        if len(fields) < 4:
            raise ResourceObservationError("Linux CPU counters are incomplete")
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return sum(fields), idle

    @staticmethod
    def _linux_memory_mib() -> tuple[int, int]:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
            total_kib = values["MemTotal"]
            available_kib = values["MemAvailable"]
        except (OSError, KeyError, ValueError) as exc:
            raise ResourceObservationError("Unable to read Linux memory counters") from exc
        return max(1, total_kib // 1024), max(0, available_kib // 1024)

    def _sample_windows_cpu_percent(self) -> int:
        first = self._windows_cpu_times()
        self._sleep(self.sample_interval_seconds)
        second = self._windows_cpu_times()
        return _cpu_percent_from_counters(first, second)

    @staticmethod
    def _windows_cpu_times() -> tuple[int, int]:
        class FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

        idle = FileTime()
        kernel = FileTime()
        user = FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise ResourceObservationError("Unable to read Windows CPU counters")

        def value(raw: FileTime) -> int:
            return (raw.high << 32) | raw.low

        # Windows kernel time includes idle time.
        return value(kernel) + value(user), value(idle)

    @staticmethod
    def _windows_memory_mib() -> tuple[int, int]:
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("memory_load", ctypes.c_uint32),
                ("total_physical", ctypes.c_uint64),
                ("available_physical", ctypes.c_uint64),
                ("total_page_file", ctypes.c_uint64),
                ("available_page_file", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("available_virtual", ctypes.c_uint64),
                ("available_extended_virtual", ctypes.c_uint64),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
            ctypes.byref(status)
        ):
            raise ResourceObservationError("Unable to read Windows memory counters")
        mib = 1024 * 1024
        return (
            max(1, status.total_physical // mib),
            max(0, status.available_physical // mib),
        )

    @staticmethod
    def _darwin_cpu_pressure_percent(logical_cpu_count: int) -> int:
        getloadavg = getattr(os, "getloadavg", None)
        if getloadavg is None:
            raise ResourceObservationError("Unable to read macOS CPU load")
        try:
            load = float(getloadavg()[0])
        except OSError as exc:
            raise ResourceObservationError("Unable to read macOS CPU load") from exc
        return min(100, max(0, round(load * 100 / logical_cpu_count)))

    @staticmethod
    def _darwin_memory_mib() -> tuple[int, int]:
        try:
            total_result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            vm_result = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            total_bytes = int(total_result.stdout.strip())
            page_match = re.search(r"page size of (\d+) bytes", vm_result.stdout)
            if page_match is None:
                raise ValueError("vm_stat page size is missing")
            page_size = int(page_match.group(1))
            pages: dict[str, int] = {}
            for line in vm_result.stdout.splitlines()[1:]:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                pages[key.strip()] = int(raw.strip().rstrip("."))
            available_pages = sum(
                pages.get(key, 0)
                for key in (
                    "Pages free",
                    "Pages inactive",
                    "Pages speculative",
                    "Pages purgeable",
                )
            )
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
        ) as exc:
            raise ResourceObservationError("Unable to read macOS memory counters") from exc
        mib = 1024 * 1024
        return (
            max(1, total_bytes // mib),
            max(0, page_size * available_pages // mib),
        )


def deterministic_resource_observation_id(
    observation: ResourceObservationSnapshot,
) -> UUID:
    payload = observation.model_dump(
        mode="json",
        exclude={"observation_id"},
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return uuid5(_OBSERVATION_NAMESPACE, canonical)


def discover_local_execution_resources(
    metrics: HostResourceMetrics | None,
    *,
    policy: ResourceSafetyPolicy,
) -> list[ExecutionResource]:
    """Return the conservative default local CPU/RAM/LLM resource pools."""

    if metrics is None:
        cpu_capacity = 1
        cpu_headroom = 1
        memory_capacity = 1
        memory_headroom = 1
        platform_name = "unknown"
    else:
        cpu_capacity = metrics.logical_cpu_count
        cpu_headroom = policy.cpu_system_headroom_units(cpu_capacity)
        memory_capacity = metrics.memory_total_mib
        memory_headroom = policy.memory_system_headroom_mib(memory_capacity)
        platform_name = metrics.platform

    return sorted(
        [
            ExecutionResource(
                resource_id=HOST_CPU_RESOURCE_ID,
                resource_class=ExecutionResourceClass.CPU_GENERAL,
                capacity=cpu_capacity,
                system_headroom=cpu_headroom,
                metadata={"discovered": True, "host": platform_name, "unit": "slot"},
            ),
            ExecutionResource(
                resource_id=HOST_MEMORY_RESOURCE_ID,
                resource_class=ExecutionResourceClass.MEMORY_RAM,
                capacity=memory_capacity,
                system_headroom=memory_headroom,
                metadata={"discovered": True, "host": platform_name, "unit": "MiB"},
            ),
            ExecutionResource(
                resource_id=LOCAL_LLM_RESOURCE_ID,
                resource_class=ExecutionResourceClass.LLM_INFERENCE,
                capacity=policy.llm_concurrency_limit,
                metadata={
                    "backend": "local",
                    "default_concurrency": policy.llm_concurrency_limit,
                    "unit": "inference",
                },
            ),
        ],
        key=execution_resource_sort_key,
    )


def build_resource_observation(
    *,
    scheduler_cycle: int,
    captured_at: datetime,
    resources: list[ExecutionResource],
    reservations: list[ResourceReservation],
    policy: ResourceSafetyPolicy,
    metrics: HostResourceMetrics | None,
    probe_errors: list[str] | None = None,
    host_id: str = "local",
) -> ResourceObservationSnapshot:
    """Convert one host sample and committed state into a total safe envelope."""

    captured = _as_utc(captured_at)
    errors = [value.strip() for value in (probe_errors or []) if value.strip()]
    healthy = metrics is not None and not errors
    committed_by_resource: dict[str, int] = {}
    for reservation in reservations:
        committed_by_resource[reservation.resource_id] = (
            committed_by_resource.get(reservation.resource_id, 0)
            + reservation.units
        )

    total_llm_committed = sum(
        reservation.units
        for reservation in reservations
        if reservation.resource_class is ExecutionResourceClass.LLM_INFERENCE
    )
    if total_llm_committed > policy.llm_concurrency_limit:
        raise ResourceObservationError(
            "committed LLM reservations exceed the configured safety limit"
        )

    capacities: list[ObservedResourceCapacity] = []
    llm_new_budget = policy.llm_concurrency_limit - total_llm_committed
    for resource in sorted(resources, key=execution_resource_sort_key):
        committed = committed_by_resource.get(resource.resource_id, 0)
        if not resource.enabled:
            admission_capacity = committed
            observed_available = 0
            uncertainty = 0
            measurement = CapacityMeasurement.CONFIGURED
        elif resource.resource_id == HOST_CPU_RESOURCE_ID:
            if healthy and metrics is not None:
                pressure = metrics.cpu_utilization_percent
                if metrics.load_1m is not None:
                    pressure = max(
                        pressure,
                        min(
                            100,
                            round(
                                metrics.load_1m
                                * 100
                                / metrics.logical_cpu_count
                            ),
                        ),
                    )
                idle_units = math.floor(
                    metrics.logical_cpu_count * (100 - pressure) / 100
                )
                if pressure >= policy.max_cpu_pressure_percent:
                    idle_units = 0
                raw_new = max(0, idle_units - resource.system_headroom)
                uncertainty = math.ceil(
                    raw_new * policy.uncertainty_headroom_percent / 100
                )
                additional = max(0, raw_new - uncertainty)
                observed_available = idle_units
                admission_capacity = min(
                    resource.admissible_capacity,
                    committed + additional,
                )
                measurement = CapacityMeasurement.HOST
            else:
                observed_available = 0
                uncertainty = 0
                admission_capacity = committed
                measurement = CapacityMeasurement.FAIL_CLOSED
        elif resource.resource_id == HOST_MEMORY_RESOURCE_ID:
            if healthy and metrics is not None:
                observed_available = metrics.memory_available_mib
                raw_new = max(
                    0,
                    observed_available - resource.system_headroom,
                )
                uncertainty = math.ceil(
                    raw_new * policy.uncertainty_headroom_percent / 100
                )
                additional = max(0, raw_new - uncertainty)
                admission_capacity = min(
                    resource.admissible_capacity,
                    committed + additional,
                )
                measurement = CapacityMeasurement.HOST
            else:
                observed_available = 0
                uncertainty = 0
                admission_capacity = committed
                measurement = CapacityMeasurement.FAIL_CLOSED
        elif resource.resource_class in {
            ExecutionResourceClass.CPU_GENERAL,
            ExecutionResourceClass.MEMORY_RAM,
        }:
            # Only the discovered host pools may turn one host-wide CPU/RAM
            # budget into new capacity. This prevents duplicate configured
            # pools from multiplying a single machine's measured capacity.
            observed_available = 0
            uncertainty = 0
            admission_capacity = committed
            measurement = CapacityMeasurement.FAIL_CLOSED
        elif resource.resource_class is ExecutionResourceClass.LLM_INFERENCE:
            observed_available = max(0, resource.admissible_capacity - committed)
            uncertainty = 0
            if resource.resource_id == LOCAL_LLM_RESOURCE_ID:
                additional = min(observed_available, llm_new_budget)
                llm_new_budget -= additional
            else:
                additional = 0
            admission_capacity = committed + additional
            measurement = CapacityMeasurement.CONFIGURED
        else:
            observed_available = max(0, resource.admissible_capacity - committed)
            uncertainty = 0
            admission_capacity = resource.admissible_capacity
            measurement = CapacityMeasurement.CONFIGURED

        capacities.append(
            ObservedResourceCapacity(
                resource_id=resource.resource_id,
                resource_class=resource.resource_class,
                measurement=measurement,
                configured_capacity=resource.capacity,
                configured_headroom_units=resource.system_headroom,
                committed_units=committed,
                observed_available_units=observed_available,
                uncertainty_headroom_units=uncertainty,
                admission_capacity=admission_capacity,
            )
        )

    provisional = ResourceObservationSnapshot.model_construct(
        observation_id=UUID(int=0),
        scheduler_cycle=scheduler_cycle,
        captured_at=captured,
        valid_until=captured
        + timedelta(seconds=policy.observation_max_age_seconds),
        safety_policy_version=policy.policy_version,
        policy=policy.model_copy(deep=True),
        host_id=host_id,
        healthy=healthy,
        probe_errors=errors,
        metrics=metrics,
        capacities=capacities,
    )
    observation_id = deterministic_resource_observation_id(provisional)
    return ResourceObservationSnapshot(
        **provisional.model_dump(exclude={"observation_id"}),
        observation_id=observation_id,
    )


class LocalResourceAdmissionController:
    """The required local-host gateway into deterministic epoch planning."""

    def __init__(
        self,
        scheduler: object,
        *,
        probe: HostResourceProbe | None = None,
        policy: ResourceSafetyPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        host_id: str = "local",
    ) -> None:
        self.scheduler = scheduler
        self.probe = probe or SystemHostResourceProbe()
        self.policy = policy or ResourceSafetyPolicy()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.host_id = host_id
        getattr(self.scheduler, "enable_resource_safety")(
            policy_version=self.policy.policy_version
        )

    def refresh(self) -> ResourceObservationSnapshot:
        """Discover, measure, and attach a fresh snapshot for the next cycle."""

        captured_at = _as_utc(self.clock())
        getattr(self.scheduler, "assess_unestimated_processes")(
            policy=self.policy
        )
        errors: list[str] = []
        try:
            metrics = self.probe.capture()
        except Exception as exc:  # the safety response to any probe failure is zero work
            metrics = None
            errors.append(f"{type(exc).__name__}: {exc}")

        discovered = discover_local_execution_resources(
            metrics,
            policy=self.policy,
        )
        if metrics is None:
            existing_resources = {
                resource.resource_id: resource
                for resource in getattr(self.scheduler, "execution_resources")(
                    enabled_only=False
                )
            }
            discovered = [
                existing_resources.get(resource.resource_id, resource)
                if resource.resource_id
                in {HOST_CPU_RESOURCE_ID, HOST_MEMORY_RESOURCE_ID}
                else resource
                for resource in discovered
            ]
        configure = getattr(self.scheduler, "configure_execution_resource")
        for resource in discovered:
            try:
                configure(resource)
            except ValueError:
                # A hot-plug/downsize cannot erase already committed capacity.
                # Keep the durable definition; the observation still exposes no
                # new CPU/RAM work beyond what the host reports safe.
                existing = getattr(self.scheduler, "resources").get(
                    resource.resource_id
                )
                if existing is None:
                    raise

        observation = build_resource_observation(
            scheduler_cycle=getattr(self.scheduler, "cycle") + 1,
            captured_at=captured_at,
            resources=getattr(self.scheduler, "execution_resources")(
                enabled_only=False
            ),
            reservations=getattr(self.scheduler, "resource_reservations")(),
            policy=self.policy,
            metrics=metrics,
            probe_errors=errors,
            host_id=self.host_id,
        )
        attach = getattr(self.scheduler, "attach_resource_observation")
        attach(observation, evaluated_at=captured_at)
        return observation.model_copy(deep=True)

    def plan_scheduling_epoch(self):
        """Refresh live pressure immediately before producing READY work."""

        self.refresh()
        return getattr(self.scheduler, "plan_scheduling_epoch")()


def _cpu_percent_from_counters(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    total_delta = second[0] - first[0]
    idle_delta = second[1] - first[1]
    if total_delta <= 0 or idle_delta < 0:
        raise ResourceObservationError("CPU counters did not advance monotonically")
    busy_delta = max(0, total_delta - idle_delta)
    return min(100, max(0, round(busy_delta * 100 / total_delta)))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("resource observation timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)
