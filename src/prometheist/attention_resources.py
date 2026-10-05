"""Deterministic execution-resource primitives for the JIT Attention Fabric."""
from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, field_validator, model_validator


class ExecutionResourceClass(str, Enum):
    """Bounded execution-capacity classes exposed to JIT Attention."""

    CPU_GENERAL = "CPU_GENERAL"
    MEMORY_RAM = "MEMORY_RAM"
    LLM_INFERENCE = "LLM_INFERENCE"
    DATABASE = "DATABASE"
    FILESYSTEM_IO = "FILESYSTEM_IO"
    NETWORK_IO = "NETWORK_IO"


_RESOURCE_CLASS_ORDER: dict[ExecutionResourceClass, int] = {
    resource_class: index
    for index, resource_class in enumerate(ExecutionResourceClass)
}

RESOURCE_ADMISSION_POLICY_VERSION = "v0.7-c-greedy-v1"


class ExecutionResource(BaseModel):
    """Durable definition of one execution-resource pool.

    ``capacity`` and ``system_headroom`` are explicit policy/configuration
    values. They are deliberately independent of host CPU/thread count so a
    resource such as local LLM inference can remain capacity 1 on otherwise
    highly parallel hardware.
    """

    resource_id: str = Field(min_length=1)
    resource_class: ExecutionResourceClass
    capacity: int = Field(ge=1)
    system_headroom: int = Field(default=0, ge=0)
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resource_id")
    @classmethod
    def normalize_resource_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("resource_id must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_headroom(self) -> "ExecutionResource":
        if self.system_headroom > self.capacity:
            raise ValueError("system_headroom must not exceed capacity")
        return self

    @property
    def admissible_capacity(self) -> int:
        """Capacity ordinary Prometheist work may reserve from this pool."""

        if not self.enabled:
            return 0
        return self.capacity - self.system_headroom


class ResourceRequirement(BaseModel):
    """Quantitative capacity required from one fungible resource class."""

    resource_class: ExecutionResourceClass
    units: int = Field(ge=1)


class ResourceEstimateSource(str, Enum):
    """Provenance for a process peak-resource estimate."""

    DECLARED = "DECLARED"
    PROFILED = "PROFILED"
    HISTORICAL_PEAK = "HISTORICAL_PEAK"
    CONSERVATIVE_DEFAULT = "CONSERVATIVE_DEFAULT"


class ProcessResourceEstimate(BaseModel):
    """Pre-launch peak estimate for one bounded worker/process step.

    CPU units are scheduler slots rather than logical threads. RAM is measured
    in MiB. The host-safety path requires this structure instead of silently
    guessing what an unknown process might consume.
    """

    cpu_units: int = Field(ge=1)
    memory_mib: int = Field(ge=1)
    llm_slots: int = Field(default=0, ge=0, le=1)
    source: ResourceEstimateSource
    basis: str = Field(min_length=1)

    @field_validator("basis")
    @classmethod
    def normalize_basis(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("basis must not be empty")
        return normalized

    def resource_requirements(self) -> list[ResourceRequirement]:
        """Return the canonical quantitative requirements for admission."""

        requirements = [
            ResourceRequirement(
                resource_class=ExecutionResourceClass.CPU_GENERAL,
                units=self.cpu_units,
            ),
            ResourceRequirement(
                resource_class=ExecutionResourceClass.MEMORY_RAM,
                units=self.memory_mib,
            ),
        ]
        if self.llm_slots:
            requirements.append(
                ResourceRequirement(
                    resource_class=ExecutionResourceClass.LLM_INFERENCE,
                    units=self.llm_slots,
                )
            )
        requirements.sort(key=resource_requirement_sort_key)
        return requirements


class ResourceReservation(BaseModel):
    """Durable allocation of one task's requirement to one resource pool."""

    reservation_id: UUID
    task_id: UUID
    resource_id: str = Field(min_length=1)
    resource_class: ExecutionResourceClass
    units: int = Field(ge=1)

    @field_validator("resource_id")
    @classmethod
    def normalize_resource_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("resource_id must not be empty")
        return normalized


class ResourceAdmissionPlan(BaseModel):
    """Complete deterministic result of one admission reconciliation."""

    policy_version: str = Field(
        default=RESOURCE_ADMISSION_POLICY_VERSION,
        min_length=1,
    )
    admitted_task_ids: list[UUID] = Field(default_factory=list)
    unadmitted_task_ids: list[UUID] = Field(default_factory=list)
    reservations: list[ResourceReservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity_sets(self) -> "ResourceAdmissionPlan":
        admitted = set(self.admitted_task_ids)
        unadmitted = set(self.unadmitted_task_ids)
        if len(admitted) != len(self.admitted_task_ids):
            raise ValueError("admitted_task_ids must not contain duplicates")
        if len(unadmitted) != len(self.unadmitted_task_ids):
            raise ValueError("unadmitted_task_ids must not contain duplicates")
        if admitted.intersection(unadmitted):
            raise ValueError("admitted and unadmitted task ids must be disjoint")

        reservation_ids = [item.reservation_id for item in self.reservations]
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError("reservations must not contain duplicate ids")
        reservation_keys = [
            (item.task_id, item.resource_id) for item in self.reservations
        ]
        if len(reservation_keys) != len(set(reservation_keys)):
            raise ValueError("a task may reserve a resource pool only once")
        if any(item.task_id not in admitted for item in self.reservations):
            raise ValueError("every reservation must belong to an admitted task")
        return self


def execution_resource_class_sort_key(resource_class: ExecutionResourceClass) -> int:
    """Return the stable declaration order for one resource class."""

    return _RESOURCE_CLASS_ORDER[resource_class]


def execution_resource_sort_key(
    resource: ExecutionResource,
) -> tuple[int, str]:
    """Return the total deterministic ordering key for resource definitions."""

    return (_RESOURCE_CLASS_ORDER[resource.resource_class], resource.resource_id)


def resource_requirement_sort_key(requirement: ResourceRequirement) -> int:
    return execution_resource_class_sort_key(requirement.resource_class)


def deterministic_reservation_id(task_id: UUID, resource_id: str) -> UUID:
    """Return the stable identity for one task/resource reservation pair."""

    normalized = resource_id.strip()
    if not normalized:
        raise ValueError("resource_id must not be empty")
    return uuid5(task_id, f"prometheist-resource-reservation:{normalized}")


def resource_reservation_sort_key(
    reservation: ResourceReservation,
) -> tuple[int, str, str, str]:
    """Return a total deterministic order for durable reservations."""

    return (
        execution_resource_class_sort_key(reservation.resource_class),
        reservation.resource_id,
        reservation.task_id.hex,
        reservation.reservation_id.hex,
    )
