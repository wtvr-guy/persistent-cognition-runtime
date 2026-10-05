"""Deterministic, task-neutral discovery and planning of executable work.

The pre-cognitive model receives one bounded application-owned catalog and returns
only integer indices. Basic JIT-memory activation is cognitive substrate and is
therefore never selectable work. Prometheist owns dependency closure, ordering,
resource admission, and execution through Attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Literal
from uuid import UUID, uuid5

import psycopg
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prometheist import event_store
from prometheist.attention_ordering import (
    MAX_ATTENTION_WORK_PRIORITY,
    AttentionWorkItem,
    deterministic_dependency_order,
)
from prometheist.models import EventType


CAPABILITY_REGISTRY_VERSION = "v0.7-capability-registry-v9"
SOURCE = "capability_registry"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class CapabilityKind(str, Enum):
    SERVICE = "SERVICE"
    TOOL = "TOOL"
    MODEL = "MODEL"
    WORKFLOW = "WORKFLOW"


class CapabilityDescriptor(BaseModel):
    """Small public descriptor safe to reveal to a disposable routing model."""

    capability_id: str = Field(min_length=1)
    kind: CapabilityKind
    description: str = Field(min_length=1)

    @field_validator("capability_id", "description")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("capability descriptor text must not be empty")
        return normalized


class CapabilityNeed(BaseModel):
    """Functionality requested by deterministic/generic discovery callers."""

    query_text: str = Field(min_length=1)
    supplemental_query_texts: list[str] = Field(default_factory=list, max_length=3)
    kinds: list[CapabilityKind] | None = None
    exclude_capability_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=3, ge=1, le=10)

    @field_validator("query_text")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("capability query_text must not be empty")
        return normalized

    @field_validator("supplemental_query_texts", "exclude_capability_ids")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("capability need lists must not contain empty values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("capability need lists must not contain duplicates")
        return normalized


class CapabilityMatch(BaseModel):
    descriptor: CapabilityDescriptor
    score: float = Field(gt=0)


class CapabilityPacket(BaseModel):
    registry_version: str = CAPABILITY_REGISTRY_VERSION
    capability_request_id: UUID
    requester_task_id: UUID
    requester_step_id: UUID
    need: CapabilityNeed
    matches: list[CapabilityMatch]
    selected_query_role: Literal["canonical", "supplemental"] | None = None
    selected_query_text: str | None = None

    @model_validator(mode="after")
    def validate_packet(self) -> "CapabilityPacket":
        if self.registry_version != CAPABILITY_REGISTRY_VERSION:
            raise ValueError("capability registry version is unsupported")
        expected = deterministic_capability_request_id(self.requester_step_id)
        if self.capability_request_id != expected:
            raise ValueError("capability request id is not deterministic")
        if bool(self.matches) != (self.selected_query_role is not None):
            raise ValueError("capability selection metadata does not match the result")
        if (self.selected_query_role is None) != (self.selected_query_text is None):
            raise ValueError("capability selection role and text must appear together")
        if len(self.matches) > self.need.limit:
            raise ValueError("capability packet exceeds the requested bound")
        return self


class CapabilityExecutionPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capability_id: str = Field(min_length=1)
    execution_priority: int = Field(ge=0, le=MAX_ATTENTION_WORK_PRIORITY)
    depends_on_capability_ids: list[str] = Field(default_factory=list)


class CapabilityExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registry_version: str = CAPABILITY_REGISTRY_VERSION
    requested_catalog_indices: list[int] = Field(default_factory=list)
    catalog_capability_ids: list[str] = Field(default_factory=list)
    items: list[CapabilityExecutionPlanItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self) -> "CapabilityExecutionPlan":
        if self.registry_version != CAPABILITY_REGISTRY_VERSION:
            raise ValueError("capability execution plan uses an unsupported registry version")
        if len(self.requested_catalog_indices) != len(set(self.requested_catalog_indices)):
            raise ValueError("requested capability indices must not contain duplicates")
        if len(self.catalog_capability_ids) != len(set(self.catalog_capability_ids)):
            raise ValueError("capability catalog must not contain duplicate ids")
        if any(
            index < 0 or index >= len(self.catalog_capability_ids)
            for index in self.requested_catalog_indices
        ):
            raise ValueError("requested capability index is outside the persisted catalog")
        ids = [item.capability_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("capability execution plan must not contain duplicate capabilities")
        positions = {capability_id: index for index, capability_id in enumerate(ids)}
        for item in self.items:
            for dependency in item.depends_on_capability_ids:
                if dependency not in positions:
                    raise ValueError("capability execution plan omits a declared dependency")
                if positions[dependency] >= positions[item.capability_id]:
                    raise ValueError("capability execution plan violates dependency order")
        return self


@dataclass(frozen=True)
class RegisteredCapability:
    """Private application-owned registration and executable binding."""

    descriptor: CapabilityDescriptor
    routing_terms: tuple[str, ...]
    executor: str
    selectable_as_external_work: bool = True
    execution_priority: int = 100
    depends_on_capability_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_executor = self.executor.strip()
        normalized_terms = tuple(term.strip() for term in self.routing_terms)
        normalized_dependencies = tuple(
            dependency.strip() for dependency in self.depends_on_capability_ids
        )
        if not normalized_executor:
            raise ValueError("capability executor must not be empty")
        if not normalized_terms or any(not term for term in normalized_terms):
            raise ValueError("capability routing terms must not be empty")
        if len(normalized_terms) != len(set(normalized_terms)):
            raise ValueError("capability routing terms must not contain duplicates")
        if not 0 <= self.execution_priority <= MAX_ATTENTION_WORK_PRIORITY:
            raise ValueError("capability execution_priority is outside the supported range")
        if any(not value for value in normalized_dependencies):
            raise ValueError("capability dependencies must not be empty")
        if len(normalized_dependencies) != len(set(normalized_dependencies)):
            raise ValueError("capability dependencies must not contain duplicates")
        capability_id = self.descriptor.capability_id
        if capability_id in normalized_dependencies:
            raise ValueError("capability must not depend on itself")
        object.__setattr__(self, "executor", normalized_executor)
        object.__setattr__(self, "routing_terms", normalized_terms)
        object.__setattr__(self, "depends_on_capability_ids", normalized_dependencies)


class CapabilityRegistry:
    def __init__(self, registrations: tuple[RegisteredCapability, ...] = ()) -> None:
        self._registrations: dict[str, RegisteredCapability] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: RegisteredCapability) -> None:
        capability_id = registration.descriptor.capability_id
        if capability_id in self._registrations:
            raise ValueError(f"capability id already registered: {capability_id}")
        self._registrations[capability_id] = registration

    def unregister(self, capability_id: str) -> RegisteredCapability:
        try:
            return self._registrations.pop(capability_id)
        except KeyError as exc:
            raise KeyError(f"Unknown capability_id: {capability_id}") from exc

    def get(self, capability_id: str) -> RegisteredCapability:
        try:
            return self._registrations[capability_id]
        except KeyError as exc:
            raise KeyError(f"Unknown capability_id: {capability_id}") from exc

    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            self._registrations[key].descriptor.model_copy(deep=True)
            for key in sorted(self._registrations)
        )

    def capability_catalog(self) -> tuple[CapabilityDescriptor, ...]:
        """Return the deterministic catalog for pre-cognitive work selection."""

        visible_ids = {
            capability_id
            for capability_id, registration in self._registrations.items()
            if registration.selectable_as_external_work
        }
        return tuple(
            self.get(capability_id).descriptor.model_copy(deep=True)
            for capability_id in sorted(visible_ids)
        )

    def resolve_catalog_indices(
        self,
        catalog: tuple[CapabilityDescriptor, ...],
        indices: list[int],
    ) -> tuple[RegisteredCapability, ...]:
        if len(indices) != len(set(indices)):
            raise ValueError("capability indices must not contain duplicates")
        if any(index < 0 or index >= len(catalog) for index in indices):
            raise ValueError("capability index is outside the supplied catalog")
        return tuple(self.get(catalog[index].capability_id) for index in indices)

    def plan_execution(
        self,
        catalog: tuple[CapabilityDescriptor, ...],
        indices: list[int],
    ) -> CapabilityExecutionPlan:
        """Expand dependencies and delegate deterministic ordering to Attention."""

        selected = self.resolve_catalog_indices(catalog, indices)
        required_ids = {registration.descriptor.capability_id for registration in selected}
        pending = list(sorted(required_ids))
        while pending:
            capability_id = pending.pop()
            registration = self.get(capability_id)
            for dependency in registration.depends_on_capability_ids:
                self.get(dependency)
                if dependency not in required_ids:
                    required_ids.add(dependency)
                    pending.append(dependency)

        ordered_ids = deterministic_dependency_order(
            [
                AttentionWorkItem(
                    item_id=capability_id,
                    priority=self.get(capability_id).execution_priority,
                    dependency_ids=list(self.get(capability_id).depends_on_capability_ids),
                )
                for capability_id in sorted(required_ids)
            ]
        )
        return CapabilityExecutionPlan(
            requested_catalog_indices=list(indices),
            catalog_capability_ids=[descriptor.capability_id for descriptor in catalog],
            items=[
                CapabilityExecutionPlanItem(
                    capability_id=capability_id,
                    execution_priority=self.get(capability_id).execution_priority,
                    depends_on_capability_ids=list(self.get(capability_id).depends_on_capability_ids),
                )
                for capability_id in ordered_ids
            ],
        )

    def discover(self, need: CapabilityNeed) -> list[CapabilityMatch]:
        matches, _role, _text = self.discover_with_trace(need)
        return matches

    def discover_with_trace(
        self,
        need: CapabilityNeed,
    ) -> tuple[list[CapabilityMatch], Literal["canonical", "supplemental"] | None, str | None]:
        matches = self._discover_query(need, need.query_text)
        if matches:
            return matches, "canonical", need.query_text
        for supplemental in need.supplemental_query_texts:
            matches = self._discover_query(need, supplemental)
            if matches:
                return matches, "supplemental", supplemental
        return [], None, None

    def _discover_query(self, need: CapabilityNeed, query_text: str) -> list[CapabilityMatch]:
        allowed_kinds = set(need.kinds) if need.kinds else None
        excluded = set(need.exclude_capability_ids)
        query_tokens = _tokens(query_text)
        query_token_set = set(query_tokens)
        matches: list[CapabilityMatch] = []
        for capability_id in sorted(self._registrations):
            registration = self._registrations[capability_id]
            descriptor = registration.descriptor
            if capability_id in excluded:
                continue
            if allowed_kinds is not None and descriptor.kind not in allowed_kinds:
                continue
            score = 0.0
            id_tokens = _tokens(capability_id.replace("_", " "))
            if _contains_phrase(query_tokens, id_tokens):
                score += 8.0
            for raw_term in registration.routing_terms:
                term_tokens = _tokens(raw_term)
                if _contains_phrase(query_tokens, term_tokens):
                    score += 3.0 + (0.25 * len(term_tokens))
                elif len(term_tokens) > 1 and set(term_tokens).issubset(query_token_set):
                    score += 1.5 + (0.15 * len(term_tokens))
            if score > 0:
                matches.append(
                    CapabilityMatch(
                        descriptor=descriptor.model_copy(deep=True),
                        score=round(score, 6),
                    )
                )
        matches.sort(key=lambda item: (-item.score, item.descriptor.capability_id))
        return matches[: need.limit]


DEFAULT_REGISTRY = CapabilityRegistry(
    (
        RegisteredCapability(
            descriptor=CapabilityDescriptor(
                capability_id="internal_memory",
                kind=CapabilityKind.SERVICE,
                description="Retrieve bounded persisted internal evidence with exact provenance.",
            ),
            routing_terms=("internal memory",),
            executor="jit_memory",
            selectable_as_external_work=False,
        ),
    )
)


def deterministic_capability_request_id(requester_step_id: UUID) -> UUID:
    return uuid5(requester_step_id, "capability-request")


def deterministic_capability_event_id(request_id: UUID, role: str) -> UUID:
    return uuid5(request_id, f"event:{role}")


def request_capability(
    conn: psycopg.Connection,
    *,
    conversation_id: UUID,
    correlation_id: UUID,
    requester_task_id: UUID,
    requester_step_id: UUID,
    need: CapabilityNeed,
    registry: CapabilityRegistry = DEFAULT_REGISTRY,
) -> CapabilityPacket:
    request_id = deterministic_capability_request_id(requester_step_id)
    event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.CAPABILITY_REQUEST,
        source=SOURCE,
        payload={
            "registry_version": CAPABILITY_REGISTRY_VERSION,
            "capability_request_id": str(request_id),
            "requester_task_id": str(requester_task_id),
            "requester_step_id": str(requester_step_id),
            "need": need.model_dump(mode="json"),
        },
        payload_text=need.query_text,
        event_id=deterministic_capability_event_id(request_id, "request"),
    )
    matches, role, selected_text = registry.discover_with_trace(need)
    packet = CapabilityPacket(
        capability_request_id=request_id,
        requester_task_id=requester_task_id,
        requester_step_id=requester_step_id,
        need=need,
        matches=matches,
        selected_query_role=role,
        selected_query_text=selected_text,
    )
    event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.CAPABILITY_PACKET,
        source=SOURCE,
        payload={"packet": packet.model_dump(mode="json")},
        event_id=deterministic_capability_event_id(request_id, "packet"),
    )
    return packet


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.casefold()))


def _contains_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )
