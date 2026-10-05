"""Durable interaction identity contracts for the authoritative v2 runtime.

This module contains only persisted data and deterministic identifiers. Semantic
routing policy belongs to the fresh v2 workers in ``percept_response_runtime``;
keeping those concerns separate prevents the retired recurrent interaction
architecture from becoming a compatibility dependency.
"""
from __future__ import annotations

from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, model_validator

from prometheist.perception import Percept, SalienceAssessment


INTERACTION_PROTOCOL_VERSION = "interaction-fixed-retrieval-v3"


class MemoryContext(Protocol):
    """Durable fields any percept envelope must expose to bounded memory expansion.

    ``DurableInteraction`` and the v0.8 situation envelope are separate durable
    records that legitimately share this contract, so memory composition depends on
    these fields rather than on one concrete envelope class.
    """

    @property
    def interaction_id(self) -> UUID: ...

    @property
    def conversation_id(self) -> UUID: ...

    @property
    def correlation_id(self) -> UUID: ...

    @property
    def task_id(self) -> UUID: ...

    @property
    def before_global_seq(self) -> int: ...

    @property
    def user_text(self) -> str: ...


class DurableInteraction(BaseModel):
    """Application-owned, restart-stable envelope for one percept."""

    protocol_version: str = INTERACTION_PROTOCOL_VERSION
    interaction_id: UUID
    conversation_id: UUID
    correlation_id: UUID
    user_prompt_event_id: UUID
    before_global_seq: int = Field(ge=1)
    task_id: UUID
    assignment_id: UUID
    user_text: str = Field(min_length=1)
    percept: Percept | None = None
    salience_assessment: SalienceAssessment | None = None

    @model_validator(mode="after")
    def validate_protocol_version(self) -> "DurableInteraction":
        if self.protocol_version != INTERACTION_PROTOCOL_VERSION:
            raise ValueError(
                "interaction protocol version is unsupported: "
                f"expected {INTERACTION_PROTOCOL_VERSION}, got {self.protocol_version}"
            )
        return self


def deterministic_interaction_id(conversation_id: UUID, correlation_id: UUID) -> UUID:
    return uuid5(conversation_id, f"interaction:{correlation_id}")


def deterministic_interaction_task_id(interaction_id: UUID) -> UUID:
    return uuid5(interaction_id, "attention-task")


def deterministic_interaction_event_id(interaction_id: UUID, role: str) -> UUID:
    return uuid5(interaction_id, f"event:{role}")


def deterministic_capability_memory_request_id(
    interaction_id: UUID,
    plan_position: int,
    capability_id: str,
) -> UUID:
    if plan_position < 0:
        raise ValueError("plan_position must be >= 0")
    normalized = capability_id.strip()
    if not normalized:
        raise ValueError("capability_id must not be empty")
    return uuid5(
        interaction_id,
        f"capability-memory-request:{plan_position}:{normalized}",
    )


def deterministic_aperture_request_id(interaction_id: UUID) -> UUID:
    return uuid5(interaction_id, "attention-aperture-memory-request")
