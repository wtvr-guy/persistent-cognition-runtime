"""Application-owned execution binding for non-selectable internal memory.

Memory expansion for user prompts belongs to the deterministic Adaptive Recall
loop in prometheist.percept_response_runtime. The retained jit_memory binding is a
task-neutral service boundary for non-interactive system work; it is never
exposed in the pre-cognitive capability catalog.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID, uuid5

import psycopg
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prometheist import event_store, jit_memory
from prometheist.capability_registry import RegisteredCapability
from prometheist.interaction_contracts import (
    deterministic_interaction_event_id,
    deterministic_interaction_id,
)
from prometheist.interaction_working_state import activate_working_state
from prometheist.models import EventType, MemoryPacket


CAPABILITY_EXECUTION_VERSION = "v0.7-capability-execution-v7"
SOURCE = "capability_runtime"
_INTERNAL_MEMORY_LIMIT = 5
_MEMORY_EVIDENCE_EXECUTORS = {"jit_memory"}


class CapabilityExecution(BaseModel):
    """Immutable structured result of one capability invocation."""

    model_config = ConfigDict(extra="forbid")
    execution_version: str = CAPABILITY_EXECUTION_VERSION
    capability_execution_id: UUID
    requester_task_id: UUID
    requester_step_id: UUID
    plan_position: int = Field(ge=0)
    capability_id: str = Field(min_length=1)
    executor: str = Field(min_length=1)
    memory_packet: MemoryPacket | None = None
    result_data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_execution(self) -> "CapabilityExecution":
        if self.execution_version != CAPABILITY_EXECUTION_VERSION:
            raise ValueError("capability execution version is unsupported")
        if self.executor in _MEMORY_EVIDENCE_EXECUTORS and self.memory_packet is None:
            raise ValueError("memory-evidence execution requires a MemoryPacket")
        return self


def _existing_execution(
    conn: psycopg.Connection,
    capability_execution_id: UUID,
) -> CapabilityExecution | None:
    event = event_store.get_event_by_id(
        conn,
        uuid5(capability_execution_id, "event:result"),
    )
    if event is None:
        return None
    payload = event.payload.get("execution")
    if not isinstance(payload, dict):
        raise RuntimeError("persisted capability result has invalid execution payload")
    return CapabilityExecution.model_validate(payload)


def execute_registered_capability(
    conn: psycopg.Connection,
    *,
    registration: RegisteredCapability,
    capability_execution_id: UUID,
    requester_task_id: UUID,
    requester_step_id: UUID,
    plan_position: int,
    conversation_id: UUID,
    correlation_id: UUID,
    task_text: str,
    before_global_seq: int,
    memory_request_id: UUID,
) -> CapabilityExecution:
    """Execute the retained task-neutral internal-memory service.

    Selectable external capabilities require explicit executor bindings when they
    are introduced. Unknown executor identifiers fail closed instead of silently
    being treated as memory operations.
    """

    existing = _existing_execution(conn, capability_execution_id)
    if existing is not None:
        return existing

    if registration.executor != "jit_memory":
        raise NotImplementedError(
            f"capability executor {registration.executor!r} has no execution binding"
        )

    need = jit_memory.build_memory_need(
        task_text,
        include_persisted_history=True,
        conversation_id=None,
        limit=_INTERNAL_MEMORY_LIMIT,
    )
    packet = jit_memory.request_memory(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        requesting_component=(
            f"task:{requester_task_id}/plan-position:{plan_position}/"
            f"capability:{registration.descriptor.capability_id}/step:{requester_step_id}"
        ),
        need=need,
        before_global_seq=before_global_seq,
        memory_request_id=memory_request_id,
        recall_stage=jit_memory.AdaptiveRecallStage.BROAD,
    )

    interaction_id = deterministic_interaction_id(conversation_id, correlation_id)
    prompt_event_id = deterministic_interaction_event_id(interaction_id, "user-prompt")
    activate_working_state(
        conn,
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        activated_event_ids=[
            *[item.source_event_id for item in packet.items],
            prompt_event_id,
        ],
        activation_key=f"capability:{plan_position}:{registration.descriptor.capability_id}",
    )

    execution = CapabilityExecution(
        capability_execution_id=capability_execution_id,
        requester_task_id=requester_task_id,
        requester_step_id=requester_step_id,
        plan_position=plan_position,
        capability_id=registration.descriptor.capability_id,
        executor=registration.executor,
        memory_packet=packet,
        result_data={
            "supported": packet.supported,
            "item_count": len(packet.items),
            "recall_stage": jit_memory.AdaptiveRecallStage.BROAD.value,
        },
    )
    event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.CAPABILITY_RESULT,
        source=SOURCE,
        payload={"execution": execution.model_dump(mode="json")},
        event_id=uuid5(capability_execution_id, "event:result"),
    )
    return execution
