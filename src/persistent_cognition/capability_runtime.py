"""Application-owned execution binding for non-selectable internal memory.

Memory expansion for user prompts belongs to the deterministic Adaptive Recall
loop in persistent_cognition.percept_response_runtime. The retained jit_memory binding is a
task-neutral service boundary for non-interactive system work; it is never
exposed in the pre-cognitive capability catalog.
"""
from __future__ import annotations

from typing import Any
from collections.abc import Mapping
from uuid import UUID, uuid5

import psycopg
from pydantic import BaseModel, ConfigDict, Field, model_validator

from persistent_cognition import event_store, jit_memory
from persistent_cognition.capability_registry import RegisteredCapability
from persistent_cognition.interaction_contracts import (
    deterministic_interaction_event_id,
    deterministic_interaction_id,
)
from persistent_cognition.interaction_working_state import activate_working_state
from persistent_cognition.models import EventType, MemoryPacket
from persistent_cognition.trusted_executors import (
    RESERVED_EXECUTORS,
    TrustedExecutionRequest,
    TrustedExecutorError,
    resolve_trusted_executor,
)


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
        return _execute_application_capability(
            conn,
            registration=registration,
            capability_execution_id=capability_execution_id,
            requester_task_id=requester_task_id,
            requester_step_id=requester_step_id,
            plan_position=plan_position,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            task_text=task_text,
            before_global_seq=before_global_seq,
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


def _execute_application_capability(
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
) -> CapabilityExecution:
    """Dispatch to an application-registered executor, or fail closed.

    The executor is resolved only by its pre-authorized name from the
    application-configured capability registration. Observed percept text is
    passed as structured data; it never selects or parameterizes the dispatch.
    """

    if registration.executor in RESERVED_EXECUTORS:
        raise RuntimeError(
            f"executor {registration.executor!r} is runtime-reserved and cannot "
            "be bound by an application"
        )
    handler = resolve_trusted_executor(registration.executor)
    if handler is None:
        raise NotImplementedError(
            f"capability executor {registration.executor!r} has no execution binding"
        )
    from persistent_cognition.trusted_executors import require_executor_revision
    require_executor_revision(registration.executor, registration.executor_revision)

    request = TrustedExecutionRequest(
        capability_id=registration.descriptor.capability_id,
        executor=registration.executor,
        requester_task_id=requester_task_id,
        requester_step_id=requester_step_id,
        plan_position=plan_position,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_text=task_text,
        before_global_seq=before_global_seq,
        execution_id=capability_execution_id,
    )
    try:
        raw_result = handler(request)
    except Exception as exc:
        raise TrustedExecutorError(type(exc).__name__) from exc
    if not isinstance(raw_result, Mapping):
        raise TrustedExecutorError("TypeError",
            f"trusted executor {registration.executor!r} must return a JSON object"
        )
    raw_result = dict(raw_result)
    try:
        from persistent_cognition.resource_limits import validate_json_intake
        validate_json_intake(raw_result)
    except (TypeError, ValueError) as exc:
        raise TrustedExecutorError(type(exc).__name__) from exc

    execution = CapabilityExecution(
        capability_execution_id=capability_execution_id,
        requester_task_id=requester_task_id,
        requester_step_id=requester_step_id,
        plan_position=plan_position,
        capability_id=registration.descriptor.capability_id,
        executor=registration.executor,
        memory_packet=None,
        result_data=dict(raw_result),
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
