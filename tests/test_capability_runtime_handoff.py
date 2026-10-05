from __future__ import annotations

import uuid

import pytest

from prometheist import capability_runtime
from prometheist.capability_registry import (
    CapabilityDescriptor,
    CapabilityKind,
    RegisteredCapability,
)


def _external_registration() -> RegisteredCapability:
    return RegisteredCapability(
        descriptor=CapabilityDescriptor(
            capability_id="external.test",
            kind=CapabilityKind.TOOL,
            description="A test-only external tool.",
        ),
        routing_terms=("external",),
        executor="unbound.external",
    )


def test_unbound_external_executor_fails_closed_before_any_effect(monkeypatch):
    monkeypatch.setattr(
        capability_runtime.event_store,
        "get_event_by_id",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        capability_runtime.jit_memory,
        "request_memory",
        lambda *args, **kwargs: pytest.fail("unbound work must not become memory retrieval"),
    )
    monkeypatch.setattr(
        capability_runtime.event_store,
        "record_event",
        lambda *args, **kwargs: pytest.fail("unbound work must not persist a result"),
    )

    with pytest.raises(NotImplementedError, match="no execution binding"):
        capability_runtime.execute_registered_capability(
            object(),
            registration=_external_registration(),
            capability_execution_id=uuid.uuid4(),
            requester_task_id=uuid.uuid4(),
            requester_step_id=uuid.uuid4(),
            plan_position=0,
            conversation_id=uuid.uuid4(),
            correlation_id=uuid.uuid4(),
            task_text="Run external work.",
            before_global_seq=10,
            memory_request_id=uuid.uuid4(),
        )
