from __future__ import annotations

import uuid

from prometheist import capability_runtime, jit_memory
from prometheist.capability_registry import DEFAULT_REGISTRY
from prometheist.models import MemoryPacket


def test_internal_memory_service_is_not_walled_by_conversation(monkeypatch):
    captured = {}
    memory_request_id = uuid.uuid4()
    active_conversation_id = uuid.uuid4()

    monkeypatch.setattr(capability_runtime, "activate_working_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        capability_runtime.event_store,
        "get_event_by_id",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        capability_runtime.event_store,
        "record_event",
        lambda *args, **kwargs: None,
    )

    def fake_request_memory(_conn, **kwargs):
        captured.update(kwargs)
        return MemoryPacket(
            memory_request_id=kwargs["memory_request_id"],
            need=kwargs["need"],
            supported=False,
            items=[],
        )

    monkeypatch.setattr(capability_runtime.jit_memory, "request_memory", fake_request_memory)

    execution = capability_runtime.execute_registered_capability(
        object(),
        registration=DEFAULT_REGISTRY.get("internal_memory"),
        capability_execution_id=uuid.uuid4(),
        requester_task_id=uuid.uuid4(),
        requester_step_id=uuid.uuid4(),
        plan_position=0,
        conversation_id=active_conversation_id,
        correlation_id=uuid.uuid4(),
        task_text="What did I establish previously?",
        before_global_seq=73,
        memory_request_id=memory_request_id,
    )

    assert captured["conversation_id"] == active_conversation_id
    assert captured["need"].conversation_id is None
    assert captured["need"].include_persisted_history is True
    assert captured["need"].active_event_ids == []
    assert captured["before_global_seq"] == 73
    assert captured["recall_stage"] is jit_memory.AdaptiveRecallStage.BROAD
    assert execution.executor == "jit_memory"
