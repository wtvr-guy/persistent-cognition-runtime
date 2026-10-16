"""Focused traversal must accept admitted structured seeds without copied text."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from persistent_cognition import jit_memory
from persistent_cognition.models import Event, EventType, MemoryNeed


def event(kind, payload):
    return Event(
        event_id=uuid4(), conversation_id=uuid4(), correlation_id=uuid4(),
        global_seq=4, conversation_seq=2, event_type=kind,
        source="test", created_at=datetime.now(timezone.utc), payload=payload,
    )


@pytest.mark.parametrize(("kind", "payload"), [
    (EventType.MEMORY_PACKET, {"packet": {"items": [], "supported": False}}),
    (EventType.MEMORY_REQUEST, {"need": {"query_text": "Project Oriole codename"}}),
    (EventType.TOOL_RESULT, {"status": "complete", "result": {"code": "AB1234"}}),
])
def test_structured_focus_validates_ids_without_requiring_a_text_payload(
    monkeypatch, kind, payload,
):
    source = event(kind, payload)
    monkeypatch.setattr(jit_memory.event_store, "get_event_by_id", lambda *_: source)
    need = MemoryNeed(focus_event_ids=[source.event_id], source_types=[kind])
    assert jit_memory._validated_focus_event_ids(
        None, need, before_global_seq=5,
    ) == [source.event_id]
    assert source.payload == payload


@pytest.mark.parametrize("failure", ["missing", "future", "wrong_role"])
def test_structured_focus_still_rejects_invalid_canonical_seeds(monkeypatch, failure):
    source = event(EventType.MEMORY_PACKET, {"packet": {"items": []}})
    selected = None if failure == "missing" else source
    monkeypatch.setattr(jit_memory.event_store, "get_event_by_id", lambda *_: selected)
    need = MemoryNeed(
        focus_event_ids=[source.event_id],
        source_types=[EventType.USER_PROMPT if failure == "wrong_role" else source.event_type],
    )
    expected = {
        "missing": "no longer exists", "future": "leakage boundary",
        "wrong_role": "disallowed source type",
    }[failure]
    with pytest.raises(RuntimeError, match=expected):
        jit_memory._validated_focus_event_ids(
            None, need, before_global_seq=4 if failure == "future" else 5,
        )
