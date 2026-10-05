from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from persistent_cognition.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket
from persistent_cognition.fixed_retrieval import merge_evidence


def _evidence(label: str, seq: int) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid4(),
        event_type=EventType.USER_PROMPT,
        source="test",
        created_at=datetime.now(timezone.utc),
        conversation_id=uuid4(),
        conversation_seq=seq,
        global_seq=seq,
        content=label,
    )


def _packet(*items: MemoryEvidence) -> MemoryPacket:
    return MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="memory"),
        supported=bool(items),
        items=list(items),
    )


def test_saturated_expansion_interleaves_with_initial_aperture_evidence():
    base_items = (_evidence("base-a", 1), _evidence("base-b", 2))
    expansion_items = (
        _evidence("expansion-a", 3),
        _evidence("expansion-b", 4),
        _evidence("expansion-c", 5),
    )

    merged = merge_evidence(
        SimpleNamespace(interaction_id=uuid4(), before_global_seq=6),
        [_packet(*base_items), _packet(*expansion_items)],
        [EventType.USER_PROMPT], 3,
    )

    assert [item.source_event_id for item in merged.items] == [
        base_items[1].source_event_id,
        expansion_items[2].source_event_id,
        base_items[0].source_event_id,
    ]
    assert set(merged.retrieval_trace["excluded"]) == {
        str(item.source_event_id) for item in expansion_items[:2]
    }
