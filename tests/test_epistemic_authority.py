from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from prometheist.epistemic_authority import (
    authority_for_event_type,
    format_authority_bound_memory_packet,
)
from prometheist.models import EventType, MemoryEvidence, MemoryNeed, MemoryPacket


def _evidence(event_type: EventType, content: str, seq: int) -> MemoryEvidence:
    return MemoryEvidence(
        source_event_id=uuid4(),
        event_type=event_type,
        source="test",
        created_at=datetime.now(timezone.utc),
        conversation_id=uuid4(),
        conversation_seq=seq,
        global_seq=seq,
        content=content,
    )


def test_assistant_output_cannot_establish_user_testimony():
    authority = authority_for_event_type(EventType.INTERACTION_RESPONSE)

    assert authority.authority_class == "MODEL_OUTPUT_ONLY"
    assert "what Prometheist or another model previously emitted" in authority.may_establish
    assert "that the user said" in authority.cannot_establish


def test_authority_inventory_keeps_user_and_model_roles_distinct():
    packet = MemoryPacket(
        memory_request_id=uuid4(),
        need=MemoryNeed(query_text="Favorite color?"),
        supported=True,
        items=[
            _evidence(EventType.USER_PROMPT, "My favorite color is green.", 1),
            _evidence(
                EventType.INTERACTION_RESPONSE,
                "The user's favorite color is poison-blue.",
                2,
            ),
        ],
    )

    rendered = format_authority_bound_memory_packet(packet)

    assert "direct_user_testimony_orders: 0" in rendered
    assert "authority_class: DIRECT_USER_TESTIMONY" in rendered
    assert "authority_class: MODEL_OUTPUT_ONLY" in rendered
    assert "MODEL_OUTPUT_ONLY is never a substitute for DIRECT_USER_TESTIMONY" in rendered


def test_every_canonical_event_type_has_an_authority_classification():
    assert {event_type: authority_for_event_type(event_type) for event_type in EventType}
