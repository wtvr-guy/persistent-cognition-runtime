from datetime import datetime, timezone

from prometheist.memory_kernel import MemoryEvent
from prometheist.memory_projection import LexicalProjection, build_projection, projection_digest


def test_projection_is_disposable_and_rebuilds_identically():
    events = (
        MemoryEvent(
            event_id="a",
            global_seq=1,
            conversation_id="c",
            conversation_seq=1,
            event_type="USER_PROMPT",
            source="user",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            text="Sarah ordered black coffee.",
            payload={"entities": ["Sarah", "black coffee"]},
        ),
    )
    projection = LexicalProjection()
    first = build_projection(events, projection)
    rebuilt = build_projection(events, projection)

    assert rebuilt == first
    assert projection_digest(rebuilt) == projection_digest(first)
    assert first[0].data["entities"] == ["black coffee", "sarah"]
    assert "coffee" in first[0].data["terms"]
