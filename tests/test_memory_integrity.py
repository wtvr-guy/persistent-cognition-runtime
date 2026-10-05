from dataclasses import replace
from datetime import datetime, timezone

from prometheist.memory_integrity import build_integrity_chain, verify_integrity_chain
from prometheist.memory_kernel import MemoryEvent


def event(event_id: str, seq: int, text: str):
    return MemoryEvent(
        event_id=event_id,
        global_seq=seq,
        conversation_id="c",
        conversation_seq=seq,
        event_type="USER_PROMPT",
        source="user",
        created_at=datetime(2026, 1, seq, tzinfo=timezone.utc),
        text=text,
        payload={"text": text},
    )


def test_integrity_chain_is_rebuildable_and_detects_tampering():
    events = (event("a", 1, "alpha"), event("b", 2, "beta"), event("c", 3, "gamma"))
    records = build_integrity_chain(events)

    assert build_integrity_chain(events) == records
    assert verify_integrity_chain(events, records).valid is True

    tampered = (events[0], replace(events[1], text="changed"), events[2])
    verification = verify_integrity_chain(tampered, records)
    assert verification.valid is False
    assert verification.first_invalid_event_id == "b"
