from datetime import datetime, timezone

from prometheist.associative_memory import Association, associative_recall
from prometheist.memory_kernel import CueState, MemoryEvent


def _event(event_id: str, seq: int, text: str) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        global_seq=seq,
        conversation_id="c1",
        conversation_seq=seq,
        event_type="USER_PROMPT",
        source="test",
        created_at=datetime(2026, 1, seq, tzinfo=timezone.utc),
        text=text,
        payload={},
    )


def test_directly_matched_target_still_receives_association_activation():
    target = _event(
        "prior",
        1,
        "I started drinking black coffee instead of jasmine tea.",
    )
    source = _event(
        "change",
        2,
        "I switched from black coffee to green tea.",
    )
    association = Association(
        association_id="previous-state",
        source_kind="EVENT",
        source="change",
        target_kind="EVENT",
        target="prior",
        relationship="PREVIOUS_STATE",
        strength=1.0,
        provenance_event_ids=("prior", "change"),
        required_cue_terms=("before",),
    )

    packet = associative_recall(
        (target, source),
        CueState(
            query_text="What did I drink immediately before switching to green tea?",
            limit=2,
        ),
        (association,),
    )

    trace = next(item for item in packet.trace.items if item.event_id == "prior")
    assert trace.baseline_score >= 0.15  # target is already a direct lexical match
    assert trace.associative_activation > 0.0
    assert trace.total_score == trace.associative_activation
    assert trace.association_hops[0].relationship == "PREVIOUS_STATE"
