from datetime import datetime, timezone

from prometheist.memory_kernel import CueState, MemoryEvent, recall


def ev(event_id: str, seq: int, text: str, *, entities=()):
    return MemoryEvent(
        event_id=event_id,
        global_seq=seq,
        conversation_id="c1",
        conversation_seq=seq,
        event_type="USER_PROMPT",
        source="user",
        created_at=datetime(2026, 1, seq, tzinfo=timezone.utc),
        text=text,
        payload={"entities": list(entities)},
    )


def test_recall_is_deterministic_and_bounded():
    events = [
        ev("a", 1, "I like black coffee", entities=("coffee",)),
        ev("b", 2, "I bought a bicycle", entities=("bicycle",)),
        ev("c", 3, "Black coffee is still my morning drink", entities=("coffee",)),
    ]
    cue = CueState(query_text="What coffee do I drink?", entities=("coffee",), limit=2)

    first = recall(events, cue)
    second = recall(events, cue)

    assert first == second
    assert len(first.items) == 2
    assert [item.event_id for item in first.items] == ["c", "a"]
    assert first.trace.policy_version == "deterministic-cues-v2"
    assert all(item.reasons for item in first.trace.items)


def test_equal_relevance_preserves_both_chronology_endpoints():
    events = [
        ev("oldest", 1, "Project Falcon launch code alpha"),
        ev("middle-a", 2, "Project Falcon launch code bravo"),
        ev("middle-b", 3, "Project Falcon launch code charlie"),
        ev("newest", 4, "Project Falcon launch code delta"),
    ]

    packet = recall(
        events,
        CueState(query_text="Project Falcon launch code", limit=2),
    )

    assert [item.event_id for item in packet.items] == ["newest", "oldest"]


def test_unknown_query_returns_no_evidence_instead_of_nearest_noise():
    events = [ev("a", 1, "I like black coffee"), ev("b", 2, "I drive a Toyota")]
    packet = recall(events, CueState(query_text="What is my favorite color?", limit=5))
    assert packet.items == ()


def test_no_cues_returns_newest_history():
    events = [ev("a", 1, "first"), ev("b", 2, "second"), ev("c", 3, "third")]
    packet = recall(events, CueState(limit=2))
    assert [item.event_id for item in packet.items] == ["c", "b"]


def test_blank_entities_do_not_dilute_entity_score():
    events = [ev("a", 1, "Sarah called", entities=("Sarah",))]
    packet = recall(
        events,
        CueState(entities=("  ", "ＳＡＲＡＨ", ""), minimum_score=0.1, limit=1),
    )
    assert [item.event_id for item in packet.items] == ["a"]
    assert packet.trace.normalized_entities == ("sarah",)
    assert packet.trace.items[0].score.entity == 1.0


def test_trace_ignored_terms_are_normalized_once_semantically():
    events = [ev("a", 1, "Jordan drinks black coffee")]
    packet = recall(
        events,
        CueState(query_text="ＪＯＲＤＡＮ coffee", ignored_terms=("Jordan",), limit=1),
    )
    assert packet.trace.cue_tokens == ("coffee",)
