from datetime import datetime, timezone

import pytest

from prometheist.associative_memory import Association, associative_recall
from prometheist.memory_kernel import CueState, MemoryEvent


def ev(event_id: str, seq: int, text: str, *, entities=()):
    return MemoryEvent(
        event_id=event_id,
        global_seq=seq,
        conversation_id="c1",
        conversation_seq=seq,
        event_type="USER_PROMPT",
        source="test",
        created_at=datetime(2026, 1, seq, tzinfo=timezone.utc),
        text=text,
        payload={"entities": list(entities)},
    )


def test_event_association_recovers_related_evidence_with_provenance():
    events = [
        ev("old", 1, "I usually drank a latte."),
        ev("change", 2, "I stopped adding milk to coffee and drink it black."),
    ]
    association = Association(
        association_id="a1",
        source_kind="EVENT",
        source="change",
        target_kind="EVENT",
        target="old",
        relationship="PREVIOUS_STATE",
        strength=1.0,
        provenance_event_ids=("old", "change"),
        required_cue_terms=("before",),
    )

    packet = associative_recall(
        events,
        CueState(query_text="What did I drink before the coffee change?", limit=2),
        [association],
    )

    assert "old" in [event.event_id for event in packet.items]
    old_trace = next(item for item in packet.trace.items if item.event_id == "old")
    assert old_trace.associative_activation > 0
    assert old_trace.association_hops[0].relationship == "PREVIOUS_STATE"
    assert old_trace.association_hops[0].provenance_event_ids == ("old", "change")


def test_required_cue_terms_prevent_stale_state_pollution():
    events = [
        ev("old", 1, "I usually drank a latte."),
        ev("change", 2, "I stopped adding milk to coffee and drink it black."),
    ]
    association = Association(
        association_id="a1",
        source_kind="EVENT",
        source="change",
        target_kind="EVENT",
        target="old",
        relationship="PREVIOUS_STATE",
        strength=1.0,
        required_cue_terms=("before",),
    )

    packet = associative_recall(
        events,
        CueState(query_text="What coffee do I drink now?", limit=2),
        [association],
    )

    # The PREVIOUS_STATE edge is gated off without the required "before" cue.
    # Because the old event also has no direct score high enough to survive the
    # minimum-score threshold, it should be absent from both the evidence packet
    # and the ranked trace rather than appearing with zero activation.
    assert "old" not in [event.event_id for event in packet.items]
    assert all(item.event_id != "old" for item in packet.trace.items)


def test_term_to_event_association_supports_concept_to_instance_recall():
    events = [ev("car", 1, "I bought a 2021 Toyota Corolla.")]
    association = Association(
        association_id="a1",
        source_kind="TERM",
        source="vehicle",
        target_kind="EVENT",
        target="car",
        relationship="CONCEPT_INSTANCE",
        strength=1.0,
        provenance_event_ids=("car",),
    )

    packet = associative_recall(
        events,
        CueState(query_text="What vehicle do I own?", limit=1),
        [association],
    )

    assert [event.event_id for event in packet.items] == ["car"]
    assert packet.trace.items[0].associative_activation > 0


def test_term_associations_use_query_token_morphology_normalization():
    events = [ev("car", 1, "I bought a 2021 Toyota Corolla.")]
    association = Association(
        association_id="a1",
        source_kind="TERM",
        source="vehicles",
        target_kind="EVENT",
        target="car",
        relationship="CONCEPT_INSTANCE",
        strength=1.0,
    )

    packet = associative_recall(
        events,
        CueState(query_text="Which vehicle do I own?", limit=1),
        [association],
    )

    assert [event.event_id for event in packet.items] == ["car"]
    assert packet.trace.items[0].association_hops[0].source_node == "term:vehicle"


def test_unknown_query_still_abstains():
    events = [ev("coffee", 1, "I drink black coffee.")]
    associations = [
        Association(
            association_id="a1",
            source_kind="TERM",
            source="vehicle",
            target_kind="EVENT",
            target="coffee",
            relationship="UNRELATED_TEST_EDGE",
            strength=1.0,
        )
    ]

    packet = associative_recall(
        events,
        CueState(query_text="What is my favorite color?", limit=5),
        associations,
    )

    assert packet.items == ()


def test_two_hop_spreading_is_bounded_and_deterministic():
    events = [ev("target", 1, "The target evidence.")]
    associations = [
        Association("a1", "TERM", "alpha", "TERM", "beta", "RELATED", 1.0),
        Association("a2", "TERM", "beta", "EVENT", "target", "RELATED", 1.0),
    ]
    cue = CueState(query_text="alpha", limit=1)

    first = associative_recall(events, cue, associations, max_hops=2)
    second = associative_recall(events, cue, associations, max_hops=2)

    assert first == second
    assert [event.event_id for event in first.items] == ["target"]
    assert first.trace.items[0].association_hops[-1].hop == 2
    assert first.trace.items[0].associative_activation == pytest.approx(0.85**2)


def test_topical_overlap_without_requested_attribute_abstains():
    events = [
        ev(
            "parking",
            1,
            "Parking log: a delivery vehicle was noted near bay 42.",
        )
    ]

    packet = associative_recall(
        events,
        CueState(query_text="Who insures my vehicle?", limit=5),
        [],
    )

    # "vehicle" alone clears the broad activation score, but it is only half
    # of the substantive query. It may seed routing; it is not sufficient
    # evidence for the requested insurer attribute.
    assert packet.items == ()


def test_state_modifier_does_not_dilute_direct_support():
    events = [
        ev(
            "work",
            1,
            "I left Acme Design and started working at Northstar Labs this month.",
        )
    ]

    packet = associative_recall(
        events,
        CueState(query_text="Where do I work now?", limit=1),
        [],
    )

    assert [event.event_id for event in packet.items] == ["work"]


def test_temporal_cue_can_support_sparse_direct_lexical_match():
    events = [
        ev(
            "considering",
            1,
            "I'm thinking about buying a Subaru Outback this spring.",
        )
    ]

    packet = associative_recall(
        events,
        CueState(
            query_text="What car was I considering buying in March?",
            reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            limit=1,
        ),
        [],
    )

    assert [event.event_id for event in packet.items] == ["considering"]


def test_explicit_entity_support_survives_predicate_paraphrase():
    events = [
        ev(
            "claim",
            1,
            "Sarah Kim said Jordan hates mushrooms.",
            entities=("Sarah Kim", "mushrooms"),
        )
    ]

    packet = associative_recall(
        events,
        CueState(
            query_text="Who claimed Jordan hated mushrooms?",
            entities=("mushrooms",),
            ignored_terms=("Jordan",),
            limit=1,
        ),
        [],
    )

    assert [event.event_id for event in packet.items] == ["claim"]


def test_entity_tokens_do_not_dilute_historical_lexical_support():
    events = [
        ev(
            "old_job",
            1,
            "I'm still working at Acme Design downtown.",
            entities=("Acme Design",),
        )
    ]

    packet = associative_recall(
        events,
        CueState(
            query_text="Where did I work before Northstar Labs?",
            entities=("Northstar Labs",),
            limit=1,
        ),
        [],
    )

    # Northstar Labs is already an explicit entity cue. Its tokens should not
    # also count against lexical coverage for evidence describing the prior job.
    assert [event.event_id for event in packet.items] == ["old_job"]
