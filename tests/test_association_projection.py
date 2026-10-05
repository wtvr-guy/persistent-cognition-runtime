from datetime import datetime, timezone

from prometheist.association_projection import (
    association_projection_digest,
    derive_associations,
)
from prometheist.memory_kernel import MemoryEvent


def ev(event_id: str, seq: int, text: str, *, source="user", entities=()):
    return MemoryEvent(
        event_id=event_id,
        global_seq=seq,
        conversation_id="c1",
        conversation_seq=seq,
        event_type="USER_PROMPT",
        source=source,
        created_at=datetime(2026, 1, min(seq, 28), tzinfo=timezone.utc),
        text=text,
        payload={"entities": list(entities)},
    )


def test_derives_previous_state_with_provenance():
    events = (
        ev("old", 1, "I usually get a latte in the morning.", entities=("latte",)),
        ev("change", 2, "I stopped adding milk to coffee and drink it black.", entities=("coffee",)),
    )

    first = derive_associations(events)
    second = derive_associations(reversed(events))

    assert first == second
    assert association_projection_digest(first) == association_projection_digest(second)
    assert len(first) == 1
    association = first[0]
    assert association.relationship == "PREVIOUS_STATE"
    assert association.source == "change"
    assert association.target == "old"
    assert association.provenance_event_ids == ("old", "change")
    assert association.required_cue_terms == ("before",)


def test_derives_resolution_only_with_explicit_entity_overlap():
    events = (
        ev(
            "owed",
            1,
            "My landlord still owes me the security deposit.",
            entities=("security deposit",),
        ),
        ev(
            "other",
            2,
            "The store returned my jacket.",
            entities=("jacket",),
        ),
        ev(
            "resolved",
            3,
            "The landlord returned the security deposit.",
            entities=("security deposit",),
        ),
    )

    associations = derive_associations(events)
    resolutions = [a for a in associations if a.relationship == "RESOLVED_BY"]

    assert len(resolutions) == 1
    assert resolutions[0].source == "owed"
    assert resolutions[0].target == "resolved"
    assert resolutions[0].provenance_event_ids == ("owed", "resolved")


def test_vehicle_concept_instance_requires_acquisition_evidence_and_ownership_cue():
    events = (
        ev("considered", 1, "I am considering a Subaru Outback."),
        ev("bought", 2, "I bought a used 2021 Toyota Corolla."),
    )

    associations = derive_associations(events)
    vehicle_edges = [a for a in associations if a.relationship == "CONCEPT_INSTANCE"]

    assert len(vehicle_edges) == 1
    assert vehicle_edges[0].source_kind == "TERM"
    assert vehicle_edges[0].source == "vehicle"
    assert vehicle_edges[0].target == "bought"
    assert vehicle_edges[0].provenance_event_ids == ("bought",)
    assert vehicle_edges[0].required_cue_terms == ("own",)


def test_vehicle_concept_recognizes_unseen_model_from_manufacturer_entity():
    events = (
        ev(
            "bought",
            1,
            "I bought a 2024 Mazda CX-30 today.",
            entities=("2024 Mazda CX-30",),
        ),
    )

    associations = derive_associations(events)
    vehicle_edges = [a for a in associations if a.relationship == "CONCEPT_INSTANCE"]

    assert len(vehicle_edges) == 1
    assert vehicle_edges[0].target == "bought"
    assert vehicle_edges[0].source == "vehicle"


def test_vehicle_disposition_is_derived_as_later_ownership_state_evidence():
    events = (
        ev("bought", 1, "I bought a used 2021 Toyota Corolla."),
        ev("sold", 2, "I sold the Corolla to my neighbor."),
    )

    associations = derive_associations(events)
    dispositions = [a for a in associations if a.relationship == "CONCEPT_DISPOSITION"]

    assert len(dispositions) == 1
    disposition = dispositions[0]
    assert disposition.source_kind == "TERM"
    assert disposition.source == "vehicle"
    assert disposition.target == "sold"
    assert disposition.provenance_event_ids == ("sold",)
    assert disposition.required_cue_terms == ("own",)


def test_unrelated_events_do_not_manufacture_associations():
    events = (
        ev("a", 1, "I read a history book.", entities=("history book",)),
        ev("b", 2, "The package returned to the sender.", entities=("package",)),
    )

    assert derive_associations(events) == ()
