from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from prometheist import db
from prometheist.semantic_memory import (
    EvidenceRelation,
    EvidenceSourceKind,
    ResolutionStatus,
    current_semantic_resolution,
    record_semantic_evidence,
    selected_assertion,
    semantic_assertions,
    semantic_evidence,
    semantic_resolution_as_of,
    semantic_resolution_history,
)


@pytest.fixture
def conn():
    with db.get_connection() as connection:
        yield connection


def _at(offset_minutes: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)


def _record(
    conn,
    *,
    value,
    observed,
    known=None,
    resolved=None,
    subject="person:mike",
    property_name="preferred_drink",
    confidence=1.0,
    source=None,
    claim_valid_from=None,
    claim_valid_until=None,
    relation=EvidenceRelation.SUPPORTS,
):
    return record_semantic_evidence(
        conn,
        subject=subject,
        property=property_name,
        value=value,
        confidence=confidence,
        source_id=source or uuid4(),
        observed_at=observed,
        known_at=known or observed,
        resolved_at=resolved or known or observed,
        derivation_method="test/v2",
        claim_valid_from=claim_valid_from,
        claim_valid_until=claim_valid_until,
        relation=relation,
    )


def test_initial_observation_creates_assertion_evidence_and_resolution(conn):
    result = _record(conn, value="latte", observed=_at(0))

    assert result.assertion_created is True
    assert result.evidence_created is True
    assert result.resolution_created is True
    assert result.resolution.status is ResolutionStatus.ACCEPTED
    assert result.resolution.selected_assertion_id == result.assertion.assertion_id
    assert selected_assertion(conn, result.resolution) == result.assertion


def test_corroboration_accumulates_evidence_without_duplicate_assertion(conn):
    first = _record(conn, value="latte", observed=_at(0), confidence=0.9)
    second = _record(conn, value="latte", observed=_at(10), confidence=0.4)

    assert first.assertion.assertion_id == second.assertion.assertion_id
    assert second.assertion_created is False
    assert second.evidence_created is True
    # The conclusion is unchanged, but the resolution advances its effective
    # observation time so a later-arriving stale contradiction cannot displace it.
    assert second.resolution_created is True
    assert second.resolution.support_observed_at == _at(10)
    assert len(semantic_assertions(conn, "person:mike", "preferred_drink")) == 1
    evidence = semantic_evidence(conn, "person:mike", "preferred_drink")
    assert len(evidence) == 2
    assert {item.confidence for item in evidence} == {0.9, 0.4}
    assert current_semantic_resolution(
        conn, "person:mike", "preferred_drink"
    ).selected_assertion_id == first.assertion.assertion_id


def test_later_observation_represents_temporal_change_not_contradiction(conn):
    early = _record(conn, value="latte", observed=_at(0))
    later = _record(conn, value="espresso", observed=_at(60))

    current = current_semantic_resolution(conn, "person:mike", "preferred_drink")
    assert current.status is ResolutionStatus.ACCEPTED
    assert current.selected_assertion_id == later.assertion.assertion_id
    history = semantic_resolution_history(conn, "person:mike", "preferred_drink")
    assert len(history) == 2
    assert history[-1].supersedes == history[-2].resolution_id
    assert early.assertion in semantic_assertions(
        conn, "person:mike", "preferred_drink"
    )


def test_historical_backfill_does_not_replace_current_resolution(conn):
    current = _record(
        conn,
        value="engineer",
        observed=_at(100),
        subject="person:mike",
        property_name="job_title",
    )
    backfill = _record(
        conn,
        value="manager",
        observed=_at(10),
        known=_at(120),
        subject="person:mike",
        property_name="job_title",
    )

    resolution = current_semantic_resolution(conn, "person:mike", "job_title")
    assert resolution.selected_assertion_id == current.assertion.assertion_id
    assert backfill.resolution_created is False
    assert len(semantic_assertions(conn, "person:mike", "job_title")) == 2
    assert len(semantic_evidence(conn, "person:mike", "job_title")) == 2


def test_bitemporal_query_excludes_evidence_not_yet_known(conn):
    _record(
        conn,
        value="manager",
        observed=_at(10),
        known=_at(120),
        subject="person:mike",
        property_name="job_title",
    )

    before_learning = semantic_resolution_as_of(
        conn,
        "person:mike",
        "job_title",
        valid_at=_at(20),
        known_at=_at(100),
    )
    after_learning = semantic_resolution_as_of(
        conn,
        "person:mike",
        "job_title",
        valid_at=_at(20),
        known_at=_at(130),
    )

    assert before_learning.status is ResolutionStatus.UNKNOWN
    assert after_learning.status is ResolutionStatus.ACCEPTED


def test_bitemporal_query_reconstructs_different_real_world_times(conn):
    manager = _record(
        conn,
        value="manager",
        observed=_at(10),
        subject="person:mike",
        property_name="job_title",
    )
    engineer = _record(
        conn,
        value="engineer",
        observed=_at(100),
        subject="person:mike",
        property_name="job_title",
    )

    early = semantic_resolution_as_of(
        conn,
        "person:mike",
        "job_title",
        valid_at=_at(20),
        known_at=_at(150),
    )
    late = semantic_resolution_as_of(
        conn,
        "person:mike",
        "job_title",
        valid_at=_at(120),
        known_at=_at(150),
    )

    assert early.selected_assertion_id == manager.assertion.assertion_id
    assert late.selected_assertion_id == engineer.assertion.assertion_id


def test_later_corroboration_blocks_stale_conflicting_backfill(conn):
    initial = _record(conn, value="latte", observed=_at(0))
    corroborated = _record(conn, value="latte", observed=_at(20))
    stale_conflict = _record(
        conn,
        value="tea",
        observed=_at(10),
        known=_at(30),
    )

    current = current_semantic_resolution(conn, "person:mike", "preferred_drink")
    assert current.status is ResolutionStatus.ACCEPTED
    assert current.selected_assertion_id == initial.assertion.assertion_id
    assert current.support_observed_at == corroborated.evidence.observed_at
    assert stale_conflict.resolution_created is False


def test_opposition_older_than_latest_support_does_not_create_current_ambiguity(conn):
    accepted = _record(conn, value="latte", observed=_at(0))
    _record(conn, value="latte", observed=_at(20))
    stale_opposition = _record(
        conn,
        value="latte",
        observed=_at(10),
        known=_at(30),
        relation=EvidenceRelation.OPPOSES,
    )

    current = current_semantic_resolution(conn, "person:mike", "preferred_drink")
    assert current.status is ResolutionStatus.ACCEPTED
    assert current.selected_assertion_id == accepted.assertion.assertion_id
    assert current.support_observed_at == _at(20)
    assert stale_opposition.resolution_created is False


def test_equal_time_conflict_is_ambiguous_not_uuid_tie_break(conn):
    first = _record(conn, value="latte", observed=_at(0))
    second = _record(conn, value="tea", observed=_at(0))

    current = current_semantic_resolution(conn, "person:mike", "preferred_drink")
    assert current.status is ResolutionStatus.AMBIGUOUS
    assert current.selected_assertion_id is None
    assert set(current.candidate_assertion_ids) == {
        first.assertion.assertion_id,
        second.assertion.assertion_id,
    }


def test_explicit_claim_validity_is_distinct_from_observation_time(conn):
    assertion = _record(
        conn,
        value=True,
        observed=_at(100),
        known=_at(100),
        property_name="vegetarian",
        claim_valid_from=_at(-1000),
    )

    before_learning = semantic_resolution_as_of(
        conn,
        "person:mike",
        "vegetarian",
        valid_at=_at(-500),
        known_at=_at(50),
    )
    after_learning = semantic_resolution_as_of(
        conn,
        "person:mike",
        "vegetarian",
        valid_at=_at(-500),
        known_at=_at(120),
    )

    assert assertion.assertion.claim_valid_from == _at(-1000)
    assert before_learning.status is ResolutionStatus.UNKNOWN
    assert after_learning.selected_assertion_id == assertion.assertion.assertion_id


def test_expired_historical_claim_does_not_become_current(conn):
    historical = _record(
        conn,
        value="Seattle",
        observed=_at(100),
        known=_at(100),
        property_name="city",
        claim_valid_from=_at(-1000),
        claim_valid_until=_at(-500),
    )

    current = current_semantic_resolution(conn, "person:mike", "city")
    assert current.status is ResolutionStatus.UNKNOWN
    assert current.selected_assertion_id is None

    past = semantic_resolution_as_of(
        conn,
        "person:mike",
        "city",
        valid_at=_at(-750),
        known_at=_at(120),
    )
    assert past.selected_assertion_id == historical.assertion.assertion_id


def test_opposing_evidence_makes_selected_assertion_ambiguous(conn):
    source = uuid4()
    initial = _record(conn, value="latte", observed=_at(0), source=source)
    opposed = _record(
        conn,
        value="latte",
        observed=_at(5),
        source=uuid4(),
        relation=EvidenceRelation.OPPOSES,
    )

    assert initial.assertion.assertion_id == opposed.assertion.assertion_id
    current = current_semantic_resolution(conn, "person:mike", "preferred_drink")
    assert current.status is ResolutionStatus.AMBIGUOUS
    assert current.selected_assertion_id is None


def test_later_support_can_resolve_older_opposition_in_history(conn):
    _record(conn, value="latte", observed=_at(0))
    _record(
        conn,
        value="latte",
        observed=_at(5),
        relation=EvidenceRelation.OPPOSES,
    )
    later = _record(conn, value="latte", observed=_at(10))

    contested = semantic_resolution_as_of(
        conn,
        "person:mike",
        "preferred_drink",
        valid_at=_at(5),
        known_at=_at(20),
    )
    restored = semantic_resolution_as_of(
        conn,
        "person:mike",
        "preferred_drink",
        valid_at=_at(15),
        known_at=_at(20),
    )

    assert contested.status is ResolutionStatus.AMBIGUOUS
    assert restored.status is ResolutionStatus.ACCEPTED
    assert restored.selected_assertion_id == later.assertion.assertion_id


def test_semantic_evidence_can_reference_an_action(conn):
    action_id = uuid4()
    result = record_semantic_evidence(
        conn,
        subject="person:mike",
        property="planning_style",
        value="checklists",
        source_id=action_id,
        source_kind=EvidenceSourceKind.ACTION,
        observed_at=_at(0),
        known_at=_at(5),
        resolved_at=_at(5),
        confidence=0.8,
        derivation_method="reflection/v1",
    )

    assert result.evidence.source_kind is EvidenceSourceKind.ACTION
    assert result.evidence.source_id == action_id


def test_equal_time_equivalent_assertions_do_not_create_false_ambiguity(conn):
    first = _record(
        conn,
        value=True,
        observed=_at(0),
        property_name="vegetarian",
        claim_valid_from=_at(-100),
    )
    second = _record(
        conn,
        value=True,
        observed=_at(0),
        property_name="vegetarian",
        claim_valid_from=_at(-100),
        claim_valid_until=_at(1000),
    )

    resolution = current_semantic_resolution(conn, "person:mike", "vegetarian")
    assert resolution.status is ResolutionStatus.ACCEPTED
    assert resolution.selected_assertion_id == first.assertion.assertion_id
    assert set(resolution.candidate_assertion_ids) == {
        first.assertion.assertion_id,
        second.assertion.assertion_id,
    }


def test_conflicting_retry_fails_closed(conn):
    source = uuid4()
    _record(
        conn,
        value="latte",
        observed=_at(0),
        known=_at(10),
        source=source,
        confidence=0.9,
    )

    with pytest.raises(ValueError, match="conflicting immutable semantic evidence retry"):
        _record(
            conn,
            value="latte",
            observed=_at(0),
            known=_at(10),
            source=source,
            confidence=0.2,
        )


def test_evidence_history_is_not_capped_by_old_situation_window(conn):
    for index in range(65):
        _record(
            conn,
            value="latte",
            observed=_at(index),
            source=uuid4(),
        )

    assert len(semantic_assertions(conn, "person:mike", "preferred_drink")) == 1
    assert len(semantic_evidence(conn, "person:mike", "preferred_drink")) == 65


def test_resolution_history_remains_append_only(conn):
    first = _record(conn, value="latte", observed=_at(0))
    history_before = semantic_resolution_history(
        conn, "person:mike", "preferred_drink"
    )
    second = _record(conn, value="espresso", observed=_at(30))
    history_after = semantic_resolution_history(
        conn, "person:mike", "preferred_drink"
    )

    assert history_after[: len(history_before)] == history_before
    assert history_after[-1].supersedes == first.resolution.resolution_id
    assert history_after[-1].resolution_id == second.resolution.resolution_id
