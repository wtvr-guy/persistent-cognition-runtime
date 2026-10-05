import uuid
from pathlib import Path

import pytest

from prometheist.scale_corpus import build_scaled_document, load_document


ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"


@pytest.mark.parametrize(
    "filename",
    [
        "jordan_vale_v1.json",
        "avery_chen_v1.json",
        "morgan_reyes_v05_robustness.json",
    ],
)
def test_scaled_corpus_is_uuid_safe_and_preserves_oracle_references(filename):
    base = load_document(BENCHMARKS / filename)
    scaled = build_scaled_document(base, target_event_count=750)

    assert len(scaled["events"]) == 750
    event_ids = {event["event_id"] for event in scaled["events"]}
    conversation_ids = {event["conversation_id"] for event in scaled["events"]}

    assert len(event_ids) == 750
    assert all(str(uuid.UUID(event_id)) == event_id for event_id in event_ids)
    assert all(
        str(uuid.UUID(conversation_id)) == conversation_id
        for conversation_id in conversation_ids
    )

    for question in scaled["questions"]:
        assert set(question.get("required_event_ids", ())).issubset(event_ids)
        assert set(question.get("relevant_event_ids", ())).issubset(event_ids)

    base_text_by_id = {event["event_id"]: event["text"] for event in base["events"]}
    preserved = {
        event["benchmark_origin_event_id"]: event["text"]
        for event in scaled["events"]
        if "benchmark_origin_event_id" in event
    }
    assert preserved == base_text_by_id
    assert scaled["scale"]["probe_event_count"] >= 1


def test_scaled_corpus_is_deterministic_and_prefix_stable_with_probes():
    base = load_document(BENCHMARKS / "morgan_reyes_v05_robustness.json")

    first = build_scaled_document(
        base,
        target_event_count=750,
        seed=1234,
        probe_every=250,
    )
    second = build_scaled_document(
        base,
        target_event_count=750,
        seed=1234,
        probe_every=250,
    )
    larger = build_scaled_document(
        base,
        target_event_count=1250,
        seed=1234,
        probe_every=250,
    )

    assert first == second
    assert first["events"] == larger["events"][:750]

    first_probe_questions = [
        question
        for question in first["questions"]
        if question.get("category") == "scale_exact_probe"
    ]
    larger_probe_questions = [
        question
        for question in larger["questions"]
        if question.get("category") == "scale_exact_probe"
    ]
    assert first_probe_questions == larger_probe_questions[: len(first_probe_questions)]
    assert len(larger_probe_questions) > len(first_probe_questions) >= 1


def test_scaled_corpus_records_probe_and_noise_profile_without_double_counting():
    base = load_document(BENCHMARKS / "avery_chen_v1.json")
    scaled = build_scaled_document(
        base,
        target_event_count=130,
        seed=99,
        confusable_every=12,
        probe_every=24,
    )

    generated_count = 130 - len(base["events"])
    probe_count = generated_count // 24
    confusable_slots = generated_count // 12
    overlapping_probe_confusable_slots = generated_count // 24
    expected_confusable = confusable_slots - overlapping_probe_confusable_slots

    assert scaled["scale"]["generated_event_count"] == generated_count
    assert scaled["scale"]["probe_event_count"] == probe_count
    assert scaled["scale"]["distractor_event_count"] == generated_count - probe_count
    assert scaled["scale"]["confusable_distractor_count"] == expected_confusable
    assert scaled["scale"]["total_question_count"] == (
        len(base["questions"]) + probe_count
    )
    assert scaled["scale"]["id_scheme"] == "uuid5"

    probes = [
        event
        for event in scaled["events"]
        if event.get("payload", {}).get("synthetic_scale_probe")
    ]
    assert len(probes) == probe_count
    assert all(event["event_type"] == "USER_PROMPT" for event in probes)
