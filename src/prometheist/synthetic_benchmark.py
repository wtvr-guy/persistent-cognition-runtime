"""Synthetic-life benchmark for the deterministic memory kernel.

The benchmark keeps oracle truth separate from the event stream.  The kernel
only receives generated events; oracle fields are used solely by the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from prometheist.memory_kernel import CueState, MemoryEvent, recall, reciprocal_rank


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    question_count: int
    successful_questions: int
    question_success_rate: float
    evidence_recall: float
    mean_reciprocal_rank: float
    unknown_abstention_rate: float
    failures: tuple[str, ...]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _event(raw: dict[str, Any]) -> MemoryEvent:
    return MemoryEvent(
        event_id=raw["event_id"],
        global_seq=raw["global_seq"],
        conversation_id=raw["conversation_id"],
        conversation_seq=raw["conversation_seq"],
        event_type=raw["event_type"],
        source=raw["source"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        text=raw["text"],
        payload=raw.get("payload", {}),
    )


def run_benchmark(path: str | Path, *, limit: int = 5) -> BenchmarkResult:
    document = _load(Path(path))
    events = tuple(_event(raw) for raw in document["events"])
    principal_name = document["persona"]["name"]
    principal_terms = tuple(principal_name.split())

    successes = 0
    required_total = 0
    required_found = 0
    rr_total = 0.0
    unknown_total = 0
    unknown_abstained = 0
    failures: list[str] = []

    for question in document["questions"]:
        reference_time = (
            datetime.fromisoformat(question["reference_time"])
            if question.get("reference_time")
            else None
        )
        entities = tuple(
            entity for entity in question.get("entities", [])
            if entity.casefold() != principal_name.casefold()
        )
        cue = CueState(
            query_text=question["query"],
            entities=entities,
            ignored_terms=principal_terms,
            reference_time=reference_time,
            limit=limit,
        )
        packet = recall(events, cue)
        returned = [event.event_id for event in packet.items]
        returned_set = set(returned)
        required = set(question.get("required_event_ids", []))
        relevant = set(question.get("relevant_event_ids", []))

        if question.get("expect_no_evidence", False):
            unknown_total += 1
            # No *relevant* evidence exists.  The strongest behavior is an empty
            # packet; nearest-neighbor noise counts as failure.
            passed = len(returned) == 0
            if passed:
                unknown_abstained += 1
        else:
            passed = required.issubset(returned_set)
            required_total += len(required)
            required_found += len(required & returned_set)
            rr_total += reciprocal_rank(packet, relevant or required)

        if passed:
            successes += 1
        else:
            failures.append(
                f"{question['id']}: required={sorted(required)} returned={returned} query={question['query']!r}"
            )

    answerable_count = max(1, len(document["questions"]) - unknown_total)
    return BenchmarkResult(
        question_count=len(document["questions"]),
        successful_questions=successes,
        question_success_rate=successes / len(document["questions"]),
        evidence_recall=required_found / required_total if required_total else 1.0,
        mean_reciprocal_rank=rr_total / answerable_count,
        unknown_abstention_rate=unknown_abstained / unknown_total if unknown_total else 1.0,
        failures=tuple(failures),
    )


def main() -> None:
    default_path = Path(__file__).resolve().parents[2] / "benchmarks" / "jordan_vale_v1.json"
    result = run_benchmark(default_path)
    print(f"questions: {result.successful_questions}/{result.question_count}")
    print(f"question_success_rate: {result.question_success_rate:.3f}")
    print(f"evidence_recall: {result.evidence_recall:.3f}")
    print(f"mean_reciprocal_rank: {result.mean_reciprocal_rank:.3f}")
    print(f"unknown_abstention_rate: {result.unknown_abstention_rate:.3f}")
    if result.failures:
        print("failures:")
        for failure in result.failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
