"""Comparative synthetic-life benchmark for Memory Kernel v0.3 associative recall."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from prometheist.associative_memory import Association, associative_recall
from prometheist.memory_kernel import CueState, MemoryEvent
from prometheist.synthetic_benchmark import BenchmarkResult, run_benchmark


@dataclass(frozen=True, slots=True)
class ComparativeBenchmarkResult:
    baseline: BenchmarkResult
    associative: BenchmarkResult


def _load_json(path: Path) -> dict[str, Any]:
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


def _association(raw: dict[str, Any]) -> Association:
    return Association(
        association_id=raw["association_id"],
        source_kind=raw["source_kind"],
        source=raw["source"],
        target_kind=raw["target_kind"],
        target=raw["target"],
        relationship=raw["relationship"],
        strength=float(raw["strength"]),
        provenance_event_ids=tuple(raw.get("provenance_event_ids", [])),
        required_cue_terms=tuple(raw.get("required_cue_terms", [])),
    )


def run_associative_benchmark(
    benchmark_path: str | Path,
    association_path: str | Path,
    *,
    limit: int = 5,
) -> BenchmarkResult:
    document = _load_json(Path(benchmark_path))
    association_document = _load_json(Path(association_path))
    events = tuple(_event(raw) for raw in document["events"])
    associations = tuple(
        _association(raw) for raw in association_document["associations"]
    )
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
            entity
            for entity in question.get("entities", [])
            if entity.casefold() != principal_name.casefold()
        )
        cue = CueState(
            query_text=question["query"],
            entities=entities,
            ignored_terms=principal_terms,
            reference_time=reference_time,
            limit=limit,
        )
        packet = associative_recall(events, cue, associations)
        returned = [event.event_id for event in packet.items]
        returned_set = set(returned)
        required = set(question.get("required_event_ids", []))
        relevant = set(question.get("relevant_event_ids", []))

        if question.get("expect_no_evidence", False):
            unknown_total += 1
            passed = len(returned) == 0
            if passed:
                unknown_abstained += 1
        else:
            passed = required.issubset(returned_set)
            required_total += len(required)
            required_found += len(required & returned_set)
            for index, event_id in enumerate(returned, start=1):
                if event_id in (relevant or required):
                    rr_total += 1.0 / index
                    break

        if passed:
            successes += 1
        else:
            failures.append(
                f"{question['id']}: required={sorted(required)} "
                f"returned={returned} query={question['query']!r}"
            )

    answerable_count = max(1, len(document["questions"]) - unknown_total)
    return BenchmarkResult(
        question_count=len(document["questions"]),
        successful_questions=successes,
        question_success_rate=successes / len(document["questions"]),
        evidence_recall=required_found / required_total if required_total else 1.0,
        mean_reciprocal_rank=rr_total / answerable_count,
        unknown_abstention_rate=(
            unknown_abstained / unknown_total if unknown_total else 1.0
        ),
        failures=tuple(failures),
    )


def run_comparison(
    benchmark_path: str | Path,
    association_path: str | Path,
    *,
    limit: int = 5,
) -> ComparativeBenchmarkResult:
    return ComparativeBenchmarkResult(
        baseline=run_benchmark(benchmark_path, limit=limit),
        associative=run_associative_benchmark(
            benchmark_path,
            association_path,
            limit=limit,
        ),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    benchmark_path = root / "benchmarks" / "jordan_vale_v1.json"
    association_path = root / "benchmarks" / "jordan_vale_associations_v1.json"
    result = run_comparison(benchmark_path, association_path)

    print("v0.2 baseline")
    print(
        f"  questions: {result.baseline.successful_questions}/"
        f"{result.baseline.question_count}"
    )
    print(f"  question_success_rate: {result.baseline.question_success_rate:.3f}")
    print(f"  evidence_recall: {result.baseline.evidence_recall:.3f}")
    print(f"  mean_reciprocal_rank: {result.baseline.mean_reciprocal_rank:.3f}")
    print(f"  unknown_abstention_rate: {result.baseline.unknown_abstention_rate:.3f}")

    print("v0.3 associative")
    print(
        f"  questions: {result.associative.successful_questions}/"
        f"{result.associative.question_count}"
    )
    print(f"  question_success_rate: {result.associative.question_success_rate:.3f}")
    print(f"  evidence_recall: {result.associative.evidence_recall:.3f}")
    print(f"  mean_reciprocal_rank: {result.associative.mean_reciprocal_rank:.3f}")
    print(f"  unknown_abstention_rate: {result.associative.unknown_abstention_rate:.3f}")
    if result.associative.failures:
        print("associative failures:")
        for failure in result.associative.failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
