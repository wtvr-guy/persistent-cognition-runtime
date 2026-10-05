"""Accuracy and latency benchmark over deterministic large persona histories."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from prometheist.association_projection import derive_associations
from prometheist.associative_memory import associative_recall
from prometheist.memory_kernel import CueState, MemoryEvent
from prometheist.scale_corpus import (
    DEFAULT_BASE_CORPORA,
    DEFAULT_EVENT_COUNTS,
    DEFAULT_PROBE_EVERY,
    DEFAULT_SEED,
    build_scaled_document,
    load_document,
    write_scaled_document,
)
from prometheist.synthetic_benchmark import BenchmarkResult


@dataclass(frozen=True, slots=True)
class ScaleBenchmarkResult:
    persona: str
    event_count: int
    base_question_count: int
    probe_question_count: int
    association_count: int
    generation_seconds: float
    derivation_seconds: float
    recall_seconds: float
    recall_mean_ms: float
    recall_median_ms: float
    recall_p95_ms: float
    recall_max_ms: float
    benchmark: BenchmarkResult


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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int((len(ordered) - 1) * percentile + 0.999999),
        ),
    )
    return ordered[index]


def _cue_for_question(
    question: dict[str, Any],
    *,
    principal_name: str,
    limit: int,
) -> CueState:
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
    return CueState(
        query_text=question["query"],
        entities=entities,
        ignored_terms=tuple(principal_name.split()),
        reference_time=reference_time,
        limit=limit,
    )


def run_scale_document(
    base_document: dict[str, Any],
    *,
    target_event_count: int,
    seed: int = DEFAULT_SEED,
    confusable_every: int = 12,
    probe_every: int = DEFAULT_PROBE_EVERY,
    limit: int = 5,
) -> ScaleBenchmarkResult:
    generation_started = perf_counter()
    document = build_scaled_document(
        base_document,
        target_event_count=target_event_count,
        seed=seed,
        confusable_every=confusable_every,
        probe_every=probe_every,
    )
    generation_seconds = perf_counter() - generation_started
    events = tuple(_event(raw) for raw in document["events"])

    derivation_started = perf_counter()
    associations = derive_associations(events)
    derivation_seconds = perf_counter() - derivation_started

    principal_name = document["persona"]["name"]
    successes = 0
    required_total = 0
    required_found = 0
    rr_total = 0.0
    unknown_total = 0
    unknown_abstained = 0
    failures: list[str] = []
    recall_durations: list[float] = []

    for question in document["questions"]:
        cue = _cue_for_question(
            question,
            principal_name=principal_name,
            limit=limit,
        )
        started = perf_counter()
        packet = associative_recall(events, cue, associations)
        duration = perf_counter() - started
        recall_durations.append(duration)

        returned = [event.event_id for event in packet.items]
        returned_set = set(returned)
        required = set(question.get("required_event_ids", []))
        relevant = set(question.get("relevant_event_ids", []))

        if question.get("expect_no_evidence", False):
            unknown_total += 1
            passed = not returned
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
    benchmark = BenchmarkResult(
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
    recall_seconds = sum(recall_durations)
    durations_ms = [duration * 1000.0 for duration in recall_durations]
    scale = document["scale"]

    return ScaleBenchmarkResult(
        persona=principal_name,
        event_count=len(events),
        base_question_count=int(scale["base_question_count"]),
        probe_question_count=int(scale["probe_event_count"]),
        association_count=len(associations),
        generation_seconds=generation_seconds,
        derivation_seconds=derivation_seconds,
        recall_seconds=recall_seconds,
        recall_mean_ms=mean(durations_ms) if durations_ms else 0.0,
        recall_median_ms=median(durations_ms) if durations_ms else 0.0,
        recall_p95_ms=_percentile(durations_ms, 0.95),
        recall_max_ms=max(durations_ms, default=0.0),
        benchmark=benchmark,
    )


def _result_dict(result: ScaleBenchmarkResult) -> dict[str, Any]:
    raw = asdict(result)
    raw["benchmark"]["failures"] = list(result.benchmark.failures)
    return raw


def _print_result(result: ScaleBenchmarkResult) -> None:
    benchmark = result.benchmark
    print(
        " | ".join(
            (
                result.persona,
                f"events={result.event_count}",
                f"base_questions={result.base_question_count}",
                f"probe_questions={result.probe_question_count}",
                f"associations={result.association_count}",
                f"accuracy={benchmark.successful_questions}/{benchmark.question_count}",
                f"evidence_recall={benchmark.evidence_recall:.3f}",
                f"mrr={benchmark.mean_reciprocal_rank:.3f}",
                f"unknown_abstention={benchmark.unknown_abstention_rate:.3f}",
                f"derive_ms={result.derivation_seconds * 1000.0:.1f}",
                f"recall_p50_ms={result.recall_median_ms:.1f}",
                f"recall_p95_ms={result.recall_p95_ms:.1f}",
                f"recall_max_ms={result.recall_max_ms:.1f}",
            )
        )
    )
    for failure in benchmark.failures:
        print(f"  FAILURE {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure pure associative Memory Kernel accuracy/latency as persona histories scale."
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVENT_COUNTS),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confusable-every", type=int, default=12)
    parser.add_argument("--probe-every", type=int, default=DEFAULT_PROBE_EVERY)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--base",
        nargs="*",
        default=list(DEFAULT_BASE_CORPORA),
        help="Base benchmark filenames under benchmarks/.",
    )
    parser.add_argument(
        "--materialize-dir",
        type=Path,
        help="Optionally write each generated JSON corpus to this directory.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optionally write machine-readable benchmark results as JSON.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    results: list[ScaleBenchmarkResult] = []
    for base_name in args.base:
        base_path = benchmark_dir / base_name
        base_document = load_document(base_path)
        for event_count in args.events:
            if args.materialize_dir:
                write_scaled_document(
                    base_path,
                    args.materialize_dir / f"{base_path.stem}_scale_{event_count}.json",
                    target_event_count=event_count,
                    seed=args.seed,
                    confusable_every=args.confusable_every,
                    probe_every=args.probe_every,
                )
            result = run_scale_document(
                base_document,
                target_event_count=event_count,
                seed=args.seed,
                confusable_every=args.confusable_every,
                probe_every=args.probe_every,
                limit=args.limit,
            )
            results.append(result)
            _print_result(result)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps([_result_dict(result) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )
        print(args.report)


if __name__ == "__main__":
    main()
