"""PostgreSQL-backed scale benchmark for the v0.5 associative memory path.

This module is intentionally destructive to the selected benchmark database: it
replaces all events for each scale profile. A hard database-name guard prevents
accidental execution against the ordinary development database.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any
import uuid

import psycopg
from psycopg.types.json import Json

from prometheist.db import require_row
from prometheist.memory_kernel import CueState
from prometheist.postgres_memory_kernel import associative_recall_from_postgres, rebuild
from prometheist.scale_corpus import (
    DEFAULT_BASE_CORPORA,
    DEFAULT_EVENT_COUNTS,
    DEFAULT_PROBE_EVERY,
    DEFAULT_SEED,
    build_scaled_document,
    load_document,
)
from prometheist.synthetic_benchmark import BenchmarkResult


@dataclass(frozen=True, slots=True)
class PostgresScaleBenchmarkResult:
    persona: str
    event_count: int
    base_question_count: int
    probe_question_count: int
    candidate_limit: int
    association_limit: int
    load_seconds: float
    rebuild_seconds: float
    recall_seconds: float
    recall_mean_ms: float
    recall_median_ms: float
    recall_p95_ms: float
    recall_max_ms: float
    projection_entry_count: int
    association_entry_count: int
    benchmark: BenchmarkResult


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


def _require_benchmark_database(conn: psycopg.Connection) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        database_name = str(require_row(cur.fetchone(), context="current_database()")[0])
    lowered = database_name.casefold()
    if "test" not in lowered and "benchmark" not in lowered:
        raise RuntimeError(
            "Refusing destructive scale benchmark against database "
            f"{database_name!r}; use a dedicated database whose name contains "
            "'test' or 'benchmark'."
        )
    return database_name


def _reset_database(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            TRUNCATE TABLE
                memory_association_entries,
                memory_projection_entries,
                memory_projection_runs,
                event_integrity,
                events,
                conversations
            RESTART IDENTITY CASCADE
            """
        )
    conn.commit()


def _load_document(conn: psycopg.Connection, document: dict[str, Any]) -> None:
    conversation_state: dict[str, dict[str, Any]] = {}
    for event in document["events"]:
        conversation_id = event["conversation_id"]
        state = conversation_state.setdefault(
            conversation_id,
            {
                "started_at": datetime.fromisoformat(event["created_at"]),
                "next_event_seq": 1,
            },
        )
        created_at = datetime.fromisoformat(event["created_at"])
        if created_at < state["started_at"]:
            state["started_at"] = created_at
        state["next_event_seq"] = max(
            state["next_event_seq"],
            int(event["conversation_seq"]) + 1,
        )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO conversations (conversation_id, started_at, next_event_seq)
            VALUES (%s, %s, %s)
            """,
            [
                (
                    uuid.UUID(conversation_id),
                    state["started_at"],
                    state["next_event_seq"],
                )
                for conversation_id, state in sorted(conversation_state.items())
            ],
        )
        cur.executemany(
            """
            INSERT INTO events (
                event_id, conversation_id, correlation_id, global_seq,
                conversation_seq, event_type, source, created_at,
                payload, payload_text, schema_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
            """,
            [
                (
                    uuid.UUID(event["event_id"]),
                    uuid.UUID(event["conversation_id"]),
                    uuid.uuid5(uuid.UUID(event["event_id"]), "scale-correlation"),
                    int(event["global_seq"]),
                    int(event["conversation_seq"]),
                    event["event_type"],
                    event["source"],
                    datetime.fromisoformat(event["created_at"]),
                    Json({**event.get("payload", {}), "text": event["text"]}),
                    event["text"],
                )
                for event in document["events"]
            ],
        )
        cur.execute(
            "SELECT setval(pg_get_serial_sequence('events', 'global_seq'), %s, true)",
            (len(document["events"]),),
        )
    conn.commit()


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


def run_postgres_scale_document(
    conn: psycopg.Connection,
    base_document: dict[str, Any],
    *,
    target_event_count: int,
    seed: int = DEFAULT_SEED,
    confusable_every: int = 12,
    probe_every: int = DEFAULT_PROBE_EVERY,
    limit: int = 5,
    candidate_limit: int = 500,
    association_limit: int = 250,
) -> PostgresScaleBenchmarkResult:
    document = build_scaled_document(
        base_document,
        target_event_count=target_event_count,
        seed=seed,
        confusable_every=confusable_every,
        probe_every=probe_every,
    )
    _require_benchmark_database(conn)
    _reset_database(conn)

    load_started = perf_counter()
    _load_document(conn, document)
    load_seconds = perf_counter() - load_started

    rebuild_started = perf_counter()
    rebuild_result = rebuild(conn)
    rebuild_seconds = perf_counter() - rebuild_started

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
        packet = associative_recall_from_postgres(
            conn,
            cue,
            candidate_limit=candidate_limit,
            association_limit=association_limit,
        )
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
    durations_ms = [duration * 1000.0 for duration in recall_durations]
    scale = document["scale"]

    return PostgresScaleBenchmarkResult(
        persona=principal_name,
        event_count=len(document["events"]),
        base_question_count=int(scale["base_question_count"]),
        probe_question_count=int(scale["probe_event_count"]),
        candidate_limit=candidate_limit,
        association_limit=association_limit,
        load_seconds=load_seconds,
        rebuild_seconds=rebuild_seconds,
        recall_seconds=sum(recall_durations),
        recall_mean_ms=mean(durations_ms) if durations_ms else 0.0,
        recall_median_ms=median(durations_ms) if durations_ms else 0.0,
        recall_p95_ms=_percentile(durations_ms, 0.95),
        recall_max_ms=max(durations_ms, default=0.0),
        projection_entry_count=int(rebuild_result["projection_entry_count"]),
        association_entry_count=int(rebuild_result["association_entry_count"]),
        benchmark=benchmark,
    )


def _result_dict(result: PostgresScaleBenchmarkResult) -> dict[str, Any]:
    raw = asdict(result)
    raw["benchmark"]["failures"] = list(result.benchmark.failures)
    return raw


def _print_result(result: PostgresScaleBenchmarkResult) -> None:
    benchmark = result.benchmark
    print(
        " | ".join(
            (
                result.persona,
                f"events={result.event_count}",
                f"base_questions={result.base_question_count}",
                f"probe_questions={result.probe_question_count}",
                f"accuracy={benchmark.successful_questions}/{benchmark.question_count}",
                f"evidence_recall={benchmark.evidence_recall:.3f}",
                f"mrr={benchmark.mean_reciprocal_rank:.3f}",
                f"unknown_abstention={benchmark.unknown_abstention_rate:.3f}",
                f"load_s={result.load_seconds:.3f}",
                f"rebuild_s={result.rebuild_seconds:.3f}",
                f"recall_p50_ms={result.recall_median_ms:.1f}",
                f"recall_p95_ms={result.recall_p95_ms:.1f}",
                f"recall_max_ms={result.recall_max_ms:.1f}",
                f"candidate_limit={result.candidate_limit}",
            )
        )
    )
    for failure in benchmark.failures:
        print(f"  FAILURE {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure indexed PostgreSQL associative recall over large deterministic persona histories."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("PROMETHEIST_BENCHMARK_DATABASE_URL"),
        help="Dedicated test/benchmark PostgreSQL URL. Never use the dev database.",
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
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--association-limit", type=int, default=250)
    parser.add_argument(
        "--base",
        nargs="*",
        default=list(DEFAULT_BASE_CORPORA),
        help="Base benchmark filenames under benchmarks/.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not args.database_url:
        parser.error(
            "--database-url or PROMETHEIST_BENCHMARK_DATABASE_URL is required; "
            "use a dedicated database whose name contains 'test' or 'benchmark'."
        )

    benchmark_dir = Path(__file__).resolve().parents[2] / "benchmarks"
    results: list[PostgresScaleBenchmarkResult] = []
    with psycopg.connect(args.database_url) as conn:
        database_name = _require_benchmark_database(conn)
        print(f"benchmark_database={database_name}")
        for base_name in args.base:
            base_document = load_document(benchmark_dir / base_name)
            for event_count in args.events:
                result = run_postgres_scale_document(
                    conn,
                    base_document,
                    target_event_count=event_count,
                    seed=args.seed,
                    confusable_every=args.confusable_every,
                    probe_every=args.probe_every,
                    limit=args.limit,
                    candidate_limit=args.candidate_limit,
                    association_limit=args.association_limit,
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
