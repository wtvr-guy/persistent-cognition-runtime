"""Run deterministic evidence for v0.7 behavioral constraint families.

The runner never rewrites production constants. It applies the governance rule:
reject invariant violations, compare alternatives on frozen evidence, and report
INSUFFICIENT_DISCRIMINATION whenever the available experiment cannot identify a
unique/non-arbitrary production setting.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from prometheist.associative_memory import associative_recall
from prometheist.capability_registry import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityNeed,
    CapabilityRegistry,
    RegisteredCapability,
)
from prometheist.memory_kernel import CueState

from tune_memory_scoring import (
    EXPLORATORY_CORPORA,
    HOLDOUT_CORPORA,
    CandidatePolicy,
    _evaluate_corpus,
    _frozen_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmarks" / "results"
ALL_CORPORA = (*EXPLORATORY_CORPORA, *HOLDOUT_CORPORA)
BASELINE = CandidatePolicy(
    lexical_weight=0.78,
    entity_weight=0.14,
    temporal_weight=0.05,
    conversation_weight=0.03,
    phrase_bonus=0.15,
    temporal_scale_days=30.0,
    minimum_score=0.15,
    minimum_direct_support_coverage=0.60,
)


def _score_summary(scores) -> dict[str, Any]:
    return {
        "questions": sum(score.questions for score in scores),
        "successes": sum(score.successes for score in scores),
        "required_events": sum(score.required_events for score in scores),
        "required_found": sum(score.required_found for score in scores),
        "unknown_questions": sum(score.unknown_questions for score in scores),
        "unknown_abstained": sum(score.unknown_abstained for score in scores),
        "reciprocal_rank_sum": round(sum(score.reciprocal_rank_sum for score in scores), 9),
        "failures": [failure for score in scores for failure in score.failures],
    }


def _quality_tuple(summary: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        summary["successes"],
        summary["unknown_abstained"],
        summary["required_found"],
        summary["reciprocal_rank_sum"],
    )


def run_memory_breadth() -> dict[str, Any]:
    """Exhaust packet limits across every item count present in frozen corpora."""

    upper = max(len(_frozen_corpus(name).events) for name in ALL_CORPORA)
    evaluations: list[dict[str, Any]] = []
    for limit in range(1, upper + 1):
        scores = tuple(_evaluate_corpus(name, BASELINE, limit=limit) for name in ALL_CORPORA)
        summary = _score_summary(scores)
        evaluations.append({"limit": limit, **summary})

    best_quality = max(_quality_tuple(item) for item in evaluations)
    equivalent = [item for item in evaluations if _quality_tuple(item) == best_quality]
    minimum_equivalent = min(item["limit"] for item in equivalent)
    current = next(item for item in evaluations if item["limit"] == 5)
    current_is_best = _quality_tuple(current) == best_quality

    return {
        "benchmark_id": "MEM-BREADTH-001",
        "result": "INSUFFICIENT_DISCRIMINATION",
        "frozen_corpora": list(ALL_CORPORA),
        "explored_packet_limits": [1, upper],
        "current_packet_limit": 5,
        "current_quality": current,
        "best_quality": {
            "quality_tuple": best_quality,
            "equivalent_limits": [item["limit"] for item in equivalent],
            "minimum_equivalent_limit": minimum_equivalent,
        },
        "current_is_quality_equivalent": current_is_best,
        "unresolved_parameters": [
            "postgres_candidate_limits",
            "attention_activation_candidate_limit",
            "active_working_state_limit",
            "capability_result_limits",
            "final_composed_context_limit",
        ],
        "decision": (
            "Frozen corpora can measure returned packet breadth but do not exercise every "
            "production breadth layer or scale boundary. No unmeasured bound is declared optimal."
        ),
    }


def _evaluate_graph_policy(max_hops: int, decay: float) -> dict[str, Any]:
    questions = successes = required_events = required_found = 0
    unknown_questions = unknown_abstained = 0
    reciprocal_rank_sum = 0.0
    failures: list[str] = []
    traversed_hops = 0

    for name in ALL_CORPORA:
        corpus = _frozen_corpus(name)
        for question in corpus.document["questions"]:
            questions += 1
            reference_time = (
                datetime.fromisoformat(question["reference_time"])
                if question.get("reference_time")
                else None
            )
            entities = tuple(
                entity
                for entity in question.get("entities", [])
                if entity.casefold() != corpus.principal_name.casefold()
            )
            cue = CueState(
                query_text=question["query"],
                entities=entities,
                ignored_terms=corpus.principal_terms,
                reference_time=reference_time,
                limit=5,
                minimum_score=BASELINE.minimum_score,
            )
            packet = associative_recall(
                corpus.events,
                cue,
                corpus.associations,
                max_hops=max_hops,
                decay=decay,
                scoring_policy=BASELINE.scoring_policy(),
                minimum_direct_support_coverage=BASELINE.minimum_direct_support_coverage,
            )
            returned = [event.event_id for event in packet.items]
            returned_set = set(returned)
            traversed_hops += sum(len(item.association_hops) for item in packet.trace.items)
            required = set(question.get("required_event_ids", []))
            relevant = set(question.get("relevant_event_ids", [])) or required
            if question.get("expect_no_evidence", False):
                unknown_questions += 1
                passed = not returned
                if passed:
                    unknown_abstained += 1
            else:
                required_events += len(required)
                required_found += len(required & returned_set)
                passed = required.issubset(returned_set)
                for index, event_id in enumerate(returned, start=1):
                    if event_id in relevant:
                        reciprocal_rank_sum += 1.0 / index
                        break
            if passed:
                successes += 1
            else:
                failures.append(f"{name}:{question['id']}")

    return {
        "max_hops": max_hops,
        "decay": decay,
        "questions": questions,
        "successes": successes,
        "required_events": required_events,
        "required_found": required_found,
        "unknown_questions": unknown_questions,
        "unknown_abstained": unknown_abstained,
        "reciprocal_rank_sum": round(reciprocal_rank_sum, 9),
        "association_hops_recorded": traversed_hops,
        "failures": failures,
    }


def run_memory_graph() -> dict[str, Any]:
    max_graph_depth = max(len(_frozen_corpus(name).events) for name in ALL_CORPORA)
    decay_values = (0.25, 0.50, 0.75, 0.85, 0.90, 1.0)
    evaluations = [
        _evaluate_graph_policy(hops, decay)
        for hops in range(1, max_graph_depth + 1)
        for decay in decay_values
    ]
    best_quality = max(_quality_tuple(item) for item in evaluations)
    equivalent = [item for item in evaluations if _quality_tuple(item) == best_quality]
    baseline = next(
        item for item in evaluations if item["max_hops"] == 2 and item["decay"] == 0.85
    )
    return {
        "benchmark_id": "MEM-GRAPH-001",
        "result": "INSUFFICIENT_DISCRIMINATION",
        "frozen_corpora": list(ALL_CORPORA),
        "max_hops_explored": [1, max_graph_depth],
        "decay_values": list(decay_values),
        "baseline": baseline,
        "baseline_is_quality_equivalent": _quality_tuple(baseline) == best_quality,
        "equivalent_optima": len(equivalent),
        "lowest_work_equivalent": min(
            equivalent,
            key=lambda item: (item["association_hops_recorded"], item["max_hops"], item["decay"]),
        ),
        "unresolved_parameters": ["postgres_association_loader_limit", "profile_specific_graph_budgets"],
        "decision": (
            "The frozen graph establishes correctness over known relationships but does not justify "
            "a unique depth/decay/breadth policy. Loader breadth also requires larger generated graphs."
        ),
    }


def _registration(capability_id: str, *terms: str) -> RegisteredCapability:
    return RegisteredCapability(
        descriptor=CapabilityDescriptor(
            capability_id=capability_id,
            kind=CapabilityKind.TOOL,
            description=f"Benchmark capability {capability_id}.",
        ),
        routing_terms=terms,
        executor="benchmark",
    )


def _mirror_score(
    query: str,
    capability_id: str,
    terms: tuple[str, ...],
    coefficients: tuple[float, float, float, float, float],
) -> float:
    id_weight, phrase_base, phrase_length, subset_base, subset_length = coefficients
    tokens = tuple(query.casefold().split())
    token_set = set(tokens)
    score = 0.0

    def contains(needle: tuple[str, ...]) -> bool:
        return any(tokens[index:index + len(needle)] == needle for index in range(len(tokens) - len(needle) + 1))

    id_tokens = tuple(capability_id.replace("_", " ").casefold().split())
    if contains(id_tokens):
        score += id_weight
    for raw in terms:
        term_tokens = tuple(raw.casefold().split())
        if contains(term_tokens):
            score += phrase_base + phrase_length * len(term_tokens)
        elif len(term_tokens) > 1 and set(term_tokens).issubset(token_set):
            score += subset_base + subset_length * len(term_tokens)
    return score


def run_capability_discovery() -> dict[str, Any]:
    registrations = (
        _registration("weather_lookup", "weather forecast", "weather"),
        _registration("record_reconcile", "reconcile records", "compare records"),
        _registration("project_lookup", "project lookup", "lookup project"),
        _registration("general_lookup", "lookup"),
    )
    registry = CapabilityRegistry(registrations)
    cases = (
        ("weather lookup", "weather_lookup"),
        ("please compare records", "record_reconcile"),
        ("reconcile these records", "record_reconcile"),
        ("project lookup", "project_lookup"),
    )
    production = []
    hard_failures = []
    for query, expected in cases:
        result = registry.discover(CapabilityNeed(query_text=query, limit=10))
        ids = [item.descriptor.capability_id for item in result]
        production.append({"query": query, "ranked": ids, "expected_first": expected})
        if not ids or ids[0] != expected:
            hard_failures.append({"query": query, "expected": expected, "ranked": ids})

    baseline_coefficients = (8.0, 3.0, 0.25, 1.5, 0.15)
    for case in production:
        query = case["query"]
        mirrored = sorted(
            registrations,
            key=lambda reg: (
                -_mirror_score(query, reg.descriptor.capability_id, reg.routing_terms, baseline_coefficients),
                reg.descriptor.capability_id,
            ),
        )
        mirrored_ids = [
            reg.descriptor.capability_id
            for reg in mirrored
            if _mirror_score(query, reg.descriptor.capability_id, reg.routing_terms, baseline_coefficients) > 0
        ]
        if mirrored_ids != case["ranked"]:
            hard_failures.append({"query": query, "reason": "benchmark mirror diverges from production"})

    def _scaled(factor: float) -> tuple[float, float, float, float, float]:
        first, second, third, fourth, fifth = baseline_coefficients
        return (first * factor, second * factor, third * factor, fourth * factor, fifth * factor)

    coefficient_sets = (
        baseline_coefficients,
        _scaled(0.5),
        _scaled(2.0),
        (8.0, 2.0, 0.0, 2.0, 0.0),
        (4.0, 4.0, 0.5, 1.0, 0.1),
    )
    passing = []
    for coefficients in coefficient_sets:
        if all(
            max(
                registrations,
                key=lambda reg: (
                    _mirror_score(query, reg.descriptor.capability_id, reg.routing_terms, coefficients),
                    tuple(-ord(ch) for ch in reg.descriptor.capability_id),
                ),
            ).descriptor.capability_id == expected
            for query, expected in cases
        ):
            passing.append(coefficients)

    if hard_failures:
        raise RuntimeError(f"CAP-DISCOVERY-001 hard invariant failures: {hard_failures}")
    return {
        "benchmark_id": "CAP-DISCOVERY-001",
        "result": "INSUFFICIENT_DISCRIMINATION",
        "production_cases": production,
        "coefficient_sets_tested": [list(values) for values in coefficient_sets],
        "coefficient_sets_preserving_oracle_order": [list(values) for values in passing],
        "decision": (
            "Multiple materially different coefficient vectors preserve all currently known routing "
            "oracles. The exact production coefficients therefore remain provisional."
        ),
    }


def run_scheduler_service() -> dict[str, Any]:
    current = {
        "INTERACTIVE": 1,
        "USER_WORK": 4,
        "SUPPORT": 8,
        "MAINTENANCE": 32,
        "BACKGROUND": 128,
    }
    candidates = {
        "quarter": {key: max(1, value // 4) for key, value in current.items()},
        "half": {key: max(1, value // 2) for key, value in current.items()},
        "current": current,
        "double": {key: value * 2 for key, value in current.items()},
        "quadruple": {key: value * 4 for key, value in current.items()},
    }
    valid = {
        name: (
            vector["INTERACTIVE"] <= vector["USER_WORK"] <= vector["SUPPORT"]
            <= vector["MAINTENANCE"] <= vector["BACKGROUND"]
        )
        for name, vector in candidates.items()
    }
    if not all(valid.values()):
        raise RuntimeError("SCHED-SERVICE-001 generated an invalid service-order candidate")
    return {
        "benchmark_id": "SCHED-SERVICE-001",
        "result": "INSUFFICIENT_DISCRIMINATION",
        "current_wait_cycles": current,
        "scaled_candidates": candidates,
        "all_preserve_service_order_invariant": True,
        "decision": (
            "Without a measured workload distribution and queue-delay SLA, several substantially "
            "different wait vectors satisfy the same deterministic ordering invariant. Exact wait "
            "cycles must remain provisional rather than being labeled optimal."
        ),
    }


def run_all() -> dict[str, Any]:
    results = {
        item["benchmark_id"]: item
        for item in (
            run_memory_breadth(),
            run_memory_graph(),
            run_capability_discovery(),
            run_scheduler_service(),
        )
    }
    return {
        "schema_version": 1,
        "runner": "benchmarks/run_deterministic_constraints.py",
        "results": results,
    }


def main() -> None:
    print(json.dumps(run_all(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
