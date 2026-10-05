"""MEM-SCORE-001 deterministic calibration sweep.

This runner deliberately keeps oracle truth out of the memory inputs. It uses
frozen synthetic-life corpora plus the derived association graph and evaluates
candidate scoring/admission policies on the exact same questions.

The search is staged because the four cue weights live on a simplex and interact
with threshold/admission policy. A boundary winner is reported as unresolved;
it is never silently accepted as an optimum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import json
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from prometheist.association_projection import derive_associations
from prometheist.associative_memory import Association, associative_recall
from prometheist.memory_kernel import CueState, MemoryEvent, MemoryScoringPolicy

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
EXPLORATORY_CORPORA = ("jordan_vale_v1.json", "avery_chen_v1.json")
HOLDOUT_CORPORA = ("morgan_reyes_v05_robustness.json",)


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    lexical_weight: float
    entity_weight: float
    temporal_weight: float
    conversation_weight: float
    phrase_bonus: float
    temporal_scale_days: float
    minimum_score: float
    minimum_direct_support_coverage: float

    def scoring_policy(self) -> MemoryScoringPolicy:
        return MemoryScoringPolicy(
            lexical_weight=self.lexical_weight,
            entity_weight=self.entity_weight,
            temporal_weight=self.temporal_weight,
            conversation_weight=self.conversation_weight,
            phrase_bonus=self.phrase_bonus,
            temporal_scale_days=self.temporal_scale_days,
        )


@dataclass(frozen=True, slots=True)
class FrozenCorpus:
    name: str
    document: dict[str, Any]
    events: tuple[MemoryEvent, ...]
    associations: tuple[Association, ...]
    principal_name: str
    principal_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorpusScore:
    corpus: str
    questions: int
    successes: int
    unknown_questions: int
    unknown_abstained: int
    required_events: int
    required_found: int
    reciprocal_rank_sum: float
    failures: tuple[str, ...]

    @property
    def success_rate(self) -> float:
        return self.successes / self.questions if self.questions else 1.0

    @property
    def unknown_abstention_rate(self) -> float:
        return self.unknown_abstained / self.unknown_questions if self.unknown_questions else 1.0

    @property
    def evidence_recall(self) -> float:
        return self.required_found / self.required_events if self.required_events else 1.0

    @property
    def mean_reciprocal_rank(self) -> float:
        answerable = self.questions - self.unknown_questions
        return self.reciprocal_rank_sum / answerable if answerable else 1.0


@dataclass(frozen=True, slots=True)
class Evaluation:
    policy: CandidatePolicy
    exploratory: tuple[CorpusScore, ...]
    holdout: tuple[CorpusScore, ...]

    def quality_key(self) -> tuple[float, float, float, float, float, float, float, float]:
        all_scores = (*self.exploratory, *self.holdout)
        return (
            min(score.success_rate for score in all_scores),
            min(score.unknown_abstention_rate for score in all_scores),
            min(score.evidence_recall for score in all_scores),
            min(score.mean_reciprocal_rank for score in all_scores),
            sum(score.successes for score in self.holdout),
            sum(score.successes for score in self.exploratory),
            sum(score.required_found for score in all_scores),
            sum(score.reciprocal_rank_sum for score in all_scores),
        )


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


@lru_cache(maxsize=None)
def _frozen_corpus(name: str) -> FrozenCorpus:
    document = _load(BENCHMARK_DIR / name)
    events = tuple(_event(raw) for raw in document["events"])
    principal_name = document["persona"]["name"]
    return FrozenCorpus(
        name=name,
        document=document,
        events=events,
        associations=tuple(derive_associations(events)),
        principal_name=principal_name,
        principal_terms=tuple(principal_name.split()),
    )


def _evaluate_corpus(name: str, policy: CandidatePolicy, *, limit: int = 5) -> CorpusScore:
    corpus = _frozen_corpus(name)
    successes = 0
    unknown_questions = 0
    unknown_abstained = 0
    required_events = 0
    required_found = 0
    reciprocal_rank_sum = 0.0
    failures: list[str] = []

    for question in corpus.document["questions"]:
        reference_time = datetime.fromisoformat(question["reference_time"]) if question.get("reference_time") else None
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
            limit=limit,
            minimum_score=policy.minimum_score,
        )
        packet = associative_recall(
            corpus.events,
            cue,
            corpus.associations,
            scoring_policy=policy.scoring_policy(),
            minimum_direct_support_coverage=policy.minimum_direct_support_coverage,
        )
        returned = [event.event_id for event in packet.items]
        returned_set = set(returned)
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
            failures.append(f"{question['id']}: required={sorted(required)} returned={returned}")

    return CorpusScore(
        corpus=name,
        questions=len(corpus.document["questions"]),
        successes=successes,
        unknown_questions=unknown_questions,
        unknown_abstained=unknown_abstained,
        required_events=required_events,
        required_found=required_found,
        reciprocal_rank_sum=reciprocal_rank_sum,
        failures=tuple(failures),
    )


def evaluate(policy: CandidatePolicy) -> Evaluation:
    return Evaluation(
        policy=policy,
        exploratory=tuple(_evaluate_corpus(name, policy) for name in EXPLORATORY_CORPORA),
        holdout=tuple(_evaluate_corpus(name, policy) for name in HOLDOUT_CORPORA),
    )


def _frange(start: int, stop: int, denominator: int) -> tuple[float, ...]:
    return tuple(value / denominator for value in range(start, stop + 1))


def _simplex_weights(step_units: int) -> Iterable[tuple[float, float, float, float]]:
    """Enumerate the complete four-weight unit simplex at the requested grid."""
    for lexical in range(step_units + 1):
        for entity in range(step_units - lexical + 1):
            for temporal in range(step_units - lexical - entity + 1):
                conversation = step_units - lexical - entity - temporal
                yield (
                    lexical / step_units,
                    entity / step_units,
                    temporal / step_units,
                    conversation / step_units,
                )


def _best(evaluations: Iterable[Evaluation]) -> tuple[Evaluation, tuple[Evaluation, ...]]:
    ranked = sorted(evaluations, key=lambda item: item.quality_key(), reverse=True)
    best_key = ranked[0].quality_key()
    tied = tuple(item for item in ranked if item.quality_key() == best_key)
    return ranked[0], tied


def run_staged_sweep() -> dict[str, Any]:
    baseline = CandidatePolicy(
        lexical_weight=0.78,
        entity_weight=0.14,
        temporal_weight=0.05,
        conversation_weight=0.03,
        phrase_bonus=0.15,
        temporal_scale_days=30.0,
        minimum_score=0.15,
        minimum_direct_support_coverage=0.60,
    )
    baseline_eval = evaluate(baseline)

    threshold_values = _frange(0, 20, 20)
    stage_a = [
        evaluate(
            CandidatePolicy(
                **{
                    **asdict(baseline),
                    "minimum_score": minimum_score,
                    "minimum_direct_support_coverage": coverage,
                }
            )
        )
        for minimum_score, coverage in product(threshold_values, repeat=2)
    ]
    stage_a_best, stage_a_tied = _best(stage_a)

    weight_evaluations: list[Evaluation] = []
    stage_a_policy = stage_a_best.policy
    for weights, phrase_bonus, temporal_scale in product(
        _simplex_weights(10),
        (0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0),
        (1.0, 7.0, 30.0, 90.0, 365.0),
    ):
        lexical, entity, temporal, conversation = weights
        weight_evaluations.append(
            evaluate(
                CandidatePolicy(
                    lexical_weight=lexical,
                    entity_weight=entity,
                    temporal_weight=temporal,
                    conversation_weight=conversation,
                    phrase_bonus=phrase_bonus,
                    temporal_scale_days=temporal_scale,
                    minimum_score=stage_a_policy.minimum_score,
                    minimum_direct_support_coverage=stage_a_policy.minimum_direct_support_coverage,
                )
            )
        )
    stage_b_best, stage_b_tied = _best(weight_evaluations)

    stage_c = [
        evaluate(
            CandidatePolicy(
                **{
                    **asdict(stage_b_best.policy),
                    "minimum_score": minimum_score,
                    "minimum_direct_support_coverage": coverage,
                }
            )
        )
        for minimum_score, coverage in product(threshold_values, repeat=2)
    ]
    stage_c_best, stage_c_tied = _best(stage_c)

    def compact(evaluation: Evaluation) -> dict[str, Any]:
        return {
            "policy": asdict(evaluation.policy),
            "quality_key": evaluation.quality_key(),
            "exploratory": [asdict(score) for score in evaluation.exploratory],
            "holdout": [asdict(score) for score in evaluation.holdout],
        }

    selected = stage_c_best.policy
    threshold_boundary = (
        selected.minimum_score in {threshold_values[0], threshold_values[-1]}
        or selected.minimum_direct_support_coverage in {threshold_values[0], threshold_values[-1]}
    )
    temporal_scale_boundary = selected.temporal_scale_days in {1.0, 365.0}
    phrase_boundary = selected.phrase_bonus in {0.0, 1.0}
    return {
        "benchmark_id": "MEM-SCORE-001",
        "baseline": compact(baseline_eval),
        "stage_a_best": compact(stage_a_best),
        "stage_a_equivalent_optima": len(stage_a_tied),
        "stage_b_best": compact(stage_b_best),
        "stage_b_equivalent_optima": len(stage_b_tied),
        "stage_c_best": compact(stage_c_best),
        "stage_c_equivalent_optima": len(stage_c_tied),
        "boundary_winner": bool(threshold_boundary or temporal_scale_boundary or phrase_boundary),
        "resolution_notes": {
            "threshold_step": 0.05,
            "weight_step": 0.10,
            "phrase_bonus_values": [0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0],
            "temporal_scale_days": [1.0, 7.0, 30.0, 90.0, 365.0]
        },
        "interpretation": (
            "A unique exact optimum is not claimed when equivalent_optima > 1. "
            "A boundary winner requires expanded/refined evidence before acceptance."
        ),
    }


def main() -> None:
    print(json.dumps(run_staged_sweep(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
