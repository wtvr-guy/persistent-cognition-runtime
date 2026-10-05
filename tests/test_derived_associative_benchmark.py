from pathlib import Path

from prometheist.derived_associative_benchmark import run_derived_benchmark
from prometheist.synthetic_benchmark import run_benchmark


ROOT = Path(__file__).resolve().parent.parent
JORDAN = ROOT / "benchmarks" / "jordan_vale_v1.json"
AVERY = ROOT / "benchmarks" / "avery_chen_v1.json"


def test_derived_associations_replace_curated_jordan_edges():
    result = run_derived_benchmark(JORDAN)

    assert result.successful_questions == 18
    assert result.question_success_rate == 1.0
    assert result.evidence_recall == 1.0
    assert result.unknown_abstention_rate == 1.0
    assert result.failures == ()


def test_same_rules_generalize_to_held_out_avery_persona():
    baseline = run_benchmark(AVERY)
    derived = run_derived_benchmark(AVERY)

    assert derived.successful_questions == 6
    assert derived.question_success_rate == 1.0
    assert derived.evidence_recall == 1.0
    assert derived.unknown_abstention_rate == 1.0
    assert derived.failures == ()
    assert derived.successful_questions > baseline.successful_questions
