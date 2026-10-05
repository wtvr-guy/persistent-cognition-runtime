from pathlib import Path

from prometheist.associative_benchmark import run_comparison


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "benchmarks" / "jordan_vale_v1.json"
ASSOCIATIONS = ROOT / "benchmarks" / "jordan_vale_associations_v1.json"


def test_associative_recall_improves_jordan_control_baseline():
    result = run_comparison(BENCHMARK, ASSOCIATIONS)

    assert result.baseline.successful_questions == 15
    assert result.baseline.question_success_rate == 15 / 18
    assert result.baseline.unknown_abstention_rate == 1.0

    assert result.associative.successful_questions == 18
    assert result.associative.question_success_rate == 1.0
    assert result.associative.evidence_recall == 1.0
    assert result.associative.unknown_abstention_rate == 1.0
    assert result.associative.failures == ()
