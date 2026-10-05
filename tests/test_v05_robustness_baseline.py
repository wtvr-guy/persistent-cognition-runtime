from pathlib import Path

from prometheist.derived_associative_benchmark import run_derived_benchmark


ROOT = Path(__file__).resolve().parent.parent
MORGAN = ROOT / "benchmarks" / "morgan_reyes_v05_robustness.json"


def test_v05_lifecycle_routing_closes_the_frozen_robustness_failures():
    result = run_derived_benchmark(MORGAN)

    assert result.question_count == 7
    assert result.successful_questions == 7
    assert result.question_success_rate == 1.0
    assert result.evidence_recall == 1.0
    assert result.unknown_abstention_rate == 1.0
    assert result.failures == ()
