from pathlib import Path

import pytest

from prometheist.scale_benchmark import run_scale_document
from prometheist.scale_corpus import load_document


ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"


@pytest.mark.parametrize(
    ("filename", "expected_questions"),
    [
        ("jordan_vale_v1.json", 18),
        ("avery_chen_v1.json", 6),
        ("morgan_reyes_v05_robustness.json", 7),
    ],
)
def test_scale_runner_reproduces_base_results_after_uuid_remap(
    filename,
    expected_questions,
):
    base = load_document(BENCHMARKS / filename)
    result = run_scale_document(
        base,
        target_event_count=len(base["events"]),
    )

    assert result.event_count == len(base["events"])
    assert result.benchmark.question_count == expected_questions
    assert result.benchmark.successful_questions == expected_questions
    assert result.benchmark.question_success_rate == 1.0
    assert result.benchmark.evidence_recall == 1.0
    assert result.benchmark.unknown_abstention_rate == 1.0
    assert result.benchmark.failures == ()
