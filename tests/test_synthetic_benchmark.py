import json
from pathlib import Path

from prometheist.synthetic_benchmark import run_benchmark


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "jordan_vale_v1.json"


def test_synthetic_life_benchmark_sets_a_retrieval_regression_floor():
    result = run_benchmark(BENCHMARK_PATH)

    # v0.2 is intentionally lexical/cue based rather than a finished
    # associative memory system. These floors prevent regressions while
    # preserving difficult failures for v0.3 to solve.
    assert result.question_success_rate >= 0.75, result.failures
    assert result.evidence_recall >= 0.80, result.failures
    assert result.unknown_abstention_rate == 1.0, result.failures


def test_unknown_abstention_is_open_world_absence_not_null_field():
    document = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    oracle = document["persona"]["oracle"]
    unknown_questions = [
        question
        for question in document["questions"]
        if question.get("expect_no_evidence", False)
    ]

    assert len(unknown_questions) == 1
    assert unknown_questions[0]["query"] == "What is Jordan's blood type?"

    # The evaluator must not encode the answer as an explicit null/unknown
    # profile field. The queried property is absent from the profile entirely.
    assert "blood_type" not in oracle
    assert "blood type" not in {str(key).replace("_", " ").casefold() for key in oracle}

    # More importantly, the memory corpus itself contains no representation of
    # the queried fact: no event text, payload key, or payload value mentions it.
    serialized_events = json.dumps(document["events"], sort_keys=True).casefold()
    assert "blood type" not in serialized_events
    assert "blood_type" not in serialized_events
