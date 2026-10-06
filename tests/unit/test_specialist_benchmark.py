from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import pytest

from prometheist.specialist_benchmark import (
    WorkerCase, capture_journal, digest, execute_case, export_reviewed, load_cases,
    summarize, write_json,
)

ROOT = Path(__file__).resolve().parents[2]


class ScriptedInference:
    """Harness test only: scripted results are never benchmark model scores."""
    def __init__(self, *outputs):
        self.outputs = iter(outputs)

    def request(self, request):
        return {"output": json.dumps(next(self.outputs)), "test_double": True}


def work_case(**updates):
    data = dict(case_id="work", group_id="work", split="train",
                kind="PRECOGNITIVE_USER_PROMPT_WORK",
                inputs={"percept": "Add two numbers.", "catalog": [
                    {"capability_id": "add", "kind": "TOOL", "description": "Add two numbers."}
                ]}, expected={"capability_indices": [0]}, score_fields=["capability_indices"])
    data.update(updates)
    return WorkerCase.model_validate(data)


PROFILE = {"id": "test-fp32", "model": "test"}


def test_worker_rejects_unavailable_capability_and_preserves_retry():
    record = execute_case(work_case(), ScriptedInference(
        {"capability_indices": [99]}, {"capability_indices": [0]}), PROFILE)
    assert record["correct"] is True
    assert [c["request"]["max_tokens"] for c in record["calls"]] == [48, 96]
    assert [c["validation"]["status"] for c in record["calls"]] == ["INVALID", "VALID"]
    assert "unavailable capability" in record["calls"][0]["validation"]["error"]


def test_worker_malformed_output_is_error_not_repaired():
    record = execute_case(work_case(), ScriptedInference(
        {"response_required": False}, {"response_required": False}), PROFILE)
    assert record["status"] == "ERROR"
    assert not record["correct"]


def test_deterministic_bypass_is_excluded_from_model_accuracy():
    case = work_case(inputs={"percept": "Add two numbers.", "catalog": []},
                     expected={"capability_indices": []})
    record = execute_case(case, ScriptedInference(), PROFILE)
    assert record["deterministic_bypass"]
    assert record["calls"] == []
    result = summarize([record])["by_worker"]["test-fp32/PRECOGNITIVE_USER_PROMPT_WORK/train"]
    assert result["contract_accuracy"] is None
    assert result["semantic_cases"] == 0


def test_wrong_semantic_choice_can_be_valid_schema():
    record = execute_case(work_case(), ScriptedInference({"capability_indices": []}), PROFILE)
    assert record["status"] == "VALID"
    assert record["correct"] is False


def test_triage_policy_ceiling_is_applied_by_real_worker():
    case = next(c for c in load_cases(ROOT / "benchmarks/specialists/worker_cases.jsonl")
                if c.kind == "PERCEPT_TRIAGE")
    record = execute_case(case, ScriptedInference({
        "task_required": True, "candidate_task_class": "CONSOLIDATE",
        "evidence_domains": ["DERIVED_INTERNAL"], "urgency_class": "ELEVATED",
    }), PROFILE)
    assert record["status"] == "ERROR"
    assert "outside source policy" in record["error"]["message"]


def test_review_uses_related_channel_bounds():
    case = next(c for c in load_cases(ROOT / "benchmarks/specialists/worker_cases.jsonl")
                if c.kind == "SELF_SCHEMA_REVIEW")
    record = execute_case(case, ScriptedInference({
        "opposition_indices": [99], "rationale": "contrary evidence",
    }), PROFILE)
    assert record["status"] == "ERROR"
    assert "unavailable related evidence index" in record["error"]["message"]


def test_no_group_leaks_across_splits(tmp_path):
    path = tmp_path / "cases.jsonl"
    cases = [work_case(), work_case(case_id="paraphrase", split="test")]
    path.write_text("\n".join(c.model_dump_json() for c in cases))
    with pytest.raises(ValueError, match="group crosses"):
        load_cases(path)


def test_duplicate_inputs_cannot_leak_by_changing_group(tmp_path):
    path = tmp_path / "cases.jsonl"
    cases = [work_case(), work_case(case_id="copy", group_id="copy", split="test")]
    path.write_text("\n".join(c.model_dump_json() for c in cases))
    with pytest.raises(ValueError, match="identical worker input"):
        load_cases(path)


def make_export_run(tmp_path, case):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "cases.jsonl").write_text(case.model_dump_json() + "\n")
    write_json(directory / "manifest.json", {"corpus_sha256": digest([case.model_dump(mode="json")])})
    record = execute_case(case, ScriptedInference({"capability_indices": [0]}), PROFILE)
    (directory / "results.jsonl").write_text(json.dumps(record) + "\n")
    reviews = tmp_path / "reviews.jsonl"
    review = {"case_sha256": digest(case.model_dump(mode="json")), "decision": "APPROVE",
              "reviewer": "test-reviewer", "label": {"capability_indices": [0]}}
    reviews.write_text(json.dumps(review) + "\n")
    return Namespace(run=directory, reviews=reviews, output=tmp_path / "training.jsonl")


def test_export_requires_review_and_revalidates_labels(tmp_path):
    args = make_export_run(tmp_path, work_case())
    export_reviewed(args)
    row = json.loads(args.output.read_text())
    assert row["label"] == {"capability_indices": [0]}
    assert row["worker_request"]["kind"] == "PRECOGNITIVE_USER_PROMPT_WORK"
    assert row["review"]["reviewer"] == "test-reviewer"


def test_export_rejects_held_out_data(tmp_path):
    args = make_export_run(tmp_path, work_case(split="test"))
    with pytest.raises(ValueError, match="held-out"):
        export_reviewed(args)
    assert not args.output.exists()


def test_export_detects_modified_output(tmp_path):
    args = make_export_run(tmp_path, work_case())
    path = args.run / "results.jsonl"
    value = json.loads(path.read_text())
    value["actual"] = {"capability_indices": []}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="result hash mismatch"):
        export_reviewed(args)


def test_captured_teacher_outputs_remain_unreviewed(tmp_path):
    source = tmp_path / "journal.jsonl"
    entry = {"artifact_type": "LLM_INVOCATION", "payload": {
        "kind": "V2_RESPONSE_POLICY", "output": "wrong but schema-valid output",
    }}
    source.write_text(json.dumps(entry) + "\n")
    destination = tmp_path / "captured.jsonl"
    capture_journal(Namespace(source=source, output=destination))
    captured = json.loads(destination.read_text())
    assert captured["artifact"] == entry
    assert captured["label_status"] == "UNREVIEWED_TEACHER_OUTPUT"


def test_functiongemma_native_parser_is_closed_and_nonexecuting():
    spec = importlib.util.spec_from_file_location("specialist_inference", ROOT / "scripts/specialist_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert json.loads(module.parse_function_selection(
        "<start_function_call>call:select_work{capability_indices:[0, 2]}<end_function_call>"
    )) == {"capability_indices": [0, 2]}
    for bad in ("call:delete_file{path:secret}", "{'capability_indices': [0]}",
                "<start_function_call>call:select_work{capability_indices:[__import__('os')]}<end_function_call>"):
        with pytest.raises(ValueError):
            module.parse_function_selection(bad)
    assert json.loads(module.parse_hammer_selection(
        '<tool_call>{"name":"select_work","arguments":{"capability_indices":[0]}}</tool_call>'
    )) == {"capability_indices": [0]}
    with pytest.raises(ValueError):
        module.parse_hammer_selection('{"name":"delete_file","arguments":{}}')


def test_precision_comparison_reports_changed_decision():
    case = work_case()
    a = execute_case(case, ScriptedInference({"capability_indices": [0]}), PROFILE)
    b = execute_case(case, ScriptedInference({"capability_indices": []}),
                     {"id": "test-fp16", "model": "test"})
    assert summarize([a, b])["precision_pairs"][0]["different"]
