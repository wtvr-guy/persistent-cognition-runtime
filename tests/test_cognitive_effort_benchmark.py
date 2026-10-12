"""Database-free unit tests for the auditable cognitive-effort suite."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.cognitive_effort.energy import energy_joules, read_power_samples
from benchmarks.cognitive_effort.measurement import AuditTrail, run_trial, verify_audit
from benchmarks.cognitive_effort.report import report
from benchmarks.cognitive_effort.runner import _local_base_url, _model_call
from benchmarks.cognitive_effort.spec import CASES, fixture_digest
from persistent_cognition.model_admission import classify_task


def test_frozen_classifier_oracle_matches_current_registered_code():
    assert len({case.case_id for case in CASES}) == len(CASES) == 16
    assert len(fixture_digest()) == 64
    for case in CASES:
        actual, _reason = classify_task(case.text)
        assert actual == case.expected, case.case_id


def test_audit_roundtrip_tracks_exact_input_and_success(tmp_path: Path):
    audit = AuditTrail(tmp_path / "run")
    audit.artifact("manifest.json", {"fixed": "baseline"})
    audit.append("run_start", run_id="fixed")
    result = run_trial(
        audit, "test_01", "simple",
        {"input": "ABC123", "expected": "coding"},
        lambda: {"passed": True, "actual": "coding"},
        sample_seconds=0.05,
    )
    audit.append("run_end", issue_count=0, completed=True)
    audit.close()
    assert result["status"] == "PASS"
    assert result["case"]["input"] == "ABC123"
    assert result["elapsed_ns"] >= 0
    assert verify_audit(tmp_path / "run")["valid"]
    report_payload = report(tmp_path / "run")
    assert report_payload["phases"]["simple"]["passes"] == 1
    assert report_payload["electric_energy"]["status"] == "NOT_MEASURED"
    assert (tmp_path / "run" / "summary.csv").exists()


def test_exception_traceback_is_retained_in_trial_artifact(tmp_path: Path):
    audit = AuditTrail(tmp_path / "run")
    audit.append("run_start", run_id="fault")
    def explode() -> dict:
        raise RuntimeError("synthetic-failure-123")
    result = run_trial(
        audit, "failing_01", "fault", {"input": "fault"},
        explode, sample_seconds=0.05,
    )
    audit.append("run_end", issue_count=1, completed=True)
    audit.close()
    assert result["status"] == "ERROR"
    assert result["failure"]["type"] == "RuntimeError"
    assert "explode" in result["failure"]["traceback"]
    assert verify_audit(tmp_path / "run")["valid"]
    assert report(tmp_path / "run")["overall_pass"] is False


def test_tampering_is_detected(tmp_path: Path):
    audit = AuditTrail(tmp_path / "run")
    audit.artifact("manifest.json", {"fixed": True})
    audit.append("run_start", run_id="tamper")
    audit.append("run_end", issue_count=0, completed=True)
    audit.close()
    target = tmp_path / "run" / "manifest.json"
    target.write_text('{"fixed":false}', encoding="utf-8")
    result = verify_audit(tmp_path / "run")
    assert not result["valid"]
    assert "digest mismatch" in result["errors"][0]


def test_interrupted_run_is_explicitly_invalid(tmp_path: Path):
    audit = AuditTrail(tmp_path / "run")
    audit.append("run_start", run_id="interrupted")
    audit.append("trial_start", trial_id="partial", phase="unfinished")
    audit.close()
    result = verify_audit(tmp_path / "run")
    assert not result["valid"]
    assert any("incomplete" in error for error in result["errors"])


def test_malformed_audit_chain_is_detected(tmp_path: Path):
    audit = AuditTrail(tmp_path / "run")
    audit.append("run_start", run_id="broken")
    audit.append("run_end", completed=True)
    audit.close()
    stream = tmp_path / "run" / "events.jsonl"
    lines = stream.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["fields"]["completed"] = False
    lines[1] = json.dumps(entry)
    stream.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_audit(tmp_path / "run")["valid"]


def test_energy_from_constant_power_samples_is_exact():
    assert energy_joules([(10.0, 100.0), (11.0, 100.0), (12.0, 100.0)], 10, 12) == 200
    assert energy_joules([(10.0, 0.0), (11.0, 100.0), (12.0, 100.0)], 10, 12) == 150
    assert energy_joules([(10.0, 0.0), (11.0, 100.0), (12.0, 100.0)], 10.5, 11.5) == 87.5


def test_energy_rejects_out_of_range_or_unobserved_gaps():
    samples = [(10.0, 50.0), (20.0, 60.0)]
    with pytest.raises(ValueError, match="does not cover"):
        energy_joules(samples, 9.0, 11.0)
    with pytest.raises(ValueError, match="sampling gap"):
        energy_joules(samples, 11, 19, max_gap_seconds=2)


def test_power_csv_requires_utc_and_increasing_timestamps(tmp_path: Path):
    power_file = tmp_path / "power.csv"
    power_file.write_text(
        "timestamp_utc,power_watts\n"
        "2026-10-12T00:00:00Z,25\n"
        "2026-10-12T00:00:01+00:00,35\n", encoding="utf-8",
    )
    assert len(read_power_samples(power_file)) == 2
    power_file.write_text(
        "timestamp_utc,power_watts\n2026-10-12T00:00:00,25\n"
        "2026-10-12T00:00:01,35\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires UTC offset"):
        read_power_samples(power_file)


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:11434", "http://example.com:11434",
    "http://127.0.0.1:11434/admin", "http://user:pass@127.0.0.1:11434",
])
def test_model_benchmark_refuses_nonlocal_or_unexpected_endpoints(url: str):
    with pytest.raises(ValueError, match="local plaintext HTTP"):
        _local_base_url(url)


def test_model_response_requires_exact_schema(monkeypatch):
    from benchmarks.cognitive_effort import runner
    class FakeClient:
        pass
    def fake_request(_client, _method, _route, **_kwargs):
        return {"message": {"content": '{"task":"coding","surprise":"extra"}'}}
    monkeypatch.setattr(runner, "_request", fake_request)
    with pytest.raises(ValueError, match="only the task"):
        _model_call(FakeClient(), "test-model", "/code example")


def test_trial_files_are_not_overwritten(tmp_path: Path):
    audit = AuditTrail(tmp_path / "run")
    audit.append("run_start", run_id="repeat")
    run_trial(audit, "unique", "x", {}, lambda: {"passed": True}, sample_seconds=.05)
    with pytest.raises(FileExistsError):
        run_trial(audit, "unique", "x", {}, lambda: {"passed": True}, sample_seconds=.05)
    audit.close()
