from __future__ import annotations

from benchmarks import native_constraint_calibration as calibration
from prometheist.admission_diagnostics import RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX


def test_extract_admission_diagnostics_returns_structured_payload():
    text = (
        "before\n"
        + RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX
        + '{"resource_observation":{"healthy":true},"unassigned_tasks":[]}\n'
        + "after\n"
    )

    assert calibration._extract_admission_diagnostics(text) == {
        "resource_observation": {"healthy": True},
        "unassigned_tasks": [],
    }


def test_pre_pipeline_resource_probe_is_a_single_relabelled_sample(monkeypatch):
    calls = []

    def fake_measure_resources(samples: int):
        calls.append(samples)
        return {
            "benchmark_id": "RES-NATIVE-001",
            "result": "PILOT_NATIVE_MEASUREMENT",
            "sample_count": samples,
        }

    monkeypatch.setattr(calibration, "measure_resources", fake_measure_resources)

    result = calibration.measure_pre_pipeline_resources()

    assert calls == [1]
    assert result["benchmark_id"] == "RES-PRE-PIPELINE-001"
    assert result["sample_count"] == 1
    assert "directly before" in result["decision"]
