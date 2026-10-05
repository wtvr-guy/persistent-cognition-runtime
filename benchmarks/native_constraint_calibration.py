"""Collect target-host evidence for v0.7 environment-calibrated constraints.

This is a measurement harness, not an optimizer. It records host samples,
Ollama latency/output behavior across candidate token caps, the duration of the
frozen worker/runtime acceptance block, and the real-Ollama v2 response-pipeline
continuity probe. A pilot run remains evidence to inform calibration; it
does not automatically rewrite production policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import httpx

from prometheist.admission_diagnostics import RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX
from prometheist.attention_observation import SystemHostResourceProbe
from prometheist.native_policy import native_resource_safety_policy
from prometheist.ollama_runtime import (
    OllamaRuntimeProbe,
    configured_ollama_base_url,
    configured_ollama_model,
)

ROOT = Path(__file__).resolve().parents[1]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def measure_resources(samples: int) -> dict[str, Any]:
    probe = SystemHostResourceProbe()
    capture_started_at = datetime.now(timezone.utc)
    captured = [probe.capture() for _ in range(samples)]
    capture_completed_at = datetime.now(timezone.utc)
    policy = native_resource_safety_policy()
    return {
        "benchmark_id": "RES-NATIVE-001",
        "result": "PILOT_NATIVE_MEASUREMENT",
        "sample_count": samples,
        "capture_started_at": capture_started_at.isoformat(),
        "capture_completed_at": capture_completed_at.isoformat(),
        "policy_under_measurement": policy.model_dump(mode="json"),
        "platforms": sorted({item.platform for item in captured}),
        "logical_cpu_counts": sorted({item.logical_cpu_count for item in captured}),
        "cpu_utilization_percent": _summary(
            [float(item.cpu_utilization_percent) for item in captured]
        ),
        "memory_total_mib": sorted({item.memory_total_mib for item in captured}),
        "memory_available_mib": _summary(
            [float(item.memory_available_mib) for item in captured]
        ),
        "raw_samples": [item.model_dump(mode="json") for item in captured],
        "decision": (
            "Pilot observations establish actual target-host pressure ranges. Repeat under the "
            "idle, CPU-pressure, memory-pressure, cold-model, and warm-model scenarios before "
            "changing or verifying admission margins."
        ),
    }


def measure_pre_pipeline_resources() -> dict[str, Any]:
    """Capture host pressure immediately before the real v2 pipeline probe."""

    result = measure_resources(1)
    result["benchmark_id"] = "RES-PRE-PIPELINE-001"
    result["decision"] = (
        "This single immediate sample is ordered after Ollama calibration and directly before "
        "PIPELINE-NATIVE-001. Compare it with RES-NATIVE-001 and OLLAMA-RUNTIME-001 "
        "to distinguish physical free RAM from reusable resident-model memory."
    )
    return result


def measure_ollama(token_caps: tuple[int, ...]) -> dict[str, Any]:
    base_url = configured_ollama_base_url()
    model = configured_ollama_model()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    system = "Return valid JSON matching the schema. Preserve the requested literal exactly."
    literal = "VX-NATIVE-CALIBRATION-7F31A9"
    user = (
        "Return an answer that contains this exact literal and briefly states that this is a "
        f"native calibration probe: {literal}"
    )
    observations: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=300.0, trust_env=False) as client:
        for cap in token_caps:
            started = time.monotonic()
            error: str | None = None
            content = ""
            body: dict[str, Any] = {}
            try:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "format": schema,
                        "think": False,
                        "stream": False,
                        "options": {"num_predict": cap, "temperature": 0},
                    },
                )
                response.raise_for_status()
                body = response.json()
                content = str(body.get("message", {}).get("content", ""))
            except Exception as exc:  # measurement must preserve explicit failure evidence
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            observations.append(
                {
                    "num_predict": cap,
                    "elapsed_seconds": round(elapsed, 6),
                    "http_or_parse_error": error,
                    "content_length": len(content),
                    "literal_preserved": literal in content,
                    "done_reason": body.get("done_reason"),
                    "prompt_eval_count": body.get("prompt_eval_count"),
                    "eval_count": body.get("eval_count"),
                }
            )
    return {
        "benchmark_id": "LLM-NATIVE-001",
        "result": "PILOT_NATIVE_MEASUREMENT",
        "base_url": base_url,
        "model": model,
        "token_caps": list(token_caps),
        "observations": observations,
        "decision": (
            "This pilot identifies truncation/latency behavior for a controlled structured call. "
            "Repeat cold/warm/contention and production-schema scenarios before verifying HTTP, "
            "token, or retry bounds."
        ),
    }


def measure_ollama_runtime() -> dict[str, Any]:
    """Record whether the configured model is resident before the v2 pipeline probe."""

    policy = native_resource_safety_policy()
    state = OllamaRuntimeProbe().capture()
    return {
        "benchmark_id": "OLLAMA-RUNTIME-001",
        "result": "PILOT_NATIVE_MEASUREMENT" if state.probe_ok else "NO_NATIVE_EVIDENCE",
        "runtime_state": state.model_dump(mode="json"),
        "incremental_process_memory_mib": state.incremental_process_memory_mib(policy),
        "reusable_system_memory_credit_mib": state.reusable_memory_credit_mib(policy),
        "policy_under_measurement": policy.model_dump(mode="json"),
        "decision": (
            "The running-model observation determines the interaction's incremental RAM "
            "requirement before scheduler admission. A verified resident configured model uses "
            "ordinary worker overhead; a nonresident or failed probe uses the full cold-load "
            "budget. Worker claims recheck this state before every process launch."
        ),
    }


def measure_worker_runtime() -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "--locked",
        "pytest",
        "-q",
        "tests/test_v07_increment_h.py",
        "tests/test_worker_protocol.py::test_forced_process_loss_recovers_same_step_from_postgres",
        "tests/test_cli.py",
        "tests/test_cli_admission_diagnostics.py",
        "tests/test_admission_diagnostics.py",
        "tests/test_native_constraint_calibration.py",
        "tests/test_native_policy.py",
        "tests/test_ollama_runtime.py",
        "tests/test_interaction_resource_admission.py",
        "tests/test_percept_response_contract.py",
        "tests/test_user_prompt_worker_contract.py",
        "tests/test_percept_response_failures.py",
    ]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    return {
        "benchmark_id": "WORKER-NATIVE-001",
        "result": "PILOT_NATIVE_MEASUREMENT",
        "elapsed_seconds": round(elapsed, 6),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "decision": (
            "One frozen runtime-block duration is a pilot only. Lease/timeout/retry verification "
            "requires repeated normal, slow, killed, spawn-failure, and resource-denial trials."
        ),
    }


def _extract_admission_diagnostics(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX not in line:
            continue
        raw = line.split(RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX, 1)[1].strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"parse_error": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"parse_error": raw}
    return None


def measure_response_pipeline() -> dict[str, Any]:
    """Collect one real-model v2 pipeline trace without claiming an optimum."""

    command = [
        "uv",
        "run",
        "--locked",
        "pytest",
        "-q",
        "-s",
        "-m",
        "ollama",
        (
            "tests/test_acceptance_conversation_continuity.py::"
            "test_stateless_four_turn_continuity_survives_sessions_and_distractors"
        ),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    combined = f"{completed.stdout}\n{completed.stderr}"
    lowered = combined.casefold()
    executed = "1 passed" in lowered
    skipped = "1 skipped" in lowered
    admission_diagnostics = _extract_admission_diagnostics(combined)
    return {
        "benchmark_id": "PIPELINE-NATIVE-001",
        "result": "PILOT_NATIVE_MEASUREMENT" if executed else "NO_NATIVE_EVIDENCE",
        "elapsed_seconds": round(elapsed, 6),
        "returncode": completed.returncode,
        "acceptance_executed": executed,
        "acceptance_skipped": skipped,
        "admission_diagnostics": admission_diagnostics,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-4000:],
        "decision": (
            "A passing four-turn real-Ollama continuity run exercises the v2 Composer, "
            "deterministic Adaptive Recall, and final responder on the target host. It is one "
            "acceptance workload, not a general performance optimum."
        ),
    }


def _requested_measurements_complete(results: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    worker = results.get("WORKER-NATIVE-001")
    if worker is not None and worker.get("returncode") != 0:
        failures.append("WORKER-NATIVE-001 runtime block did not pass")

    ollama = results.get("LLM-NATIVE-001")
    if ollama is not None:
        observations = list(ollama.get("observations", []))
        if not any(item.get("http_or_parse_error") is None for item in observations):
            failures.append("LLM-NATIVE-001 produced no successful Ollama observation")

    runtime = results.get("OLLAMA-RUNTIME-001")
    if runtime is not None:
        state = dict(runtime.get("runtime_state") or {})
        if not bool(state.get("probe_ok")):
            failures.append("OLLAMA-RUNTIME-001 could not inspect running-model state")

    pipeline = results.get("PIPELINE-NATIVE-001")
    if pipeline is not None and not bool(pipeline.get("acceptance_executed")):
        failures.append("PIPELINE-NATIVE-001 real-Ollama acceptance did not execute and pass")

    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-samples", type=int, default=20)
    parser.add_argument(
        "--token-caps",
        default="16,24,32,48,64,128,256,512",
        help="comma-separated Ollama num_predict candidates",
    )
    parser.add_argument("--skip-resources", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-worker", action="store_true")
    parser.add_argument("--skip-response-pipeline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.resource_samples < 1:
        raise SystemExit("--resource-samples must be positive")
    token_caps = tuple(int(value) for value in args.token_caps.split(",") if value.strip())
    if not token_caps or any(value < 1 for value in token_caps):
        raise SystemExit("--token-caps must contain positive integers")

    results: dict[str, Any] = {}
    if not args.skip_resources:
        results["RES-NATIVE-001"] = measure_resources(args.resource_samples)
    if not args.skip_worker:
        results["WORKER-NATIVE-001"] = measure_worker_runtime()
    if not args.skip_ollama:
        results["LLM-NATIVE-001"] = measure_ollama(token_caps)
    if not args.skip_response_pipeline:
        results["OLLAMA-RUNTIME-001"] = measure_ollama_runtime()
        if not args.skip_resources:
            results["RES-PRE-PIPELINE-001"] = measure_pre_pipeline_resources()
        results["PIPELINE-NATIVE-001"] = measure_response_pipeline()

    complete, failures = _requested_measurements_complete(results)
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runner": "benchmarks/native_constraint_calibration.py",
        "requested_measurements_complete": complete,
        "completion_failures": failures,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not complete:
        raise SystemExit("native constraint calibration evidence is incomplete")


if __name__ == "__main__":
    main()
