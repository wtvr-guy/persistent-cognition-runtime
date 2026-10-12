"""Audited cognitive-effort measurements on the operator's own hardware.

No database writes, production policy changes, remote services or model downloads
are performed by the default suite. The optional native gate uses a separately
confirmed disposable TEST_DATABASE_URL and is intentionally explicit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import psutil

from benchmarks.cognitive_effort.measurement import (
    AUDIT_VERSION, AuditTrail, run_trial, verify_audit, utc_now,
)
from benchmarks.cognitive_effort.report import report
from benchmarks.cognitive_effort.spec import (
    CASES, CLASSIFICATION_SCHEMA, SUITE_VERSION, classification_instruction,
    fixture_digest, vars_case,
)

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "benchmarks" / "generated" / "cognitive_effort"
BENCHMARK_SEED = 104729


def _git(command: str) -> str:
    completed = subprocess.run(
        ["git", *command.split()], cwd=ROOT, capture_output=True, text=True,
        timeout=15, check=True,
    )
    return completed.stdout.strip()


def _versioned_environment(allow_dirty: bool) -> dict:
    commit = _git("rev-parse HEAD")
    branch = _git("branch --show-current")
    dirty_lines = _git("status --porcelain")
    if dirty_lines and not allow_dirty:
        raise ValueError(
            "git working tree is dirty. Commit/stash changes or use --allow-dirty "
            "and inspect git_status in manifest. Benchmark results will not be comparable."
        )
    return {
        "commit": commit, "branch": branch, "clean": not bool(dirty_lines),
        "git_status": dirty_lines.splitlines(),
        "python": sys.version, "platform": platform.platform(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "psutil_version": psutil.__version__,
        "httpx_version": httpx.__version__,
    }


def _local_base_url(raw: str) -> str:
    parsed = urlparse(raw.rstrip("/"))
    if (
        parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("benchmark Ollama endpoint must be local plaintext HTTP without credentials/path")
    return raw.rstrip("/")


def _canonical_model(value: str) -> str:
    name = value.strip().casefold()
    if not name:
        raise ValueError("model must be nonempty")
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


def _request(client: httpx.Client, method: str, route: str, **kwargs) -> dict:
    response = client.request(method, route, **kwargs)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError(f"Ollama {route} returned non-object JSON")
    return value


def _resident(client: httpx.Client, model: str) -> tuple[bool, list[dict]]:
    response = _request(client, "GET", "/api/ps")
    models = response.get("models")
    if not isinstance(models, list):
        raise ValueError("Ollama /api/ps models must be a list")
    details = [
        {
            "name": item.get("name"),
            "model": item.get("model"),
            "size": item.get("size"),
            "size_vram": item.get("size_vram"),
            "expires_at": item.get("expires_at"),
            "digest": item.get("digest"),
        }
        for item in models if isinstance(item, dict)
    ]
    exact = sum(
        _canonical_model(str(item.get(key, ""))) == _canonical_model(model)
        for item in details for key in ("name",)
        if item.get(key)
    )
    return exact == 1, details


def _model_call(client: httpx.Client, model: str, text: str) -> dict:
    response = _request(
        client, "POST", "/api/chat", json={
            "model": model,
            "messages": [
                {"role": "system", "content": classification_instruction()},
                {"role": "user", "content": text},
            ],
            "format": CLASSIFICATION_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": "1h",
            "options": {"temperature": 0, "num_predict": 40, "num_ctx": 2048},
        },
    )
    message = response.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Ollama /api/chat has no string message.content")
    raw_text = message["content"]
    decoded = json.loads(raw_text)
    if not isinstance(decoded, dict) or set(decoded) != {"task"}:
        raise ValueError("Ollama structured response must have only the task field")
    label = decoded["task"]
    if label not in {"general", "coding"}:
        raise ValueError("Ollama task output is not in the closed enum")
    metric_keys = (
        "total_duration", "load_duration", "prompt_eval_count",
        "prompt_eval_duration", "eval_count", "eval_duration", "done_reason",
    )
    return {
        "label": label,
        "exact_message_content": raw_text,
        "content_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "model_metrics": {key: response.get(key) for key in metric_keys},
        "reported_model": response.get("model"),
        "hidden_thinking_not_persisted": True,
    }


def _disposable_test_database() -> dict:
    raw = os.environ.get("TEST_DATABASE_URL", "")
    if not raw:
        raise ValueError("--with-native requires TEST_DATABASE_URL")
    parsed = urlparse(raw)
    db_name = parsed.path.lstrip("/").lower()
    if parsed.scheme not in {"postgresql", "postgres"} or not db_name:
        raise ValueError("TEST_DATABASE_URL must be a PostgreSQL connection URL")
    if "test" not in db_name and "benchmark" not in db_name:
        raise ValueError("refusing native gate: database name must contain test or benchmark")
    production = os.environ.get("DATABASE_URL")
    if production:
        prod = urlparse(production)
        if (
            (prod.hostname or "").casefold() == (parsed.hostname or "").casefold()
            and (prod.port or 5432) == (parsed.port or 5432)
            and prod.path == parsed.path
        ):
            raise ValueError("TEST_DATABASE_URL and DATABASE_URL point to the same database")
    return {"name": db_name, "url_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def _native_gate(audit: AuditTrail, sample_seconds: float, timeout: float, model: str) -> dict:
    db = _disposable_test_database()
    environment = dict(os.environ)
    environment["REQUIRE_OLLAMA_ACCEPTANCE"] = "1"
    environment["REQUIRE_V07_LOCAL_ACCEPTANCE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PCR_OLLAMA_KEEP_ALIVE"] = "1h"
    environment["OLLAMA_MODEL"] = model
    environment["PCR_ARTIFACT_ROOT"] = str(audit.root / "native_artifacts")
    (audit.root / "native_artifacts").mkdir()
    command = [sys.executable, "-m", "pytest", "-vv", "-s", "-ra", "-m", "ollama"]
    def action() -> dict:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            errors="replace", timeout=timeout, check=False,
        )
        audit.artifact("native_output.json", {
            "argv": command, "stdout": completed.stdout, "stderr": completed.stderr,
            "exit_code": completed.returncode,
            "TEST_DATABASE_URL": db,
        })
        return {
            "passed": completed.returncode == 0,
            "exit_code": completed.returncode,
            "output_artifact": "native_output.json",
            "note": "Pytest exit zero is structural, not proof of natural-language correctness.",
        }
    return run_trial(
        audit, "native_ollama_acceptance", "native_ollama_acceptance",
        {"database": db, "command": command}, action, sample_seconds=sample_seconds,
    )


def _run(args: argparse.Namespace) -> int:
    if args.with_native and not args.with_ollama:
        raise ValueError("--with-native requires --with-ollama")
    if args.cold_repeats and not args.with_ollama:
        raise ValueError("--cold-repeats requires --with-ollama")
    if not 50 <= args.sample_ms <= 5000:
        raise ValueError("--sample-ms must be between 50 and 5000")
    for name in ("deterministic_iterations", "deterministic_repeats", "retrieval_repeats",
                 "warm_repeats"):
        if not 1 <= getattr(args, name) <= 100000:
            raise ValueError(f"{name} must be in 1..100000")
    if not 0 <= args.cold_repeats <= 20:
        raise ValueError("cold_repeats must be in 0..20")
    if args.timeout_seconds <= 0 or args.native_timeout_seconds <= 0:
        raise ValueError("timeouts must be positive")
    environment = _versioned_environment(args.allow_dirty)
    if args.expected_commit and environment["commit"] != args.expected_commit:
        raise ValueError(
            f"wrong checkout: expected {args.expected_commit} "
            f"but found {environment['commit']}"
        )
    model = args.model or os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    url = _local_base_url(args.ollama_url)
    native_db = _disposable_test_database() if args.with_native else None
    output_root = args.output_root.resolve() if args.output_root else RESULT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]
    audit = AuditTrail(output_root / run_id)
    summary_path = audit.root / "summary.json"
    issues: list[dict] = []
    sample_seconds = args.sample_ms / 1000
    try:
        manifest = {
            "suite": SUITE_VERSION, "audit_version": AUDIT_VERSION,
            "created_utc": utc_now(), "run_id": run_id,
            "environment": environment, "fixture_sha256": fixture_digest(),
            "case_count": len(CASES), "seed": BENCHMARK_SEED,
            "retrieval_fixture_sha256": hashlib.sha256(
                (ROOT / "benchmarks" / "jordan_vale_v1.json").read_bytes()).hexdigest(),
            "association_fixture_sha256": hashlib.sha256(
                (ROOT / "benchmarks" / "jordan_vale_associations_v1.json").read_bytes()
            ).hexdigest(),
            "options": {
                "deterministic_iterations": args.deterministic_iterations,
                "deterministic_repeats": args.deterministic_repeats,
                "retrieval_repeats": args.retrieval_repeats,
                "warm_repeats": args.warm_repeats,
                "cold_repeats": args.cold_repeats,
                "sample_ms": args.sample_ms,
                "with_ollama": args.with_ollama, "with_native": args.with_native,
                "model": model if args.with_ollama else None,
                "ollama_url": url if args.with_ollama else None,
                "native_database": native_db,
            },
            "limitations": [
                "No electricity measurement without an external timestamped power meter",
                "Named Ollama process RSS/CPU are heuristic and may miss runner processes",
                "Microbenchmarks do not replace end-to-end PCR native acceptance",
                "Model calls here use synthetic local classifier tasks, not production policy routing",
                "Measurements include sampler interference, external load and timing noise",
            ],
        }
        audit.artifact("manifest.json", manifest)
        audit.append("run_start", run_id=run_id, fixture_sha256=fixture_digest())
        from persistent_cognition.model_admission import classify_task
        from persistent_cognition.associative_benchmark import run_associative_benchmark
        from persistent_cognition.synthetic_benchmark import run_benchmark
        for repeat in range(args.deterministic_repeats):
            for case in CASES:
                trial_id = f"det_{repeat:03d}_{case.case_id}"
                def action(case=case) -> dict:
                    mismatches = 0
                    actual = None
                    for _ in range(args.deterministic_iterations):
                        actual, _ = classify_task(case.text)
                        mismatches += actual != case.expected
                    return {
                        "passed": mismatches == 0,
                        "expected": case.expected, "actual": actual,
                        "mismatch_count": mismatches,
                        "iterations": args.deterministic_iterations,
                    }
                output = run_trial(
                    audit, trial_id, "deterministic_classifier", vars_case(case),
                    action, sample_seconds=sample_seconds,
                )
                if output["status"] != "PASS":
                    issues.append({"trial": trial_id, "status": output["status"]})

        fixture_path = ROOT / "benchmarks" / "jordan_vale_v1.json"
        association_path = ROOT / "benchmarks" / "jordan_vale_associations_v1.json"
        for repeat in range(args.retrieval_repeats):
            for phase, expected, fn in (
                ("lexical_retrieval", 15, lambda: run_benchmark(fixture_path)),
                ("associative_retrieval", 18, lambda: run_associative_benchmark(
                    fixture_path, association_path)),
            ):
                trial_id = f"{phase}_{repeat:03d}"
                def action(fn=fn, expected=expected) -> dict:
                    measured = asdict(fn())
                    return {
                        "passed": measured["successful_questions"] == expected
                        and measured["unknown_abstention_rate"] == 1.0,
                        "expected_successes": expected, "oracle": measured,
                    }
                output = run_trial(
                    audit, trial_id, phase,
                    {"synthetic_fixture_sha256": fixture_digest(),
                     "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                     "association_sha256": hashlib.sha256(association_path.read_bytes()).hexdigest(),
                     "fixture_files": [str(fixture_path.relative_to(ROOT)),
                                       str(association_path.relative_to(ROOT))]},
                    action, sample_seconds=sample_seconds,
                )
                if output["status"] != "PASS":
                    issues.append({"trial": trial_id, "status": output["status"]})

        if args.with_ollama:
            with httpx.Client(base_url=url, timeout=args.timeout_seconds, trust_env=False) as client:
                version = _request(client, "GET", "/api/version")
                audit.append("ollama_version", version=version, model=model)
                # A different resident model must not be silently unloaded by this suite.
                for repeat in range(args.cold_repeats):
                    unload_id = f"unload_{repeat:03d}"
                    def unload() -> dict:
                        _request(client, "POST", "/api/generate", json={
                            "model": model, "keep_alive": 0, "stream": False,
                        })
                        resident, observed = _resident(client, model)
                        return {"passed": not resident, "verified_not_resident": not resident,
                                "resident_models": observed}
                    unloaded = run_trial(
                        audit, unload_id, "ollama_unload",
                        {"model": model}, unload, sample_seconds=sample_seconds,
                    )
                    if unloaded["status"] != "PASS":
                        raise RuntimeError("failed to verify target Ollama model was unloaded")
                    anchor = CASES[1]
                    for phase in ("ollama_cold", "ollama_warm_paired"):
                        was_resident, observations = _resident(client, model)
                        audit.append(
                            "residency_observation", phase=phase, repeat=repeat,
                            model=model, resident=was_resident, models=observations,
                        )
                        if was_resident != (phase == "ollama_warm_paired"):
                            raise RuntimeError(f"unexpected model residency before {phase}")
                        trial_id = f"{phase}_{repeat:03d}_{anchor.case_id}"
                        def action(anchor=anchor) -> dict:
                            response = _model_call(client, model, anchor.text)
                            return {
                                **response, "expected": anchor.expected,
                                "passed": response["label"] == anchor.expected,
                            }
                        output = run_trial(
                            audit, trial_id, phase, vars_case(anchor), action,
                            sample_seconds=sample_seconds,
                        )
                        if output["status"] != "PASS":
                            issues.append({"trial": trial_id, "status": output["status"]})
                if args.cold_repeats == 0:
                    def prewarm() -> dict:
                        _request(client, "POST", "/api/generate", json={
                            "model": model, "prompt": "", "stream": False, "keep_alive": "1h",
                        })
                        resident, observed = _resident(client, model)
                        return {"passed": resident, "verified_resident": resident,
                                "resident_models": observed}
                    prewarmed = run_trial(
                        audit, "ollama_prewarm_000", "ollama_prewarm",
                        {"model": model}, prewarm, sample_seconds=sample_seconds,
                    )
                    if prewarmed["status"] != "PASS":
                        raise RuntimeError("Ollama prewarm did not verify model residency")
                for repeat in range(args.warm_repeats):
                    shuffled = list(CASES)
                    random.Random(BENCHMARK_SEED + repeat).shuffle(shuffled)
                    for case in shuffled:
                        resident, observations = _resident(client, model)
                        if not resident:
                            raise RuntimeError("warm benchmark model is not resident")
                        audit.append(
                            "residency_observation", phase="ollama_warm", repeat=repeat,
                            case_id=case.case_id, model=model, resident=True, models=observations,
                        )
                        trial_id = f"warm_{repeat:03d}_{case.case_id}"
                        def action(case=case) -> dict:
                            response = _model_call(client, model, case.text)
                            return {
                                **response, "expected": case.expected,
                                "passed": response["label"] == case.expected,
                            }
                        output = run_trial(
                            audit, trial_id, "ollama_warm_classifier",
                            vars_case(case), action, sample_seconds=sample_seconds,
                        )
                        if output["status"] != "PASS":
                            issues.append({"trial": trial_id, "status": output["status"]})
            if args.with_native:
                output = _native_gate(audit, sample_seconds, args.native_timeout_seconds, model)
                if output["status"] != "PASS":
                    issues.append({"trial": "native_ollama_acceptance", "status": output["status"]})
        audit.append("run_end", issue_count=len(issues), completed=True)
    except BaseException as exc:
        import traceback
        audit.append(
            "run_error", error_type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        audit.close()
        print(f"Benchmark artifacts: {audit.root}", flush=True)
    final = report(audit.root)
    print(f"Audit integrity: {final['audit']['valid']}; all passed: {final['overall_pass']}")
    for phase, result in final["phases"].items():
        print(
            f"{phase}: {result['passes']}/{result['trial_count']} passed; "
            f"median {result['duration_ms_median']} ms"
        )
    print(f"Summary: {summary_path}")
    return 1 if issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run fresh local benchmarks")
    run.add_argument("--with-ollama", action="store_true")
    run.add_argument("--with-native", action="store_true")
    run.add_argument("--model")
    run.add_argument("--ollama-url", default=os.environ.get(
        "OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    run.add_argument("--cold-repeats", type=int, default=0)
    run.add_argument("--warm-repeats", type=int, default=2)
    run.add_argument("--deterministic-iterations", type=int, default=2000)
    run.add_argument("--deterministic-repeats", type=int, default=3)
    run.add_argument("--retrieval-repeats", type=int, default=3)
    run.add_argument("--sample-ms", type=int, default=250)
    run.add_argument("--timeout-seconds", type=float, default=300)
    run.add_argument("--native-timeout-seconds", type=float, default=1800)
    run.add_argument("--expected-commit")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("--output-root", type=Path)
    check = commands.add_parser("verify", help="Verify audit chain and trial artifacts")
    check.add_argument("--run-dir", type=Path, required=True)
    summarize = commands.add_parser("report", help="Recompute report from verified trial artifacts")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--power-csv", type=Path)
    summarize.add_argument("--meter-scope", choices=("wall", "cpu_package", "device"),
                           default="wall")
    summarize.add_argument("--idle-watts", type=float)
    summarize.add_argument("--max-gap-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.command == "run":
        return _run(args)
    if args.command == "verify":
        outcome = verify_audit(args.run_dir)
        print(json.dumps(outcome, indent=2))
        return 0 if outcome["valid"] else 2
    outcome = report(
        args.run_dir, power_csv=args.power_csv, idle_watts=args.idle_watts,
        meter_scope=args.meter_scope, max_power_gap_seconds=args.max_gap_seconds,
    )
    print(json.dumps(
        {"audit": outcome["audit"], "overall_pass": outcome["overall_pass"],
         "phases": outcome["phases"], "energy": outcome["electric_energy"]},
        indent=2,
    ))
    return 0 if outcome["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
