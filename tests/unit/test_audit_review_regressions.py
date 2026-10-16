"""Regression coverage for the post-CI extraction audit review."""
from contextlib import nullcontext
import os
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from uuid import uuid4

import pytest

from persistent_cognition import api, percept_response_runtime as runtime
from persistent_cognition import postgres_scale_benchmark as benchmark
from persistent_cognition.response_policy import explicit_prior_assistant_reference


def test_respond_preserves_inherited_configuration_for_every_worker(monkeypatch):
    configuration = {
        "DATABASE_URL": "postgresql://app@localhost/pcr_test",
        "PCR_ARTIFACT_ROOT": "/app/artifacts",
        "PCR_GUI_JOB_CONFIG": api.AppSettings().model_dump_json(),
        "OPENAI_API_KEY": "test-credential",
        "APP_CUSTOM_SETTING": "application-value",
        "PCR_CAPABILITY_SNAPSHOT_SHA256": "stale-parent-digest",
    }
    for key, value in configuration.items():
        monkeypatch.setenv(key, value)
    inherited = dict(os.environ)
    launches = []

    class Launcher:
        def __init__(self, *args, **kwargs):
            pass

        def launch(self, **kwargs):
            launches.append(kwargs)
            return SimpleNamespace(process=SimpleNamespace(wait=lambda **kw: 0))

    interaction = SimpleNamespace(assignment_id=uuid4(), interaction_id=uuid4())
    monkeypatch.setattr(api, "scheduler_ownership", lambda *a: nullcontext())
    monkeypatch.setattr(runtime, "begin_percept", lambda *a, **kw: interaction)
    monkeypatch.setattr(runtime, "finish_percept", lambda *a, **kw: "response")
    monkeypatch.setattr(runtime, "GuardedWorkerLauncher", Launcher)
    state = SimpleNamespace(incremental_process_memory_mib=lambda _: 512)
    probe = SimpleNamespace(capture=lambda: state)
    command = ["application-worker", "--bootstrap"]
    assert api.respond(None, "hello", uuid4(), ollama_runtime_probe=probe,
                       worker_command=command) == "response"
    assert len(launches) == len(runtime.PERCEPT_STAGES)
    expected = {**inherited, "PCR_CAPABILITY_SNAPSHOT_SHA256": runtime.DEFAULT_REGISTRY.snapshot()["sha256"]}
    for launch in launches:
        assert launch["env"] == expected
        assert launch["command"] == command
    assert dict(os.environ) == inherited


@pytest.mark.parametrize("prompt", [
    "What is Persistent Cognition?",
    "How does persistent_cognition store facts?",
    "Persistent Cognition is a runtime.",
    "What did Persistent Cognitionish recommend?",
])
def test_project_name_alone_does_not_admit_prior_model_outputs(prompt):
    assert not explicit_prior_assistant_reference(prompt)


@pytest.mark.parametrize("confirmation", [None, "wrong_test"])
def test_benchmark_confirmation_rejects_before_database_access(monkeypatch, confirmation):
    import sys
    args = ["benchmark", "--database-url", "postgresql://app@localhost/pcr_test"]
    if confirmation is not None:
        args += ["--confirm-database", confirmation]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda *a: pytest.fail("database opened"))
    with pytest.raises(SystemExit) as failure:
        benchmark.main()
    assert failure.value.code == 2


def test_matching_benchmark_confirmation_reaches_execution(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["benchmark", "--database-url",
        "postgresql://app@localhost/pcr_test", "--confirm-database", "pcr_test",
        "--events", "100", "--base", "persona.json"])
    conn = object()
    monkeypatch.setattr(benchmark.psycopg, "connect", lambda *a: nullcontext(conn))
    monkeypatch.setattr(benchmark, "_require_benchmark_database", lambda c: "pcr_test")
    monkeypatch.setattr(benchmark, "load_document", lambda p: {"persona": "test"})
    calls = []
    monkeypatch.setattr(benchmark, "run_postgres_scale_document",
                        lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(benchmark, "_print_result", lambda _: None)
    benchmark.main()
    assert len(calls) == 1
    assert calls[0][0] == (conn, {"persona": "test"})
    assert calls[0][1]["target_event_count"] == 100


@pytest.mark.parametrize("mode", ["missing", "confirmed", "skip"])
def test_scale_wrapper_requires_explicit_confirmation_and_forwards_it(tmp_path, mode):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    # Copy the real wrapper into an isolated repository and intercept uv. This
    # exercises PowerShell parameter binding/control flow without a database reset.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[2] / "scripts/run_v05_scale.ps1"
    shutil.copyfile(source, scripts / source.name)
    report = tmp_path / "calls.json"
    arguments = "-SkipTests -DatabaseUrl 'postgresql://app@localhost/pcr_benchmark' -EventCounts 100"
    if mode == "confirmed":
        arguments += " -ConfirmDatabase 'pcr_benchmark'"
    elif mode == "skip":
        arguments += " -SkipPostgres"
    script_literal = str(scripts / source.name).replace("'", "''")
    report_literal = str(report).replace("'", "''")
    harness = tmp_path / "harness.ps1"
    harness.write_text(f"""
$ErrorActionPreference = 'Stop'
$global:AuditUvInvocations = [System.Collections.Generic.List[object]]::new()
function uv {{
    $global:AuditUvInvocations.Add(@($args))
    $global:LASTEXITCODE = 0
}}
$env:PCR_BENCHMARK_DATABASE_URL = 'parent-environment-value'
$failure = $null
try {{ & '{script_literal}' {arguments} }}
catch {{ $failure = $_.Exception.Message }}
@{{ calls = @($global:AuditUvInvocations.ToArray()); failure = $failure;
    databaseEnvironment = $env:PCR_BENCHMARK_DATABASE_URL }} |
    ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 '{report_literal}'
""", encoding="utf-8")
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-File", str(harness)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    assert data["databaseEnvironment"] == "parent-environment-value"
    if mode == "missing":
        assert "-ConfirmDatabase" in data["failure"]
        assert data["calls"] == []
    else:
        assert data["failure"] is None
        assert len(data["calls"]) == (3 if mode == "confirmed" else 2)
        postgres_calls = [c for c in data["calls"]
                          if "persistent_cognition.postgres_scale_benchmark" in c]
        if mode == "skip":
            assert postgres_calls == []
        else:
            assert len(postgres_calls) == 1
            call = postgres_calls[0]
            assert call[call.index("--confirm-database") + 1] == "pcr_benchmark"
            assert call[call.index("--database-url") + 1] == "postgresql://app@localhost/pcr_benchmark"
