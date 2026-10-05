from __future__ import annotations

import json
import uuid

import pytest

from prometheist import cli
from prometheist.admission_diagnostics import RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX


def test_cli_emits_structured_resource_diagnostics_for_admission_failure(
    monkeypatch,
    capsys,
):
    def fail_interaction(conn, user_text, conversation_id):
        del conn, user_text, conversation_id
        raise RuntimeError("interaction was not safely admitted to one assignment")

    scheduler = object()
    expected = {
        "resource_observation": {
            "host_metrics": {"memory_available_mib": 5_900},
        },
        "unassigned_tasks": [
            {
                "deficits_by_resource_class": {"MEMORY_RAM": 688},
            }
        ],
    }
    monkeypatch.setattr(cli, "handle_percept_in_worker_processes", fail_interaction)
    monkeypatch.setattr(cli, "load_scheduler", lambda conn: scheduler)
    monkeypatch.setattr(
        cli,
        "build_resource_admission_diagnostics",
        lambda value: expected if value is scheduler else None,
    )

    with pytest.raises(
        RuntimeError,
        match="interaction was not safely admitted to one assignment",
    ):
        cli._handle_with_admission_diagnostics(
            object(),
            "hello",
            uuid.uuid4(),
        )

    stderr = capsys.readouterr().err.strip()
    assert stderr.startswith(RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX)
    parsed = json.loads(stderr[len(RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX) :])
    assert parsed == {**expected, "kind": "SCHEDULER_ADMISSION_DENIED"}


def test_cli_emits_worker_claim_observation_when_launch_gate_denies(
    monkeypatch,
    capsys,
):
    class Observation:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {
                "decision": "DENIED",
                "reason": "claim resource 'host-ram-mib' lacks safe capacity",
            }

    class FakeWorkerLaunchDenied(RuntimeError):
        def __init__(self):
            super().__init__("worker claim denied")
            self.observation = Observation()

    def fail_interaction(conn, user_text, conversation_id):
        del conn, user_text, conversation_id
        raise FakeWorkerLaunchDenied()

    monkeypatch.setattr(cli, "WorkerLaunchDenied", FakeWorkerLaunchDenied)
    monkeypatch.setattr(cli, "handle_percept_in_worker_processes", fail_interaction)

    with pytest.raises(FakeWorkerLaunchDenied):
        cli._handle_with_admission_diagnostics(
            object(),
            "hello",
            uuid.uuid4(),
        )

    stderr = capsys.readouterr().err.strip()
    parsed = json.loads(stderr[len(RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX) :])
    assert parsed == {
        "kind": "WORKER_CLAIM_DENIED",
        "worker_claim_observation": {
            "decision": "DENIED",
            "reason": "claim resource 'host-ram-mib' lacks safe capacity",
        },
    }
