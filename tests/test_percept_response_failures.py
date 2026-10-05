from __future__ import annotations

from types import SimpleNamespace
import uuid

import pytest

from prometheist.interaction_contracts import DurableInteraction
from prometheist.models import EventType
from prometheist.percept_response_runtime import PerceptStage
from prometheist import percept_response_runtime as runtime
from prometheist import percept_response_worker as worker


def test_final_response_event_failure_records_error_and_releases_claim(monkeypatch):
    ids = [uuid.uuid4() for _ in range(7)]
    interaction = DurableInteraction(
        interaction_id=ids[0],
        conversation_id=ids[1],
        correlation_id=ids[2],
        user_prompt_event_id=ids[3],
        before_global_seq=1,
        task_id=ids[4],
        assignment_id=ids[5],
        user_text="hello",
    )
    claim_id = ids[6]
    envelope = SimpleNamespace(
        step=SimpleNamespace(
            task_id=interaction.task_id,
            step_key=PerceptStage.PERSIST_RESULT.value,
        )
    )
    recorded_errors = []
    released = []

    monkeypatch.setattr(worker, "load_worker_claim_envelope", lambda *args, **kwargs: envelope)
    monkeypatch.setattr(worker, "load_interaction_by_task", lambda *args, **kwargs: interaction)
    monkeypatch.setattr(worker, "_ensure_percept_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker.artifact_journal,
        "load_stage_result_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        worker.artifact_journal,
        "write_stage_error_artifact",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        runtime,
        "_stage_result",
        lambda *args, **kwargs: {
            "response_required": True,
            "response_text": "completed response",
        },
    )

    def record_event(_conn, **kwargs):
        if kwargs["event_type"] is EventType.INTERACTION_RESPONSE:
            raise RuntimeError("final interaction event write failed")
        if kwargs["event_type"] is EventType.ERROR:
            recorded_errors.append(kwargs)
            return None
        raise AssertionError(f"unexpected event type: {kwargs['event_type']}")

    monkeypatch.setattr(worker.event_store, "record_event", record_event)
    monkeypatch.setattr(
        worker,
        "release_worker_claim",
        lambda *args, **kwargs: released.append(kwargs),
    )
    monkeypatch.setattr(
        worker,
        "complete_worker_claim",
        lambda *args, **kwargs: pytest.fail("failed stage must not complete its claim"),
    )

    with pytest.raises(RuntimeError, match="final interaction event write failed"):
        worker._execute_claimed_user_prompt_step(
            object(),
            object(),
            claim_id=claim_id,
            worker_id="persist-worker",
            scheduler_key="default",
        )

    assert recorded_errors[-1]["payload"] == {
        "stage": PerceptStage.PERSIST_RESULT.value,
        "error_type": "RuntimeError",
        "message": "final interaction event write failed",
    }
    assert released == [
        {
            "claim_id": claim_id,
            "worker_id": "persist-worker",
            "scheduler_key": "default",
        }
    ]
