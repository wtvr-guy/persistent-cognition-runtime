from types import SimpleNamespace
from uuid import uuid4

import pytest

from prometheist import artifact_journal, situation_runtime
from prometheist.situation_runtime import SituationStage


@pytest.mark.parametrize("journal_state", ["missing", "wrong-output", "wrong-reference", "matching"])
def test_situation_handoff_requires_agreement_with_independent_artifact(monkeypatch, journal_state):
    task = SimpleNamespace(task_id=uuid4(), assignment_id=uuid4())
    output = {"response_required": False, "response_text": None}
    refs = ["situation:original"]
    monkeypatch.setattr(
        situation_runtime, "load_worker_result",
        lambda *_args, **_kwargs: SimpleNamespace(output=output, output_refs=refs),
    )
    if journal_state != "missing":
        artifact_journal.write_stage_result_artifact(
            interaction_id=task.task_id, conversation_id=uuid4(), correlation_id=uuid4(),
            task_id=task.task_id, assignment_id=task.assignment_id, stage=SituationStage.PERSIST.value,
            output={"response_required": True, "response_text": "unexpected"} if journal_state == "wrong-output" else output,
            output_refs=["situation:unrelated"] if journal_state == "wrong-reference" else refs,
        )
    if journal_state == "matching":
        assert situation_runtime._require_situation_result(
            None, task, SituationStage.PERSIST, scheduler_key="test",
        ) == output
    else:
        with pytest.raises(RuntimeError, match="missing or conflicting situation stage artifact"):
            situation_runtime._require_situation_result(None, task, SituationStage.PERSIST, scheduler_key="test")
