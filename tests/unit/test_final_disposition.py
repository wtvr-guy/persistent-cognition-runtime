from uuid import uuid4

import pytest

from prometheist import artifact_journal


def _completed_stage():
    identity = {
        "interaction_id": uuid4(),
        "conversation_id": uuid4(),
        "correlation_id": uuid4(),
        "task_id": uuid4(),
        "assignment_id": uuid4(),
    }
    artifact_journal.write_stage_result_artifact(
        **identity,
        stage="SITUATION_PERSIST",
        output={"response_required": False, "response_text": None},
        output_refs=[],
    )
    return identity


def test_final_disposition_retries_preserve_the_original_manifest():
    identity = _completed_stage()
    arguments = {**identity, "response_required": False, "response_text": None}
    first = artifact_journal.write_final_disposition_artifact(**arguments)
    retry = artifact_journal.write_final_disposition_artifact(**arguments)
    assert retry["artifact_hash"] == first["artifact_hash"]
    assert len(artifact_journal.interaction_artifacts(identity["interaction_id"])) == 2
    assert artifact_journal.verify_interaction_chain(identity["interaction_id"])["valid"]
    artifact_journal.write_stage_error_artifact(
        **identity, stage="SITUATION_PERSIST", claim_id=uuid4(),
        error_type="OSError", message="diagnostic after completion",
    )
    assert artifact_journal.write_final_disposition_artifact(**arguments)["artifact_hash"] == first["artifact_hash"]
    with pytest.raises(ValueError, match="conflicting immutable artifact retry"):
        artifact_journal.write_final_disposition_artifact(
            **identity, response_required=True, response_text="changed completion",
        )


def test_situation_manifest_identifies_its_actual_terminal_stage():
    identity = _completed_stage()
    final = artifact_journal.write_final_disposition_artifact(
        **identity,
        response_required=False,
        response_text=None,
        last_completed_stage="SITUATION_PERSIST",
    )
    assert final["payload"]["last_completed_stage"] == "SITUATION_PERSIST"
    assert [entry["stage"] for entry in final["payload"]["artifact_chain"]] == ["SITUATION_PERSIST"]
