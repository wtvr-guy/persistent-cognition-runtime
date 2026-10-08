from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from persistent_cognition import artifact_journal, audit_report, blob_store, llm_artifact_store


def _write_full_chain(*, response_text: str = "done") -> tuple:
    interaction_id = uuid4()
    conversation_id = uuid4()
    correlation_id = uuid4()
    task_id = interaction_id
    assignment_id = uuid4()
    claim_id = uuid4()

    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        user_text="remember the phrase",
        user_prompt_event_id=uuid4(),
    )
    artifact_journal.write_stage_result_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_PRECOGNITIVE",
        output={"disposition": {"response_required": True}},
        output_refs=["memory-request:example"],
    )
    llm_artifact_store.write_llm_invocation(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_RESPOND",
        claim_id=claim_id,
        invocation_index=0,
        kind="FINAL_RESPONSE",
        model="qwen3:4b",
        base_url="http://localhost:11434",
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object"},
        max_tokens=256,
        temperature=0.2,
        output='{"answer":"hello"}',
        error_type=None,
        error_message=None,
    )
    artifact_journal.write_final_disposition_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        response_required=True,
        response_text=response_text,
    )
    return interaction_id, conversation_id, correlation_id, task_id, assignment_id


def test_render_interaction_audit_reports_full_timeline_and_disposition(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id, *_ = _write_full_chain()

    report = audit_report.render_interaction_audit(interaction_id)

    assert report.startswith("# Percept Audit\n")
    assert f"Interaction: {interaction_id}" in report
    assert "Disposition: COMPLETED (V2_PERSIST_RESULT)" in report
    assert "## Integrity" in report
    assert "Journal records: 4" in report
    assert "Hash chain: PASS" in report
    assert "## Timeline" in report
    assert '[  1] PERCEPT' in report
    assert 'user_text="remember the phrase"' in report
    assert '[  2] STAGE_RESULT' in report
    assert "stage=V2_PRECOGNITIVE output_refs=1" in report
    assert '[  3] LLM_INVOCATION' in report
    assert "stage=V2_RESPOND kind=FINAL_RESPONSE model=qwen3:4b output_chars=" in report
    assert '[  4] FINAL_DISPOSITION' in report
    assert "response_required=True" in report
    assert 'response="done"' in report

    # The timeline must be in strict journal order.
    sequence_positions = [report.index(f"[{n:>3}]") for n in (1, 2, 3, 4)]
    assert sequence_positions == sorted(sequence_positions)


def test_render_interaction_audit_reports_incomplete_interaction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=interaction_id,
        user_text="an unfinished interaction",
        user_prompt_event_id=uuid4(),
    )

    report = audit_report.render_interaction_audit(interaction_id)

    assert "Disposition: INCOMPLETE" in report
    assert "Finalized: (incomplete)" in report
    assert "Hash chain: PASS" in report


def test_render_interaction_audit_reports_stage_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=interaction_id,
        user_text="trigger a failure",
        user_prompt_event_id=uuid4(),
    )
    artifact_journal.write_stage_error_artifact(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=interaction_id,
        assignment_id=uuid4(),
        stage="V2_PRECOGNITIVE",
        claim_id=uuid4(),
        error_type="ValueError",
        message="the model returned malformed structured output",
    )

    report = audit_report.render_interaction_audit(interaction_id)

    assert "STAGE_ERROR" in report
    assert "error_type=ValueError" in report
    assert 'message="the model returned malformed structured output"' in report
    assert "Disposition: INCOMPLETE" in report


def test_render_interaction_audit_reports_broken_hash_chain(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id, *_ = _write_full_chain()
    percept_path = Path(
        artifact_journal.interaction_artifacts(interaction_id)[0]["_path"]
    )
    lines = percept_path.read_text(encoding="utf-8").splitlines()
    document = json.loads(lines[0])
    document["payload"]["user_text"] = "an attacker rewrote this after the fact"
    lines[0] = json.dumps(document)
    percept_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = audit_report.render_interaction_audit(interaction_id)

    assert "Hash chain: FAIL" in report
    assert "  - " in report


def test_render_interaction_audit_counts_referenced_blobs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    descriptor = blob_store.put_blob(b"a large exact model generation", media_type="text/plain")
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=interaction_id,
        user_text="store something large",
        user_prompt_event_id=uuid4(),
    )
    artifact_journal.write_stage_result_artifact(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=interaction_id,
        assignment_id=uuid4(),
        stage="V2_RESPOND",
        output={"large_result": descriptor.to_dict()},
        output_refs=[],
    )

    report = audit_report.render_interaction_audit(interaction_id)

    assert "Referenced blobs: 1/1 verified" in report


def test_render_interaction_audit_reports_unverifiable_missing_blob(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    missing_descriptor = blob_store.BlobDescriptor(
        media_type="text/plain", digest="sha256:" + "ab" * 32, size=4
    )
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=interaction_id,
        user_text="reference a blob that was never written",
        user_prompt_event_id=uuid4(),
    )
    artifact_journal.write_stage_result_artifact(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=interaction_id,
        assignment_id=uuid4(),
        stage="V2_RESPOND",
        output={"large_result": missing_descriptor.to_dict()},
        output_refs=[],
    )

    report = audit_report.render_interaction_audit(interaction_id)

    assert "Referenced blobs: 0/1 verified" in report


def test_render_interaction_audit_raises_for_unknown_interaction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    with pytest.raises(ValueError):
        audit_report.render_interaction_audit(uuid4())
