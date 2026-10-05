from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from prometheist import artifact_journal, journal_signing


def _finalized_interaction(tmp_path, monkeypatch) -> tuple:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id, conversation_id=uuid4(), correlation_id=uuid4(),
        task_id=interaction_id, user_text="sign this interaction", user_prompt_event_id=uuid4(),
    )
    artifact_journal.write_final_disposition_artifact(
        interaction_id=interaction_id, conversation_id=uuid4(), correlation_id=uuid4(),
        task_id=interaction_id, assignment_id=uuid4(), response_required=True, response_text="done",
    )
    return interaction_id


def test_ensure_signing_key_generates_once_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    first = journal_signing.ensure_signing_key()
    second = journal_signing.ensure_signing_key()
    assert first == second
    assert (tmp_path / "artifacts" / "keys" / f"{first}.private").exists()
    assert (tmp_path / "artifacts" / "keys" / f"{first}.public").exists()


def test_sign_journal_head_refuses_an_incomplete_interaction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id, conversation_id=uuid4(), correlation_id=uuid4(),
        task_id=interaction_id, user_text="not finalized", user_prompt_event_id=uuid4(),
    )
    with pytest.raises(ValueError, match="no FINAL_DISPOSITION"):
        journal_signing.sign_journal_head(interaction_id)


def test_sign_journal_head_refuses_an_invalid_chain(tmp_path, monkeypatch) -> None:
    interaction_id = _finalized_interaction(tmp_path, monkeypatch)
    percept_path = Path(artifact_journal.interaction_artifacts(interaction_id)[0]["_path"])
    lines = percept_path.read_text(encoding="utf-8").splitlines()
    document = json.loads(lines[0])
    document["payload"]["user_text"] = "tampered"
    lines[0] = json.dumps(document)
    percept_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid artifact chain"):
        journal_signing.sign_journal_head(interaction_id)


def test_sign_and_verify_round_trip(tmp_path, monkeypatch) -> None:
    interaction_id = _finalized_interaction(tmp_path, monkeypatch)

    anchor = journal_signing.sign_journal_head(interaction_id)

    assert anchor["interaction_id"] == str(interaction_id)
    assert anchor["algorithm"] == "ed25519"
    report = journal_signing.verify_signed_journal_head(interaction_id)
    assert report == {
        "interaction_id": str(interaction_id), "signed": True, "valid": True,
        "signing_key_id": anchor["signing_key_id"], "journal_head": anchor["journal_head"],
        "signed_at": anchor["signed_at"],
    }


def test_verify_reports_unsigned_for_an_interaction_that_was_never_signed(tmp_path, monkeypatch) -> None:
    interaction_id = _finalized_interaction(tmp_path, monkeypatch)
    report = journal_signing.verify_signed_journal_head(interaction_id)
    assert report == {
        "interaction_id": str(interaction_id), "signed": False, "valid": False,
        "reason": "no signed anchor found",
    }


def test_verify_detects_a_tampered_anchor_file(tmp_path, monkeypatch) -> None:
    interaction_id = _finalized_interaction(tmp_path, monkeypatch)
    journal_signing.sign_journal_head(interaction_id)
    anchor_path = tmp_path / "artifacts" / "anchors" / f"{interaction_id}.json"
    document = json.loads(anchor_path.read_text(encoding="utf-8"))
    document["record_count"] = document["record_count"] + 1000
    anchor_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    report = journal_signing.verify_signed_journal_head(interaction_id)

    assert report["signed"] is True
    assert report["valid"] is False
    assert "InvalidSignature" in report["reason"]


def test_verify_detects_the_journal_being_rewritten_after_signing(tmp_path, monkeypatch) -> None:
    """The exact SuperLocalMemory-style attack: rewrite local artifacts after
    they were anchored, recomputing every downstream hash so the rewritten
    chain stays perfectly internally self-consistent -- and leave the anchor
    file itself untouched. ``verify_interaction_chain`` alone cannot catch
    this (it only checks internal consistency); the external signature can.
    """

    interaction_id = _finalized_interaction(tmp_path, monkeypatch)
    journal_signing.sign_journal_head(interaction_id)

    artifacts = artifact_journal.interaction_artifacts(interaction_id)
    path = Path(artifacts[0]["_path"])
    documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    document = documents[0]
    document["payload"]["user_text"] = "an attacker rewrote history after anchoring"
    document["payload_hash"] = artifact_journal._sha256(document["payload"])

    previous_id, previous_hash = document.get("previous_artifact_id"), document.get("previous_artifact_hash")
    for index, artifact in enumerate(artifacts):
        current = document if index == 0 else documents[index]
        current["previous_artifact_id"], current["previous_artifact_hash"] = previous_id, previous_hash
        current["artifact_hash"] = artifact_journal._sha256(
            {key: value for key, value in current.items() if key not in {"artifact_hash", "_path"}}
        )
        documents[index] = current
        previous_id, previous_hash = current["artifact_id"], current["artifact_hash"]
    path.write_text("\n".join(json.dumps(item) for item in documents) + "\n", encoding="utf-8")

    # The rewritten chain is still perfectly self-consistent on its own terms.
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"] is True

    report = journal_signing.verify_signed_journal_head(interaction_id)

    assert report["signed"] is True
    assert report["valid"] is False
    assert "no longer matches" in report["reason"]


def test_verify_succeeds_using_only_the_public_key(tmp_path, monkeypatch) -> None:
    """A separate trust domain needs only the anchor and the public key."""

    interaction_id = _finalized_interaction(tmp_path, monkeypatch)
    anchor = journal_signing.sign_journal_head(interaction_id)
    private_key_path = tmp_path / "artifacts" / "keys" / f"{anchor['signing_key_id']}.private"
    private_key_path.unlink()

    report = journal_signing.verify_signed_journal_head(interaction_id)

    assert report["valid"] is True


def test_key_id_for_public_bytes_is_deterministic() -> None:
    assert journal_signing.key_id_for_public_bytes(b"same") == journal_signing.key_id_for_public_bytes(b"same")
    assert journal_signing.key_id_for_public_bytes(b"a") != journal_signing.key_id_for_public_bytes(b"b")
