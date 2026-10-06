"""Independent immutable JSON artifact journal for Persistent Cognition.

PostgreSQL remains the indexed operational store, but it is not the sole surviving
representation of cognition or execution.  Every meaningful interaction boundary
and every canonical event can be mirrored here as an atomically-written,
content-hashed JSON artifact.  Interaction artifacts form a hash-linked chain.

The journal deliberately lives outside the database so it remains inspectable and
usable for recovery when scheduler tables or the database itself are unavailable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable
from uuid import UUID, uuid5

from persistent_cognition.perception import Percept, SalienceAssessment
from persistent_cognition import percept_journal
from persistent_cognition.diagnostics import sanitize_message

ARTIFACT_SCHEMA_VERSION = 1
_ARTIFACT_NAMESPACE = UUID("b983b0b7-a203-5c60-96f0-b17d2d94bf1a")


def artifact_root() -> Path:
    """Return the user-controlled local artifact root."""

    configured = os.environ.get("PCR_ARTIFACT_ROOT", "").strip()
    return Path(configured) if configured else Path(".pcr") / "artifacts"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fsync_parent(path: Path) -> None:
    """Best-effort directory fsync; Windows does not expose POSIX directory fsync."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


_REPLACE_RETRY_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_SECONDS = 0.05
_TRANSIENT_WINDOWS_ERRNOS = (5, 32)  # Access is denied / used by another process


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Rename ``source`` onto ``destination``, retrying brief Windows locks.

    On Windows, real-time antivirus scanning (e.g. Windows Defender) can hold a
    transient handle on a just-written file, making ``os.replace`` fail with
    ``PermissionError`` (WinError 5/32) even though no application code holds a
    conflicting handle. High-frequency writers such as job progress reporting
    hit this often enough to abort otherwise-successful long-running jobs, so a
    short bounded retry is used before surfacing the failure.
    """

    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            is_last_attempt = attempt == _REPLACE_RETRY_ATTEMPTS - 1
            if os.name != "nt" or exc.winerror not in _TRANSIENT_WINDOWS_ERRNOS or is_last_attempt:
                raise
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS * (attempt + 1))


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8") + b"\n"
    from persistent_cognition.private_storage import atomic_private_write
    atomic_private_write(path, encoded)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return value


def _interaction_dir(interaction_id: UUID) -> Path:
    return artifact_root() / "interactions" / str(interaction_id)


def _event_dir() -> Path:
    return artifact_root() / "events"


def interaction_artifacts(interaction_id: UUID) -> list[dict[str, Any]]:
    directory = _interaction_dir(interaction_id)
    stream = percept_journal.path_for(interaction_id)
    legacy = sorted(directory.glob("*.json")) if directory.exists() else []
    if stream.exists() and legacy:
        raise RuntimeError(f"interaction has both legacy and percept artifacts: {interaction_id}")
    if stream.exists():
        return [
            {**item, "_path": str(stream)}
            for item in percept_journal.read(stream)
            if item.get("interaction_id") == str(interaction_id)
        ]
    records: list[dict[str, Any]] = []
    for path in legacy:
        record = _load_json(path)
        record["_path"] = str(path)
        records.append(record)
    records.sort(key=lambda item: int(item["journal_sequence"]))
    return records


def _find_artifact_by_key(interaction_id: UUID, artifact_key: str) -> dict[str, Any] | None:
    for artifact in interaction_artifacts(interaction_id):
        if artifact.get("artifact_key") == artifact_key:
            return artifact
    return None


def write_interaction_artifact(
    *,
    artifact_key: str,
    artifact_type: str,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    payload: dict[str, Any],
    producer: str,
    task_id: UUID | None = None,
    assignment_id: UUID | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Append one immutable artifact to an interaction hash chain.

    ``artifact_key`` is the idempotency identity.  Repeating the same key with
    identical semantic content returns the existing artifact.  Conflicting retry
    content fails closed. Filesystem names deliberately use the deterministic
    artifact UUID rather than the potentially long semantic key so path length is
    bounded independently of stage/kind labels and claim identifiers.
    """

    payload_hash = _sha256(payload)
    directory = _interaction_dir(interaction_id)
    if not directory.exists() or not list(directory.glob("*.json")):
        stream = percept_journal.path_for(interaction_id)
        with percept_journal.locked(stream, create=True) as handle:
            entries = percept_journal.entries_unlocked(handle, stream)
            if any(
                entry.get("conversation_id") != str(conversation_id)
                or entry.get("correlation_id") != str(correlation_id)
                for entry in entries
                if entry.get("artifact_type") == "EVENT_RECORD"
            ):
                raise RuntimeError(f"percept journal scope conflicts with worker: {stream}")
            if directory.exists() and list(directory.glob("*.json")):
                raise RuntimeError(f"interaction has both legacy and percept artifacts: {interaction_id}")
            prior = [item for item in entries if item.get("interaction_id") == str(interaction_id)]
            existing = next((item for item in prior if item.get("artifact_key") == artifact_key), None)
            if existing is not None:
                _check_artifact_retry(
                    existing, artifact_key, artifact_type, conversation_id, correlation_id,
                    task_id, assignment_id, stage, producer, payload_hash,
                )
                return {**existing, "_path": str(stream)}
            previous = prior[-1] if prior else None
            envelope = _interaction_envelope(
                artifact_key, artifact_type, interaction_id, conversation_id,
                correlation_id, task_id, assignment_id, stage, producer,
                payload_hash, payload, previous,
            )
            percept_journal.append_unlocked(handle, stream, envelope)
            return {**envelope, "_path": str(stream)}

    existing = _find_artifact_by_key(interaction_id, artifact_key)
    if existing is not None:
        _check_artifact_retry(
            existing, artifact_key, artifact_type, conversation_id, correlation_id,
            task_id, assignment_id, stage, producer, payload_hash,
        )
        return existing

    prior = interaction_artifacts(interaction_id)
    previous = prior[-1] if prior else None
    sequence = int(previous["journal_sequence"]) + 1 if previous else 1
    artifact_id = uuid5(_ARTIFACT_NAMESPACE, f"{interaction_id}:{artifact_key}")
    envelope: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": str(artifact_id),
        "artifact_key": artifact_key,
        "artifact_type": artifact_type,
        "interaction_id": str(interaction_id),
        "conversation_id": str(conversation_id),
        "correlation_id": str(correlation_id),
        "task_id": str(task_id) if task_id else None,
        "assignment_id": str(assignment_id) if assignment_id else None,
        "stage": stage,
        "producer": producer,
        "journal_sequence": sequence,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "previous_artifact_id": previous.get("artifact_id") if previous else None,
        "previous_artifact_hash": previous.get("artifact_hash") if previous else None,
        "payload_hash": payload_hash,
        "payload": payload,
    }
    envelope["artifact_hash"] = _sha256(envelope)
    filename = f"{sequence:06d}-{artifact_id.hex}.json"
    target = _interaction_dir(interaction_id) / filename
    if target.exists():
        # A sequence collision can occur only if two writers raced.  Preserve
        # append-only semantics by retrying with the next free sequence.
        while target.exists():
            sequence += 1
            envelope["journal_sequence"] = sequence
            filename = f"{sequence:06d}-{artifact_id.hex}.json"
            target = _interaction_dir(interaction_id) / filename
        previous = interaction_artifacts(interaction_id)[-1]
        envelope["previous_artifact_id"] = previous.get("artifact_id")
        envelope["previous_artifact_hash"] = previous.get("artifact_hash")
        envelope["artifact_hash"] = _sha256(
            {key: value for key, value in envelope.items() if key != "artifact_hash"}
        )
    _atomic_write_json(target, envelope)
    envelope["_path"] = str(target)
    return envelope


def _check_artifact_retry(
    existing: dict[str, Any], artifact_key: str, artifact_type: str,
    conversation_id: UUID, correlation_id: UUID, task_id: UUID | None,
    assignment_id: UUID | None, stage: str | None, producer: str, payload_hash: str,
) -> None:
    expected = {
        "artifact_type": artifact_type,
        "conversation_id": str(conversation_id),
        "correlation_id": str(correlation_id),
        "task_id": str(task_id) if task_id else None,
        "assignment_id": str(assignment_id) if assignment_id else None,
        "stage": stage,
        "producer": producer,
        "payload_hash": payload_hash,
    }
    if {key: existing.get(key) for key in expected} != expected:
        raise ValueError(f"conflicting immutable artifact retry: {artifact_key}")


def _interaction_envelope(
    artifact_key: str, artifact_type: str, interaction_id: UUID,
    conversation_id: UUID, correlation_id: UUID, task_id: UUID | None,
    assignment_id: UUID | None, stage: str | None, producer: str,
    payload_hash: str, payload: dict[str, Any], previous: dict[str, Any] | None,
) -> dict[str, Any]:
    artifact_id = uuid5(_ARTIFACT_NAMESPACE, f"{interaction_id}:{artifact_key}")
    envelope = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": str(artifact_id),
        "artifact_key": artifact_key,
        "artifact_type": artifact_type,
        "interaction_id": str(interaction_id),
        "conversation_id": str(conversation_id),
        "correlation_id": str(correlation_id),
        "task_id": str(task_id) if task_id else None,
        "assignment_id": str(assignment_id) if assignment_id else None,
        "stage": stage,
        "producer": producer,
        "journal_sequence": int(previous["journal_sequence"]) + 1 if previous else 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "previous_artifact_id": previous.get("artifact_id") if previous else None,
        "previous_artifact_hash": previous.get("artifact_hash") if previous else None,
        "payload_hash": payload_hash,
        "payload": payload,
    }
    envelope["artifact_hash"] = _sha256(envelope)
    return envelope


def write_percept_artifact(
    *,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    task_id: UUID,
    user_text: str,
    user_prompt_event_id: UUID,
    percept: Percept | None = None,
    salience_assessment: SalienceAssessment | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_text": user_text,
        "user_prompt_event_id": str(user_prompt_event_id),
    }
    if percept is not None:
        payload["normalized_percept"] = percept.model_dump(mode="json")
    if salience_assessment is not None:
        payload["salience_assessment"] = salience_assessment.model_dump(mode="json")
    return write_interaction_artifact(
        artifact_key="percept",
        artifact_type="PERCEPT",
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=None,
        stage=None,
        producer="percept_response_v2",
        payload=payload,
    )


def write_stage_result_artifact(
    *,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    task_id: UUID,
    assignment_id: UUID,
    stage: str,
    output: dict[str, Any],
    output_refs: Iterable[str],
) -> dict[str, Any]:
    return write_interaction_artifact(
        artifact_key=f"stage-result:{stage}",
        artifact_type="STAGE_RESULT",
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage=stage,
        producer="percept_response_v2",
        payload={"output": output, "output_refs": list(output_refs)},
    )


def write_stage_error_artifact(
    *,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    task_id: UUID,
    assignment_id: UUID,
    stage: str,
    claim_id: UUID,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return write_interaction_artifact(
        artifact_key=f"stage-error:{stage}:{claim_id}",
        artifact_type="STAGE_ERROR",
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage=stage,
        producer="percept_response_v2",
        payload={
            "claim_id": str(claim_id),
            "error_type": error_type,
            "message": sanitize_message(message),
        },
    )


def load_stage_result_artifact(interaction_id: UUID, stage: str) -> dict[str, Any] | None:
    artifact = _find_artifact_by_key(interaction_id, f"stage-result:{stage}")
    if artifact is None:
        return None
    payload = artifact.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("output"), dict):
        raise ValueError(f"invalid stage artifact for {stage}")
    return payload


def write_final_disposition_artifact(
    *,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    task_id: UUID,
    assignment_id: UUID,
    response_required: bool,
    response_text: str | None,
    last_completed_stage: str = "V2_PERSIST_RESULT",
) -> dict[str, Any]:
    chain = interaction_artifacts(interaction_id)
    # A completion retry must describe the same prefix as the original manifest,
    # never include itself or later diagnostic artifacts in its own evidence.
    for index, item in enumerate(chain):
        if item.get("artifact_key") == "final-disposition":
            chain = chain[:index]
            break
    references = [
        {
            "artifact_id": item["artifact_id"],
            "artifact_key": item["artifact_key"],
            "artifact_type": item["artifact_type"],
            "journal_sequence": item["journal_sequence"],
            "artifact_hash": item["artifact_hash"],
            "stage": item.get("stage"),
        }
        for item in chain
    ]
    return write_interaction_artifact(
        artifact_key="final-disposition",
        artifact_type="FINAL_DISPOSITION",
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage=None,
        producer="percept_response_v2",
        payload={
            "status": "COMPLETED",
            "last_completed_stage": last_completed_stage,
            "response_required": response_required,
            "response_text": response_text,
            "artifact_chain": references,
        },
    )


def verify_interaction_chain(interaction_id: UUID) -> dict[str, Any]:
    artifacts = interaction_artifacts(interaction_id)
    previous_id: str | None = None
    previous_hash: str | None = None
    errors: list[str] = []
    for expected_sequence, artifact in enumerate(artifacts, start=1):
        sequence = int(artifact.get("journal_sequence", 0))
        if sequence != expected_sequence:
            errors.append(
                f"sequence gap: expected {expected_sequence}, found {sequence}"
            )
        if artifact.get("previous_artifact_id") != previous_id:
            errors.append(f"previous artifact id mismatch at sequence {sequence}")
        if artifact.get("previous_artifact_hash") != previous_hash:
            errors.append(f"previous artifact hash mismatch at sequence {sequence}")
        payload = artifact.get("payload")
        if artifact.get("payload_hash") != _sha256(payload):
            errors.append(f"payload hash mismatch at sequence {sequence}")
        stored_hash = artifact.get("artifact_hash")
        without_hash = {
            key: value
            for key, value in artifact.items()
            if key not in {"artifact_hash", "_path"}
        }
        if stored_hash != _sha256(without_hash):
            errors.append(f"artifact hash mismatch at sequence {sequence}")
        previous_id = artifact.get("artifact_id")
        previous_hash = stored_hash
    return {
        "interaction_id": str(interaction_id),
        "artifact_count": len(artifacts),
        "valid": not errors,
        "errors": errors,
        "complete": any(
            item.get("artifact_type") == "FINAL_DISPOSITION" for item in artifacts
        ),
        "last_artifact": artifacts[-1] if artifacts else None,
    }


def inspect_interaction(interaction_id: UUID) -> dict[str, Any]:
    artifacts = interaction_artifacts(interaction_id)
    return {
        "verification": verify_interaction_chain(interaction_id),
        "artifacts": artifacts,
    }


def latest_interaction_id(*, complete: bool | None = None) -> UUID | None:
    directory = artifact_root() / "interactions"
    stream_root = percept_journal.root()
    if not directory.exists() and not stream_root.exists():
        return None
    candidates: list[tuple[float, UUID]] = []
    children = list(directory.iterdir()) if directory.exists() else []
    children.extend(stream_root.glob("*.jsonl") if stream_root.exists() else ())
    for child in children:
        try:
            interaction_id = UUID(child.stem if child.is_file() else child.name)
        except ValueError:
            continue
        artifacts = interaction_artifacts(interaction_id)
        if not artifacts:
            continue
        is_complete = any(
            item.get("artifact_type") == "FINAL_DISPOSITION" for item in artifacts
        )
        if complete is not None and is_complete is not complete:
            continue
        newest = max(Path(item["_path"]).stat().st_mtime for item in artifacts)
        candidates.append((newest, interaction_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].hex), reverse=True)
    return candidates[0][1]


def _event_record_path(event_id: UUID) -> Path:
    return _event_dir() / f"{event_id}.json"


def _event_commit_path(event_id: UUID) -> Path:
    return _event_dir() / f"{event_id}.commit.json"


def write_event_record_artifact(
    *,
    event_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    conversation_seq: int,
    event_type: str,
    source: str,
    payload: dict[str, Any],
    payload_text: str | None,
) -> dict[str, Any]:
    record = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "EVENT_RECORD",
        "event_id": str(event_id),
        "conversation_id": str(conversation_id),
        "correlation_id": str(correlation_id),
        "conversation_seq": conversation_seq,
        "event_type": event_type,
        "source": source,
        "payload": payload,
        "payload_text": payload_text,
        "journaled_at": datetime.now(timezone.utc).isoformat(),
    }
    record["record_hash"] = _sha256(record)
    path = _event_record_path(event_id)
    if path.exists():
        existing = _load_json(path)
        comparable_existing = {key: value for key, value in existing.items() if key != "journaled_at"}
        comparable_new = {key: value for key, value in record.items() if key != "journaled_at"}
        if comparable_existing != comparable_new:
            raise ValueError(f"conflicting event artifact retry: {event_id}")
        return existing
    _atomic_write_json(path, record)
    return record


def write_event_commit_artifact(
    *,
    event_id: UUID,
    global_seq: int,
    conversation_seq: int,
    created_at: datetime,
    schema_version: int,
) -> dict[str, Any]:
    record_path = _event_record_path(event_id)
    if not record_path.exists():
        raise RuntimeError(f"event commit has no event record artifact: {event_id}")
    event_record = _load_json(record_path)
    commit = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "EVENT_DATABASE_COMMIT",
        "event_id": str(event_id),
        "record_hash": event_record["record_hash"],
        "global_seq": global_seq,
        "conversation_seq": conversation_seq,
        "created_at": created_at.isoformat(),
        "schema_version": schema_version,
    }
    commit["commit_hash"] = _sha256(commit)
    path = _event_commit_path(event_id)
    if path.exists():
        existing = _load_json(path)
        if existing != commit:
            raise ValueError(f"conflicting event commit artifact retry: {event_id}")
        return existing
    _atomic_write_json(path, commit)
    return commit


def iter_event_artifacts() -> list[dict[str, Any]]:
    directory = _event_dir()
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        if path.name.endswith(".commit.json"):
            continue
        record = _load_json(path)
        event_id = UUID(str(record["event_id"]))
        commit_path = _event_commit_path(event_id)
        commit = _load_json(commit_path) if commit_path.exists() else None
        records.append({"record": record, "commit": commit})
    records.sort(
        key=lambda item: (
            int(item["commit"]["global_seq"]) if item["commit"] else 2**63 - 1,
            item["record"]["journaled_at"],
            item["record"]["event_id"],
        )
    )
    return records
