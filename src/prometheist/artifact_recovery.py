"""Recovery operations backed by Prometheist's independent artifact journal."""
from __future__ import annotations

from datetime import datetime, timezone
import os
import subprocess
import sys
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist import artifact_journal, db, event_artifact_store
from prometheist.attention_observation import (
    HostResourceProbe,
    ResourceSafetyPolicy,
    SystemHostResourceProbe,
)
from prometheist.attention_store import DEFAULT_SCHEDULER_KEY
from prometheist.interaction_store import load_interaction
from prometheist.native_policy import native_resource_safety_policy
from prometheist.ollama_runtime import OllamaClaimHostResourceProbe, OllamaRuntimeProbe
from prometheist.percept_response_runtime import (
    PERCEPT_STAGES,
    PerceptStage,
    _stage_result,
    begin_percept,
    finish_percept,
)
from prometheist.worker_protocol import deterministic_worker_step_id
from prometheist.worker_runtime import GuardedWorkerLauncher


def restore_event_store_from_artifacts(conn: psycopg.Connection) -> dict[str, int]:
    """Restore missing canonical event rows from independent JSON artifacts.

    This operation is non-destructive: matching existing events are retained,
    missing events are inserted, and conflicting existing identities fail closed.
    Committed artifacts retain their original global sequence. A semantic event
    artifact that survived before its DB commit is assigned a new global sequence
    after all known committed events while preserving its conversation sequence.
    """

    artifact_pairs = event_artifact_store.iter_event_artifacts()
    if not artifact_pairs:
        return {"artifact_events": 0, "inserted_events": 0, "existing_events": 0}

    committed_max = max(
        (
            int(item["commit"]["global_seq"])
            for item in artifact_pairs
            if item["commit"] is not None
        ),
        default=0,
    )
    next_uncommitted_global = committed_max + 1
    inserted_event_ids: set[UUID] = set()
    existing_event_ids: set[UUID] = set()

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            for item in artifact_pairs:
                record = item["record"]
                commit = item["commit"]
                event_id = UUID(record["event_id"])
                conversation_id = UUID(record["conversation_id"])
                correlation_id = UUID(record["correlation_id"])
                conversation_seq = int(record["conversation_seq"])
                if commit is not None:
                    global_seq = int(commit["global_seq"])
                    created_at = datetime.fromisoformat(commit["created_at"])
                    schema_version = int(commit["schema_version"])
                else:
                    global_seq = next_uncommitted_global
                    next_uncommitted_global += 1
                    created_at = datetime.fromisoformat(record["journaled_at"])
                    schema_version = 1
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)

                cur.execute(
                    """
                    INSERT INTO conversations (conversation_id, started_at, next_event_seq)
                    VALUES (%s, %s, 1)
                    ON CONFLICT (conversation_id) DO NOTHING
                    """,
                    (conversation_id, created_at),
                )
                cur.execute(
                    """
                    SELECT event_id, conversation_id, correlation_id, global_seq,
                           conversation_seq, event_type, source, payload, payload_text,
                           schema_version
                    FROM events
                    WHERE event_id = %s
                    """,
                    (event_id,),
                )
                existing = cur.fetchone()
                if existing is not None:
                    expected = {
                        "event_id": event_id,
                        "conversation_id": conversation_id,
                        "correlation_id": correlation_id,
                        "conversation_seq": conversation_seq,
                        "event_type": record["event_type"],
                        "source": record["source"],
                        "payload": record["payload"],
                        "payload_text": record.get("payload_text"),
                    }
                    actual = {key: existing[key] for key in expected}
                    if actual != expected:
                        raise RuntimeError(f"event artifact conflicts with database row {event_id}")
                    existing_event_ids.add(event_id)
                    continue

                cur.execute(
                    "SELECT event_id FROM events WHERE global_seq = %s",
                    (global_seq,),
                )
                global_collision = cur.fetchone()
                if global_collision is not None:
                    if commit is not None:
                        raise RuntimeError(
                            f"committed artifact global_seq collision at {global_seq}"
                        )
                    cur.execute("SELECT COALESCE(max(global_seq), 0) + 1 AS value FROM events")
                    global_seq = int(db.require_row(cur.fetchone(), context="next free global_seq")["value"])
                    next_uncommitted_global = max(next_uncommitted_global, global_seq + 1)

                cur.execute(
                    """
                    INSERT INTO events (
                        event_id, conversation_id, correlation_id, global_seq,
                        conversation_seq, event_type, source, created_at,
                        payload, payload_text, schema_version
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event_id,
                        conversation_id,
                        correlation_id,
                        global_seq,
                        conversation_seq,
                        record["event_type"],
                        record["source"],
                        created_at,
                        Json(record["payload"]),
                        record.get("payload_text"),
                        schema_version,
                    ),
                )
                inserted_event_ids.add(event_id)

            cur.execute(
                """
                UPDATE conversations AS c
                SET next_event_seq = source.next_seq
                FROM (
                    SELECT conversation_id, COALESCE(max(conversation_seq), 0) + 1 AS next_seq
                    FROM events
                    GROUP BY conversation_id
                ) AS source
                WHERE c.conversation_id = source.conversation_id
                """
            )
            cur.execute("SELECT COALESCE(max(global_seq), 0) AS value FROM events")
            maximum = int(db.require_row(cur.fetchone(), context="restored max global_seq")["value"])
            if maximum:
                cur.execute(
                    "SELECT setval(pg_get_serial_sequence('events', 'global_seq'), %s, true)",
                    (maximum,),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "artifact_events": len(artifact_pairs),
        "inserted_events": len(inserted_event_ids),
        "existing_events": len(existing_event_ids),
    }


def _percept_recovery_seed(interaction_id: UUID) -> tuple[UUID, UUID, str]:
    chain = artifact_journal.interaction_artifacts(interaction_id)
    percept = next(
        (item for item in chain if item.get("artifact_type") == "PERCEPT"),
        None,
    )
    if percept is None:
        raise RuntimeError(f"interaction {interaction_id} has no percept artifact")
    payload = percept.get("payload") or {}
    user_text = payload.get("user_text")
    if not isinstance(user_text, str) or not user_text.strip():
        raise RuntimeError(f"interaction {interaction_id} percept has no user text")
    return (
        UUID(percept["conversation_id"]),
        UUID(percept["correlation_id"]),
        user_text,
    )


def _ensure_final_disposition(interaction, *, scheduler_key: str, conn) -> None:
    verification = artifact_journal.verify_interaction_chain(interaction.interaction_id)
    if verification["complete"]:
        return
    persisted = _stage_result(
        conn,
        interaction,
        PerceptStage.PERSIST_RESULT,
        scheduler_key,
    )
    artifact_journal.write_final_disposition_artifact(
        interaction_id=interaction.interaction_id,
        conversation_id=interaction.conversation_id,
        correlation_id=interaction.correlation_id,
        task_id=interaction.task_id,
        assignment_id=interaction.assignment_id,
        response_required=bool(persisted["response_required"]),
        response_text=(
            str(persisted["response_text"])
            if persisted.get("response_text") is not None
            else None
        ),
    )


def resume_interaction_from_artifacts(
    conn: psycopg.Connection,
    interaction_id: UUID,
    *,
    probe: HostResourceProbe | None = None,
    policy: ResourceSafetyPolicy | None = None,
    ollama_runtime_probe: OllamaRuntimeProbe | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
    worker_lease_seconds: int | None = None,
    worker_timeout_seconds: int | None = None,
) -> str | None:
    """Resume an incomplete interaction from the first missing stage.

    If the operational interaction row still exists, recovery continues it. If
    PostgreSQL execution state was lost after canonical events were restored,
    Prometheist recreates the same interaction identity from the percept artifact
    and original correlation id. Existing stage artifacts are rehydrated in
    order, so completed LLM/tool stages are not repeated.
    """

    verification = artifact_journal.verify_interaction_chain(interaction_id)
    if not verification["valid"]:
        raise RuntimeError(
            "artifact chain failed verification: " + "; ".join(verification["errors"])
        )
    if verification["complete"]:
        chain = artifact_journal.interaction_artifacts(interaction_id)
        final = next(
            item
            for item in reversed(chain)
            if item["artifact_type"] == "FINAL_DISPOSITION"
        )
        return final["payload"].get("response_text")

    effective_lease = worker_lease_seconds or int(
        os.environ.get("PROMETHEIST_WORKER_LEASE_SECONDS", "600")
    )
    effective_timeout = worker_timeout_seconds or int(
        os.environ.get("PROMETHEIST_WORKER_TIMEOUT_SECONDS", "660")
    )
    kill_wait = float(os.environ.get("PROMETHEIST_KILL_WAIT_SECONDS", "10"))
    effective_policy = policy or native_resource_safety_policy()
    physical_probe = probe or SystemHostResourceProbe()
    runtime_probe = ollama_runtime_probe or OllamaRuntimeProbe()
    runtime_state = runtime_probe.capture()

    try:
        interaction = load_interaction(conn, interaction_id, scheduler_key=scheduler_key)
    except KeyError:
        conversation_id, correlation_id, user_text = _percept_recovery_seed(interaction_id)
        interaction = begin_percept(
            conn,
            user_text,
            conversation_id,
            correlation_id=correlation_id,
            probe=physical_probe,
            policy=effective_policy,
            ollama_runtime_state=runtime_state,
            scheduler_key=scheduler_key,
        )
        if interaction.interaction_id != interaction_id:
            raise RuntimeError("recreated interaction identity does not match artifact journal")

    scheduled_memory_mib = runtime_state.incremental_process_memory_mib(effective_policy)
    claim_probe = OllamaClaimHostResourceProbe(
        base_probe=physical_probe,
        runtime_probe=runtime_probe,
        policy=effective_policy,
        scheduled_memory_mib=scheduled_memory_mib,
    )
    launcher = GuardedWorkerLauncher(
        db.get_connection,
        probe=claim_probe,
        policy=effective_policy,
        scheduler_key=scheduler_key,
    )

    for stage in PERCEPT_STAGES:
        try:
            _stage_result(conn, interaction, stage, scheduler_key)
            continue
        except RuntimeError as exc:
            expected = f"percept stage {stage.value} is incomplete"
            if str(exc) != expected:
                raise
        step_id = deterministic_worker_step_id(interaction.assignment_id, stage.value)
        worker_id = f"percept-v2-recover-{interaction.interaction_id}-{stage.value.casefold()}"
        launched = launcher.launch(
            step_id=step_id,
            worker_id=worker_id,
            command=[sys.executable, "-m", "prometheist.percept_response_worker"],
            lease_seconds=effective_lease,
        )
        try:
            return_code = launched.process.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            launched.process.kill()
            launched.process.wait(timeout=kill_wait)
            raise RuntimeError(f"recovery worker timed out at stage {stage.value}") from None
        if return_code != 0:
            raise RuntimeError(
                f"recovery worker failed at stage {stage.value} with exit code {return_code}"
            )

    response = finish_percept(
        conn,
        interaction,
        probe=physical_probe,
        policy=effective_policy,
        scheduler_key=scheduler_key,
    )
    _ensure_final_disposition(interaction, scheduler_key=scheduler_key, conn=conn)
    return response
