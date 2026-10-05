"""Closed deterministic reflexes over live, application-owned facts.

No percept text or salience score selects a process, filesystem path, resource
policy, or action. These entry points are called by the corresponding owners.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from uuid import UUID

import psycopg

from prometheist.artifact_journal import artifact_root, _atomic_write_json
from prometheist.cognitive_store import put_record
from prometheist.percept_adapters import MediaReference, MEDIA_CHUNK_BYTES, media_path
from prometheist.worker_store import release_worker_claim

HEARTBEAT_SUSPECT_SECONDS = 120
REFLEX_POLL_LIMIT = 64
POST_TERMINATION_WAIT_SECONDS = 10


def record_admission_denial(conn: psycopg.Connection, observation) -> None:
    put_record(conn, "reflex", str(observation.observation_id),
               {"action": "DENY_MODEL_OR_PROCESS_ADMISSION", "reason": observation.reason,
                "evidence": observation.model_dump(mode="json"), "authority": "CLAIM_TIME_RESOURCE_GATE"}, revision="1")


def mark_missing_heartbeats(conn: psycopg.Connection, *, now: datetime, after_claim: UUID | None = None) -> str | None:
    rows = conn.execute(
        """SELECT claim_id, worker_id, last_heartbeat_at, lease_expires_at
           FROM attention_worker_claims WHERE status = 'ACTIVE'
             AND (%s::uuid IS NULL OR claim_id > %s::uuid)
           ORDER BY claim_id LIMIT %s""", (after_claim, after_claim, REFLEX_POLL_LIMIT),
    ).fetchall()
    for claim_id, worker_id, heartbeat_at, lease_expires_at in rows:
        if (now - heartbeat_at).total_seconds() < HEARTBEAT_SUSPECT_SECONDS and now < lease_expires_at:
            continue
        put_record(conn, "lease_health", str(claim_id),
                   {"action": "MARK_LEASE_SUSPECT", "worker_id": worker_id,
                    "claim_id": str(claim_id), "last_heartbeat_at": heartbeat_at.isoformat(),
                    "lease_expires_at": lease_expires_at.isoformat(),
                    "meaning": "Liveness is uncertain; this does not authorize termination or lease stealing."},
                   revision=heartbeat_at.isoformat())
    return str(rows[-1][0]) if rows else None


def terminate_owned_worker(conn: psycopg.Connection, launched, *, scheduler_key: str) -> None:
    """Called only after the launcher's wait exceeds its declared lifetime.

    Only the retained process handle is accepted; no arbitrary PID or command
    from a model or observation can enter this boundary.
    """
    claim = launched.envelope.claim
    put_record(conn, "reflex_intention", str(claim.claim_id),
               {"action": "TERMINATE_OWNED_WORKER", "claim_id": str(claim.claim_id),
                "worker_id": claim.worker_id, "trigger": "DECLARED_WORKER_LIFETIME_EXCEEDED"}, revision="1")
    if launched.process.poll() is None:
        launched.process.kill()
    code = launched.process.wait(timeout=POST_TERMINATION_WAIT_SECONDS)
    put_record(conn, "reflex", str(claim.claim_id),
               {"action": "TERMINATE_OWNED_WORKER", "observed_exit_code": code,
                "recovery": "REHYDRATE_IDEMPOTENT_STAGE_FROM_DURABLE_HANDOFF"}, revision="1")
    release_worker_claim(conn, claim_id=claim.claim_id, worker_id=claim.worker_id, scheduler_key=scheduler_key)


def quarantine_corrupt_media(conn: psycopg.Connection, reference: MediaReference) -> bool:
    """Rehash the canonical object; invalid caller metadata cannot quarantine it."""
    path = media_path(reference)
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MEDIA_CHUNK_BYTES), b""):
            digest.update(chunk)
    if digest.hexdigest() == reference.sha256:
        return False
    data = {"action": "QUARANTINE_MEDIA", "expected_sha256": reference.sha256,
            "observed_sha256": digest.hexdigest(), "canonical_bytes_preserved": True}
    put_record(conn, "reflex", f"media:{reference.sha256}", data, revision=digest.hexdigest())
    _atomic_write_json(artifact_root() / "quarantine" / f"{reference.sha256}.json", data)
    return True


def poll_reflexes(conn: psycopg.Connection) -> None:
    from prometheist.cognitive_store import get_record
    cursor = get_record(conn, "reflex_cursor", "heartbeat") or {"after_claim": None, "revision": 0}
    after = UUID(cursor["after_claim"]) if cursor["after_claim"] else None
    next_claim = mark_missing_heartbeats(conn, now=datetime.now(timezone.utc), after_claim=after)
    revision = cursor["revision"] + 1
    put_record(conn, "reflex_cursor", "heartbeat", {"after_claim": next_claim, "revision": revision}, revision=str(revision))
