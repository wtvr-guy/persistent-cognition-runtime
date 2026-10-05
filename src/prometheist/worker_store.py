"""PostgreSQL durability and guarded leasing for disposable workers."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist.attention_assignments import AssignmentStatus, DurableAssignment
from prometheist.attention_observation import (
    HostResourceProbe,
    ResourceObservationSnapshot,
    ResourceSafetyPolicy,
    SystemHostResourceProbe,
    build_resource_observation,
)
from prometheist.attention_resources import (
    ExecutionResource,
    ExecutionResourceClass,
    ResourceReservation,
)
from prometheist.attention_store import DEFAULT_SCHEDULER_KEY
from prometheist.db import require_row
from prometheist.worker_protocol import (
    WORKER_CLAIM_POLICY_VERSION,
    WORKER_PROTOCOL_VERSION,
    WorkerCheckpoint,
    WorkerClaim,
    WorkerClaimAttempt,
    WorkerClaimDecision,
    WorkerClaimEnvelope,
    WorkerClaimResourceObservation,
    WorkerClaimStatus,
    WorkerEffectPolicy,
    WorkerResult,
    WorkerStep,
    deterministic_worker_checkpoint_id,
    deterministic_worker_claim_id,
    deterministic_worker_claim_observation_id,
    deterministic_worker_idempotency_key,
    deterministic_worker_result_id,
    deterministic_worker_step_id,
)


DEFAULT_WORKER_LEASE_SECONDS = 30
MAX_WORKER_LEASE_SECONDS = 3_600


class WorkerProtocolError(RuntimeError):
    """Durable worker state violates the Increment F contract."""


def register_worker_step(
    conn: psycopg.Connection,
    *,
    assignment_id: UUID,
    step_key: str,
    capability: str,
    input_refs: list[str] | None = None,
    effect_policy: WorkerEffectPolicy = WorkerEffectPolicy.NO_EXTERNAL_EFFECT,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerStep:
    """Create one immutable step only from a currently committed assignment."""

    normalized_key = step_key.strip()
    normalized_capability = capability.strip()
    if not normalized_key:
        raise ValueError("step_key must not be empty")
    if not normalized_capability:
        raise ValueError("capability must not be empty")

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            state = _lock_scheduler_state(cur, scheduler_key)
            current_epoch_id = state["current_epoch_id"]
            if current_epoch_id is None:
                raise WorkerProtocolError(
                    "worker steps require a committed scheduling epoch"
                )
            cur.execute(
                """
                SELECT assignment_ids
                FROM attention_scheduling_epochs
                WHERE scheduler_key = %s AND epoch_id = %s
                """,
                (scheduler_key, current_epoch_id),
            )
            epoch = cur.fetchone()
            if epoch is None:
                raise WorkerProtocolError(
                    "scheduler state references a missing current epoch"
                )
            if str(assignment_id) not in {
                str(value) for value in (epoch["assignment_ids"] or [])
            }:
                raise WorkerProtocolError(
                    "worker step assignment is not in the current committed epoch"
                )

            cur.execute(
                """
                SELECT
                    assignment_id, task_id, task_revision,
                    created_epoch_sequence, reservation_ids, status
                FROM attention_assignments
                WHERE scheduler_key = %s AND assignment_id = %s
                """,
                (scheduler_key, assignment_id),
            )
            assignment_row = cur.fetchone()
            if assignment_row is None:
                raise KeyError(assignment_id)
            assignment = _row_to_assignment(assignment_row)
            if assignment.status is not AssignmentStatus.READY:
                raise WorkerProtocolError("worker step assignment is not READY")

            cur.execute(
                """
                SELECT revision, required_capabilities
                FROM attention_tasks
                WHERE task_id = %s
                """,
                (assignment.task_id,),
            )
            task = cur.fetchone()
            if task is None:
                raise WorkerProtocolError("worker assignment references a missing task")
            if int(task["revision"]) != assignment.task_revision:
                raise WorkerProtocolError(
                    "worker assignment uses a stale task revision"
                )
            declared_capabilities = {
                str(value) for value in (task["required_capabilities"] or [])
            }
            if normalized_capability not in declared_capabilities:
                raise WorkerProtocolError(
                    "worker capability is not declared by the durable task"
                )

            step_id = deterministic_worker_step_id(assignment_id, normalized_key)
            step = WorkerStep(
                protocol_version=WORKER_PROTOCOL_VERSION,
                step_id=step_id,
                assignment_id=assignment_id,
                task_id=assignment.task_id,
                task_revision=assignment.task_revision,
                created_epoch_sequence=assignment.created_epoch_sequence,
                step_key=normalized_key,
                capability=normalized_capability,
                reservation_ids=list(assignment.reservation_ids),
                input_refs=list(input_refs or []),
                idempotency_key=deterministic_worker_idempotency_key(step_id),
                effect_policy=effect_policy,
            )
            _insert_immutable_step(cur, scheduler_key=scheduler_key, step=step)
        conn.commit()
        return step.model_copy(deep=True)
    except Exception:
        conn.rollback()
        raise


def guarded_claim_worker_step(
    conn: psycopg.Connection,
    *,
    step_id: UUID,
    worker_id: str,
    probe: HostResourceProbe | None = None,
    policy: ResourceSafetyPolicy | None = None,
    clock: Callable[[], datetime] | None = None,
    lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerClaimAttempt:
    """Re-observe host pressure and atomically grant at most one live lease.

    This is the sole persistence path that turns a READY assignment into an
    executable Prometheist worker claim.  The host probe runs immediately
    before the transaction; the exact bounded observation and decision are
    persisted whether the claim is granted or denied.
    """

    normalized_worker_id = worker_id.strip()
    if not normalized_worker_id:
        raise ValueError("worker_id must not be empty")
    _validate_lease_seconds(lease_seconds)
    expected_policy = policy
    current_time = clock or (lambda: datetime.now(timezone.utc))
    host_probe = probe or SystemHostResourceProbe()
    errors: list[str] = []
    try:
        metrics = host_probe.capture()
    except Exception as exc:  # every probe failure is a persisted denial input
        metrics = None
        errors.append(f"{type(exc).__name__}: {exc}")
    captured_at = _as_utc(current_time())

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            state = _lock_scheduler_state(cur, scheduler_key)
            committed_observation = _load_current_resource_observation(
                cur,
                scheduler_key=scheduler_key,
                observation_id=state["current_resource_observation_id"],
            )
            resource_policy = (
                committed_observation.policy.model_copy(deep=True)
                if committed_observation is not None
                else (expected_policy or ResourceSafetyPolicy()).model_copy(
                    deep=True
                )
            )
            cur.execute(
                """
                SELECT *
                FROM attention_worker_steps
                WHERE scheduler_key = %s AND step_id = %s
                FOR UPDATE
                """,
                (scheduler_key, step_id),
            )
            step_row = cur.fetchone()
            if step_row is None:
                raise KeyError(step_id)
            step = _row_to_step(step_row)

            # Match the scheduler's lock order: scheduler state first, then
            # globally ordered physical resources.  This serializes competing
            # claims without allowing worker timing to choose a winner.
            cur.execute(
                """
                SELECT
                    resource_id, resource_class, capacity,
                    system_headroom, enabled, metadata
                FROM attention_execution_resources
                ORDER BY resource_id ASC
                FOR UPDATE
                """
            )
            resources = [_row_to_resource(row) for row in cur.fetchall()]

            cur.execute(
                """
                UPDATE attention_worker_claims
                SET status = 'ABANDONED', updated_at = %s
                WHERE status = 'ACTIVE' AND lease_expires_at <= %s
                """,
                (captured_at, captured_at),
            )

            checkpoint = _load_latest_checkpoint(
                cur,
                scheduler_key=scheduler_key,
                step_id=step.step_id,
            )
            existing_result = _load_result_row(
                cur,
                scheduler_key=scheduler_key,
                step_id=step.step_id,
            )
            latest_claim = _load_latest_claim(
                cur,
                scheduler_key=scheduler_key,
                step_id=step.step_id,
            )
            active_claims = _load_active_claim_rows(cur, captured_at=captured_at)
            active_claim_ids = sorted(
                (UUID(str(row["claim_id"])) for row in active_claims),
                key=lambda value: value.hex,
            )
            active_reservations = _load_claim_reservations(
                cur,
                active_claims=active_claims,
            )

            scheduler_cycle = max(1, int(state["cycle"]))
            snapshot = build_resource_observation(
                scheduler_cycle=scheduler_cycle,
                captured_at=captured_at,
                resources=resources,
                reservations=active_reservations,
                policy=resource_policy,
                metrics=metrics,
                probe_errors=errors,
                host_id=f"worker-claim:{scheduler_key}",
            )

            reason = _structural_claim_denial(
                cur,
                state=state,
                step=step,
                latest_claim=latest_claim,
                existing_result=existing_result,
                captured_at=captured_at,
                resource_policy=resource_policy,
                expected_policy=expected_policy,
                committed_observation=committed_observation,
                scheduler_key=scheduler_key,
            )
            target_reservations: list[ResourceReservation] = []
            if reason is None:
                target_reservations = _load_target_reservations(
                    cur,
                    scheduler_key=scheduler_key,
                    step=step,
                )
                reason = _resource_claim_denial(
                    snapshot=snapshot,
                    target_reservations=target_reservations,
                )

            evaluated_at = _as_utc(current_time())
            if evaluated_at < snapshot.captured_at:
                reason = "claim-time clock moved backward before publication"
            elif evaluated_at > snapshot.valid_until:
                reason = "claim-time resource observation expired before publication"
            decision = (
                WorkerClaimDecision.GRANTED
                if reason is None
                else WorkerClaimDecision.DENIED
            )
            observation = _make_claim_observation(
                step=step,
                snapshot=snapshot,
                evaluated_at=evaluated_at,
                decision=decision,
                reason=reason or "fresh resource gate admitted the worker step",
                active_claim_ids=active_claim_ids,
            )
            if decision is WorkerClaimDecision.GRANTED:
                observation.assert_fresh(at=evaluated_at)
            _insert_immutable_claim_observation(
                cur,
                scheduler_key=scheduler_key,
                observation=observation,
            )

            if decision is WorkerClaimDecision.DENIED:
                attempt = WorkerClaimAttempt(observation=observation)
            else:
                cur.execute(
                    """
                    SELECT coalesce(max(attempt), 0) AS latest_attempt
                    FROM attention_worker_claims
                    WHERE scheduler_key = %s AND step_id = %s
                    """,
                    (scheduler_key, step.step_id),
                )
                attempt_number = int(
                    require_row(cur.fetchone(), context="latest worker claim attempt")["latest_attempt"]
                ) + 1
                claim = WorkerClaim(
                    claim_id=deterministic_worker_claim_id(
                        step.step_id,
                        attempt_number,
                    ),
                    step_id=step.step_id,
                    assignment_id=step.assignment_id,
                    task_id=step.task_id,
                    worker_id=normalized_worker_id,
                    attempt=attempt_number,
                    checkpoint_revision=(checkpoint.revision if checkpoint else 0),
                    observation_id=observation.observation_id,
                    claimed_at=evaluated_at,
                    lease_expires_at=evaluated_at
                    + timedelta(seconds=lease_seconds),
                    last_heartbeat_at=evaluated_at,
                    status=WorkerClaimStatus.ACTIVE,
                )
                _insert_claim(cur, scheduler_key=scheduler_key, claim=claim)
                envelope = WorkerClaimEnvelope(
                    step=step,
                    claim=claim,
                    checkpoint=checkpoint,
                )
                attempt = WorkerClaimAttempt(
                    observation=observation,
                    envelope=envelope,
                )
        conn.commit()
        return attempt
    except Exception:
        conn.rollback()
        raise


def heartbeat_worker_claim(
    conn: psycopg.Connection,
    *,
    claim_id: UUID,
    worker_id: str,
    lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
    clock: Callable[[], datetime] | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerClaim:
    """Extend one still-live lease; an expired lease becomes abandoned."""

    _validate_lease_seconds(lease_seconds)
    at = _as_utc((clock or (lambda: datetime.now(timezone.utc)))())
    expired = False
    updated: WorkerClaim | None = None
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            claim = _lock_claim(cur, scheduler_key=scheduler_key, claim_id=claim_id)
            _require_worker(claim, worker_id)
            if claim.status is not WorkerClaimStatus.ACTIVE:
                raise WorkerProtocolError("only an ACTIVE worker claim may heartbeat")
            if at >= claim.lease_expires_at:
                _set_claim_status(
                    cur,
                    scheduler_key=scheduler_key,
                    claim_id=claim_id,
                    status=WorkerClaimStatus.ABANDONED,
                    at=at,
                )
                expired = True
            else:
                cur.execute(
                    """
                    UPDATE attention_worker_claims
                    SET last_heartbeat_at = %s, lease_expires_at = %s,
                        updated_at = %s
                    WHERE scheduler_key = %s AND claim_id = %s
                    RETURNING *
                    """,
                    (
                        at,
                        at + timedelta(seconds=lease_seconds),
                        at,
                        scheduler_key,
                        claim_id,
                    ),
                )
                updated = _row_to_claim(
                    require_row(cur.fetchone(), context="heartbeat-extended worker claim")
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if expired:
        raise WorkerProtocolError("worker claim lease has expired")
    if updated is None:
        raise WorkerProtocolError("worker claim heartbeat did not return a durable claim")
    return updated


def checkpoint_worker_claim(
    conn: psycopg.Connection,
    *,
    claim_id: UUID,
    worker_id: str,
    state: dict[str, Any],
    output_refs: list[str] | None = None,
    clock: Callable[[], datetime] | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerCheckpoint:
    """Commit one safe checkpoint and relinquish the worker lease."""

    at = _as_utc((clock or (lambda: datetime.now(timezone.utc)))())
    expired = False
    checkpoint: WorkerCheckpoint | None = None
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            claim = _lock_claim(cur, scheduler_key=scheduler_key, claim_id=claim_id)
            _require_worker(claim, worker_id)
            if claim.status is WorkerClaimStatus.CHECKPOINTED:
                checkpoint = _load_checkpoint_for_claim(
                    cur,
                    scheduler_key=scheduler_key,
                    claim_id=claim_id,
                )
                if checkpoint is None:
                    raise WorkerProtocolError(
                        "checkpointed claim has no durable checkpoint"
                    )
                candidate = checkpoint.model_copy(
                    update={
                        "state": dict(state),
                        "output_refs": list(output_refs or []),
                    },
                    deep=True,
                )
                if candidate.model_dump(mode="json") != checkpoint.model_dump(
                    mode="json"
                ):
                    raise WorkerProtocolError("conflicting idempotent checkpoint retry")
            elif claim.status is not WorkerClaimStatus.ACTIVE:
                raise WorkerProtocolError("only an ACTIVE worker claim may checkpoint")
            elif at >= claim.lease_expires_at:
                _set_claim_status(
                    cur,
                    scheduler_key=scheduler_key,
                    claim_id=claim_id,
                    status=WorkerClaimStatus.ABANDONED,
                    at=at,
                )
                expired = True
            else:
                revision = claim.checkpoint_revision + 1
                checkpoint = WorkerCheckpoint(
                    checkpoint_id=deterministic_worker_checkpoint_id(
                        claim.step_id,
                        revision,
                    ),
                    step_id=claim.step_id,
                    claim_id=claim.claim_id,
                    revision=revision,
                    state=dict(state),
                    output_refs=list(output_refs or []),
                    created_at=at,
                )
                _insert_immutable_checkpoint(
                    cur,
                    scheduler_key=scheduler_key,
                    checkpoint=checkpoint,
                )
                _set_claim_status(
                    cur,
                    scheduler_key=scheduler_key,
                    claim_id=claim_id,
                    status=WorkerClaimStatus.CHECKPOINTED,
                    at=at,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if expired:
        raise WorkerProtocolError("worker claim lease has expired")
    if checkpoint is None:
        raise WorkerProtocolError("worker checkpoint transaction produced no state")
    return checkpoint


def complete_worker_claim(
    conn: psycopg.Connection,
    *,
    claim_id: UUID,
    worker_id: str,
    output: dict[str, Any],
    output_refs: list[str] | None = None,
    clock: Callable[[], datetime] | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerResult:
    """Persist exactly one terminal result and close the leased claim."""

    at = _as_utc((clock or (lambda: datetime.now(timezone.utc)))())
    expired = False
    result: WorkerResult | None = None
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            claim = _lock_claim(cur, scheduler_key=scheduler_key, claim_id=claim_id)
            _require_worker(claim, worker_id)
            step = _load_step(
                cur,
                scheduler_key=scheduler_key,
                step_id=claim.step_id,
            )
            existing = _load_result_row(
                cur,
                scheduler_key=scheduler_key,
                step_id=claim.step_id,
            )
            if existing is not None:
                if (
                    existing.claim_id != claim.claim_id
                    or existing.output != output
                    or existing.output_refs != list(output_refs or [])
                ):
                    raise WorkerProtocolError("conflicting worker result retry")
                result = existing
            elif claim.status is not WorkerClaimStatus.ACTIVE:
                raise WorkerProtocolError("only an ACTIVE worker claim may complete")
            elif at >= claim.lease_expires_at:
                _set_claim_status(
                    cur,
                    scheduler_key=scheduler_key,
                    claim_id=claim_id,
                    status=WorkerClaimStatus.ABANDONED,
                    at=at,
                )
                expired = True
            else:
                result = WorkerResult(
                    result_id=deterministic_worker_result_id(step.step_id),
                    step_id=step.step_id,
                    claim_id=claim.claim_id,
                    checkpoint_revision=claim.checkpoint_revision,
                    idempotency_key=step.idempotency_key,
                    output=dict(output),
                    output_refs=list(output_refs or []),
                    completed_at=at,
                )
                _insert_immutable_result(
                    cur,
                    scheduler_key=scheduler_key,
                    result=result,
                )
                _set_claim_status(
                    cur,
                    scheduler_key=scheduler_key,
                    claim_id=claim_id,
                    status=WorkerClaimStatus.COMPLETED,
                    at=at,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    if expired:
        raise WorkerProtocolError("worker claim lease has expired")
    if result is None:
        raise WorkerProtocolError("worker completion transaction produced no result")
    return result


def release_worker_claim(
    conn: psycopg.Connection,
    *,
    claim_id: UUID,
    worker_id: str,
    clock: Callable[[], datetime] | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerClaim:
    """Explicitly relinquish a claim known not to have committed an effect."""

    at = _as_utc((clock or (lambda: datetime.now(timezone.utc)))())
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            claim = _lock_claim(cur, scheduler_key=scheduler_key, claim_id=claim_id)
            _require_worker(claim, worker_id)
            if claim.status is WorkerClaimStatus.RELEASED:
                released = claim
            elif claim.status is not WorkerClaimStatus.ACTIVE:
                raise WorkerProtocolError("only an ACTIVE worker claim may be released")
            else:
                _set_claim_status(
                    cur,
                    scheduler_key=scheduler_key,
                    claim_id=claim_id,
                    status=WorkerClaimStatus.RELEASED,
                    at=at,
                )
                cur.execute(
                    """
                    SELECT * FROM attention_worker_claims
                    WHERE scheduler_key = %s AND claim_id = %s
                    """,
                    (scheduler_key, claim_id),
                )
                released = _row_to_claim(
                    require_row(cur.fetchone(), context="released worker claim")
                )
        conn.commit()
        return released
    except Exception:
        conn.rollback()
        raise


def load_worker_step(
    conn: psycopg.Connection,
    step_id: UUID,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerStep:
    with conn.cursor(row_factory=dict_row) as cur:
        return _load_step(cur, scheduler_key=scheduler_key, step_id=step_id)


def load_worker_claim(
    conn: psycopg.Connection,
    claim_id: UUID,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerClaim:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM attention_worker_claims
            WHERE scheduler_key = %s AND claim_id = %s
            """,
            (scheduler_key, claim_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(claim_id)
        return _row_to_claim(row)


def load_worker_claim_envelope(
    conn: psycopg.Connection,
    claim_id: UUID,
    *,
    worker_id: str,
    clock: Callable[[], datetime] | None = None,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerClaimEnvelope:
    """Load the complete bounded input for the process that owns a live claim."""

    at = _as_utc((clock or (lambda: datetime.now(timezone.utc)))())
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM attention_worker_claims
            WHERE scheduler_key = %s AND claim_id = %s
            """,
            (scheduler_key, claim_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(claim_id)
        claim = _row_to_claim(row)
        _require_worker(claim, worker_id)
        if claim.status is not WorkerClaimStatus.ACTIVE:
            raise WorkerProtocolError("worker claim is not ACTIVE")
        if at >= claim.lease_expires_at:
            raise WorkerProtocolError("worker claim lease has expired")
        step = _load_step(
            cur,
            scheduler_key=scheduler_key,
            step_id=claim.step_id,
        )
        checkpoint = _load_latest_checkpoint(
            cur,
            scheduler_key=scheduler_key,
            step_id=claim.step_id,
        )
    return WorkerClaimEnvelope(
        step=step,
        claim=claim,
        checkpoint=checkpoint,
    )


def load_worker_result(
    conn: psycopg.Connection,
    step_id: UUID,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> WorkerResult | None:
    with conn.cursor(row_factory=dict_row) as cur:
        return _load_result_row(
            cur,
            scheduler_key=scheduler_key,
            step_id=step_id,
        )


def load_claim_observations(
    conn: psycopg.Connection,
    step_id: UUID,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> list[WorkerClaimResourceObservation]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT snapshot
            FROM attention_worker_claim_observations
            WHERE scheduler_key = %s AND step_id = %s
            ORDER BY captured_at ASC, evaluated_at ASC, observation_id ASC
            """,
            (scheduler_key, step_id),
        )
        return [
            WorkerClaimResourceObservation.model_validate(row["snapshot"])
            for row in cur.fetchall()
        ]


def _lock_scheduler_state(cur: Any, scheduler_key: str) -> dict[str, Any]:
    cur.execute(
        """
        SELECT
            cycle, current_epoch_id, resource_safety_required,
            resource_safety_policy_version,
            current_resource_observation_id
        FROM attention_scheduler_state
        WHERE scheduler_key = %s
        FOR UPDATE
        """,
        (scheduler_key,),
    )
    state = cur.fetchone()
    if state is None:
        raise WorkerProtocolError("worker protocol requires durable scheduler state")
    return state


def _structural_claim_denial(
    cur: Any,
    *,
    state: dict[str, Any],
    step: WorkerStep,
    latest_claim: WorkerClaim | None,
    existing_result: WorkerResult | None,
    captured_at: datetime,
    resource_policy: ResourceSafetyPolicy,
    expected_policy: ResourceSafetyPolicy | None,
    committed_observation: ResourceObservationSnapshot | None,
    scheduler_key: str,
) -> str | None:
    if not bool(state["resource_safety_required"]):
        return "worker launch requires host resource safety"
    if state["resource_safety_policy_version"] != resource_policy.policy_version:
        return "claim resource policy does not match the committed scheduler policy"
    if committed_observation is None:
        return "worker launch requires a committed resource safety policy"
    if (
        committed_observation.observation_id
        != state["current_resource_observation_id"]
    ):
        return "scheduler resource policy observation is inconsistent"
    if (
        expected_policy is not None
        and expected_policy.model_dump(mode="json")
        != committed_observation.policy.model_dump(mode="json")
    ):
        return "claim resource policy does not match the exact committed policy"
    if state["current_epoch_id"] is None:
        return "worker assignment is not part of a committed epoch"
    cur.execute(
        """
        SELECT
            assignment_ids, resource_observation_id,
            resource_safety_policy_version
        FROM attention_scheduling_epochs
        WHERE scheduler_key = %s AND epoch_id = %s
        """,
        (scheduler_key, state["current_epoch_id"]),
    )
    epoch = cur.fetchone()
    if epoch is None:
        return "scheduler current epoch is missing"
    if (
        epoch["resource_observation_id"] != committed_observation.observation_id
        or epoch["resource_safety_policy_version"]
        != committed_observation.safety_policy_version
    ):
        return "worker assignment epoch does not bind the committed resource policy"
    if str(step.assignment_id) not in {
        str(value) for value in (epoch["assignment_ids"] or [])
    }:
        return "worker assignment is no longer current"
    cur.execute(
        "SELECT revision FROM attention_tasks WHERE task_id = %s",
        (step.task_id,),
    )
    task = cur.fetchone()
    if task is None or int(task["revision"]) != step.task_revision:
        return "worker step uses a stale task revision"
    if existing_result is not None:
        return "worker step already has a terminal result"
    if latest_claim is not None:
        if (
            latest_claim.status is WorkerClaimStatus.ACTIVE
            and latest_claim.lease_expires_at > captured_at
        ):
            return "worker step already has a live claim"
        if (
            latest_claim.status is WorkerClaimStatus.ABANDONED
            and not step.effect_policy.abandoned_retry_is_safe
        ):
            return "abandoned at-most-once effect requires explicit reconciliation"
    cur.execute(
        """
        SELECT claim_id
        FROM attention_worker_claims
        WHERE scheduler_key = %s
          AND assignment_id = %s
          AND status = 'ACTIVE'
          AND lease_expires_at > %s
        LIMIT 1
        """,
        (scheduler_key, step.assignment_id, captured_at),
    )
    if cur.fetchone() is not None:
        return "worker assignment already has a live claim"
    return None


def _load_current_resource_observation(
    cur: Any,
    *,
    scheduler_key: str,
    observation_id: UUID | None,
) -> ResourceObservationSnapshot | None:
    if observation_id is None:
        return None
    cur.execute(
        """
        SELECT snapshot
        FROM attention_resource_observations
        WHERE scheduler_key = %s AND observation_id = %s
        """,
        (scheduler_key, observation_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return ResourceObservationSnapshot.model_validate(row["snapshot"])


def _resource_claim_denial(
    *,
    snapshot: ResourceObservationSnapshot,
    target_reservations: list[ResourceReservation],
) -> str | None:
    if not snapshot.healthy:
        return "claim-time host resource probe failed closed"
    capacity_by_id = snapshot.capacity_by_resource_id()
    for reservation in target_reservations:
        capacity = capacity_by_id.get(reservation.resource_id)
        if capacity is None:
            return f"claim resource {reservation.resource_id!r} is unobserved"
        if capacity.available_for_new_work < reservation.units:
            return (
                f"claim resource {reservation.resource_id!r} lacks safe "
                f"capacity ({capacity.available_for_new_work} < {reservation.units})"
            )
    return None


def _make_claim_observation(
    *,
    step: WorkerStep,
    snapshot: ResourceObservationSnapshot,
    evaluated_at: datetime,
    decision: WorkerClaimDecision,
    reason: str,
    active_claim_ids: list[UUID],
) -> WorkerClaimResourceObservation:
    provisional = WorkerClaimResourceObservation.model_construct(
        observation_id=UUID(int=0),
        step_id=step.step_id,
        assignment_id=step.assignment_id,
        captured_at=snapshot.captured_at,
        evaluated_at=evaluated_at,
        valid_until=snapshot.valid_until,
        claim_policy_version=WORKER_CLAIM_POLICY_VERSION,
        resource_safety_policy_version=snapshot.safety_policy_version,
        decision=decision,
        reason=reason,
        active_claim_ids=active_claim_ids,
        target_reservation_ids=list(step.reservation_ids),
        resource_snapshot=snapshot.model_copy(deep=True),
    )
    return WorkerClaimResourceObservation(
        **provisional.model_dump(exclude={"observation_id"}),
        observation_id=deterministic_worker_claim_observation_id(provisional),
    )


def _load_active_claim_rows(cur: Any, *, captured_at: datetime) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT c.claim_id, c.scheduler_key, c.assignment_id, s.reservation_ids
        FROM attention_worker_claims c
        JOIN attention_worker_steps s
          ON s.scheduler_key = c.scheduler_key AND s.step_id = c.step_id
        WHERE c.status = 'ACTIVE' AND c.lease_expires_at > %s
        ORDER BY c.claim_id ASC
        """,
        (captured_at,),
    )
    return list(cur.fetchall())


def _load_claim_reservations(
    cur: Any,
    *,
    active_claims: list[dict[str, Any]],
) -> list[ResourceReservation]:
    reservation_ids = sorted(
        {
            UUID(str(value))
            for claim in active_claims
            for value in (claim["reservation_ids"] or [])
        },
        key=lambda value: value.hex,
    )
    if not reservation_ids:
        return []
    cur.execute(
        """
        SELECT reservation_id, task_id, resource_id, resource_class, units
        FROM attention_resource_reservations
        WHERE reservation_id = ANY(%s)
        ORDER BY resource_id ASC, reservation_id ASC
        """,
        (reservation_ids,),
    )
    rows = cur.fetchall()
    if {UUID(str(row["reservation_id"])) for row in rows} != set(reservation_ids):
        raise WorkerProtocolError(
            "an active worker claim references a released reservation"
        )
    return [_row_to_reservation(row) for row in rows]


def _load_target_reservations(
    cur: Any,
    *,
    scheduler_key: str,
    step: WorkerStep,
) -> list[ResourceReservation]:
    cur.execute(
        """
        SELECT reservation_id, task_id, resource_id, resource_class, units
        FROM attention_resource_reservations
        WHERE scheduler_key = %s AND reservation_id = ANY(%s)
        ORDER BY resource_id ASC, reservation_id ASC
        """,
        (scheduler_key, step.reservation_ids),
    )
    rows = cur.fetchall()
    if {UUID(str(row["reservation_id"])) for row in rows} != set(
        step.reservation_ids
    ):
        raise WorkerProtocolError(
            "worker step does not retain its complete current reservation set"
        )
    reservations = [_row_to_reservation(row) for row in rows]
    if any(reservation.task_id != step.task_id for reservation in reservations):
        raise WorkerProtocolError("worker step reservation belongs to another task")
    return reservations


def _insert_immutable_step(cur: Any, *, scheduler_key: str, step: WorkerStep) -> None:
    cur.execute(
        """
        INSERT INTO attention_worker_steps (
            scheduler_key, step_id, assignment_id, task_id, task_revision,
            created_epoch_sequence, protocol_version, step_key, capability,
            reservation_ids, input_refs, idempotency_key, effect_policy
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            step.step_id,
            step.assignment_id,
            step.task_id,
            step.task_revision,
            step.created_epoch_sequence,
            step.protocol_version,
            step.step_key,
            step.capability,
            Json([str(value) for value in step.reservation_ids]),
            Json(step.input_refs),
            step.idempotency_key,
            step.effect_policy.value,
        ),
    )
    stored = _load_step(cur, scheduler_key=scheduler_key, step_id=step.step_id)
    if stored.model_dump(mode="json") != step.model_dump(mode="json"):
        raise WorkerProtocolError("conflicting immutable worker step")


def _insert_immutable_claim_observation(
    cur: Any,
    *,
    scheduler_key: str,
    observation: WorkerClaimResourceObservation,
) -> None:
    cur.execute(
        """
        INSERT INTO attention_worker_claim_observations (
            scheduler_key, observation_id, step_id, assignment_id,
            captured_at, evaluated_at, valid_until, claim_policy_version,
            resource_safety_policy_version, decision, reason, snapshot
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            observation.observation_id,
            observation.step_id,
            observation.assignment_id,
            observation.captured_at,
            observation.evaluated_at,
            observation.valid_until,
            observation.claim_policy_version,
            observation.resource_safety_policy_version,
            observation.decision.value,
            observation.reason,
            Json(observation.model_dump(mode="json")),
        ),
    )
    cur.execute(
        """
        SELECT snapshot
        FROM attention_worker_claim_observations
        WHERE scheduler_key = %s AND observation_id = %s
        """,
        (scheduler_key, observation.observation_id),
    )
    row = cur.fetchone()
    if row is None:
        raise WorkerProtocolError("claim observation insert produced no row")
    stored = WorkerClaimResourceObservation.model_validate(row["snapshot"])
    if stored.model_dump(mode="json") != observation.model_dump(mode="json"):
        raise WorkerProtocolError("conflicting immutable claim observation")


def _insert_claim(cur: Any, *, scheduler_key: str, claim: WorkerClaim) -> None:
    cur.execute(
        """
        INSERT INTO attention_worker_claims (
            scheduler_key, claim_id, step_id, assignment_id, task_id,
            worker_id, attempt, checkpoint_revision, observation_id,
            claimed_at, lease_expires_at, last_heartbeat_at, status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            scheduler_key,
            claim.claim_id,
            claim.step_id,
            claim.assignment_id,
            claim.task_id,
            claim.worker_id,
            claim.attempt,
            claim.checkpoint_revision,
            claim.observation_id,
            claim.claimed_at,
            claim.lease_expires_at,
            claim.last_heartbeat_at,
            claim.status.value,
        ),
    )


def _insert_immutable_checkpoint(
    cur: Any,
    *,
    scheduler_key: str,
    checkpoint: WorkerCheckpoint,
) -> None:
    cur.execute(
        """
        INSERT INTO attention_worker_checkpoints (
            scheduler_key, checkpoint_id, step_id, claim_id,
            revision, state, output_refs, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            checkpoint.checkpoint_id,
            checkpoint.step_id,
            checkpoint.claim_id,
            checkpoint.revision,
            Json(checkpoint.state),
            Json(checkpoint.output_refs),
            checkpoint.created_at,
        ),
    )
    cur.execute(
        """
        SELECT * FROM attention_worker_checkpoints
        WHERE scheduler_key = %s AND checkpoint_id = %s
        """,
        (scheduler_key, checkpoint.checkpoint_id),
    )
    stored = _row_to_checkpoint(cur.fetchone())
    if stored.model_dump(mode="json") != checkpoint.model_dump(mode="json"):
        raise WorkerProtocolError("conflicting immutable worker checkpoint")


def _insert_immutable_result(
    cur: Any,
    *,
    scheduler_key: str,
    result: WorkerResult,
) -> None:
    cur.execute(
        """
        INSERT INTO attention_worker_results (
            scheduler_key, result_id, step_id, claim_id,
            checkpoint_revision, idempotency_key, output,
            output_refs, completed_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            scheduler_key,
            result.result_id,
            result.step_id,
            result.claim_id,
            result.checkpoint_revision,
            result.idempotency_key,
            Json(result.output),
            Json(result.output_refs),
            result.completed_at,
        ),
    )
    stored = _load_result_row(
        cur,
        scheduler_key=scheduler_key,
        step_id=result.step_id,
    )
    if stored is None or stored.model_dump(mode="json") != result.model_dump(
        mode="json"
    ):
        raise WorkerProtocolError("conflicting immutable worker result")


def _set_claim_status(
    cur: Any,
    *,
    scheduler_key: str,
    claim_id: UUID,
    status: WorkerClaimStatus,
    at: datetime,
) -> None:
    cur.execute(
        """
        UPDATE attention_worker_claims
        SET status = %s, updated_at = %s
        WHERE scheduler_key = %s AND claim_id = %s
        """,
        (status.value, at, scheduler_key, claim_id),
    )
    if cur.rowcount != 1:
        raise WorkerProtocolError("worker claim status update affected no row")


def _lock_claim(cur: Any, *, scheduler_key: str, claim_id: UUID) -> WorkerClaim:
    cur.execute(
        """
        SELECT * FROM attention_worker_claims
        WHERE scheduler_key = %s AND claim_id = %s
        FOR UPDATE
        """,
        (scheduler_key, claim_id),
    )
    row = cur.fetchone()
    if row is None:
        raise KeyError(claim_id)
    return _row_to_claim(row)


def _load_step(cur: Any, *, scheduler_key: str, step_id: UUID) -> WorkerStep:
    cur.execute(
        """
        SELECT * FROM attention_worker_steps
        WHERE scheduler_key = %s AND step_id = %s
        """,
        (scheduler_key, step_id),
    )
    row = cur.fetchone()
    if row is None:
        raise KeyError(step_id)
    return _row_to_step(row)


def _load_latest_claim(
    cur: Any,
    *,
    scheduler_key: str,
    step_id: UUID,
) -> WorkerClaim | None:
    cur.execute(
        """
        SELECT * FROM attention_worker_claims
        WHERE scheduler_key = %s AND step_id = %s
        ORDER BY attempt DESC
        LIMIT 1
        """,
        (scheduler_key, step_id),
    )
    row = cur.fetchone()
    return _row_to_claim(row) if row is not None else None


def _load_latest_checkpoint(
    cur: Any,
    *,
    scheduler_key: str,
    step_id: UUID,
) -> WorkerCheckpoint | None:
    cur.execute(
        """
        SELECT * FROM attention_worker_checkpoints
        WHERE scheduler_key = %s AND step_id = %s
        ORDER BY revision DESC
        LIMIT 1
        """,
        (scheduler_key, step_id),
    )
    row = cur.fetchone()
    return _row_to_checkpoint(row) if row is not None else None


def _load_checkpoint_for_claim(
    cur: Any,
    *,
    scheduler_key: str,
    claim_id: UUID,
) -> WorkerCheckpoint | None:
    cur.execute(
        """
        SELECT * FROM attention_worker_checkpoints
        WHERE scheduler_key = %s AND claim_id = %s
        """,
        (scheduler_key, claim_id),
    )
    row = cur.fetchone()
    return _row_to_checkpoint(row) if row is not None else None


def _load_result_row(
    cur: Any,
    *,
    scheduler_key: str,
    step_id: UUID,
) -> WorkerResult | None:
    cur.execute(
        """
        SELECT * FROM attention_worker_results
        WHERE scheduler_key = %s AND step_id = %s
        """,
        (scheduler_key, step_id),
    )
    row = cur.fetchone()
    return _row_to_result(row) if row is not None else None


def _row_to_assignment(row: dict[str, Any]) -> DurableAssignment:
    return DurableAssignment(
        assignment_id=row["assignment_id"],
        task_id=row["task_id"],
        task_revision=int(row["task_revision"]),
        created_epoch_sequence=int(row["created_epoch_sequence"]),
        reservation_ids=[UUID(str(value)) for value in row["reservation_ids"]],
        status=AssignmentStatus(row["status"]),
    )


def _row_to_step(row: dict[str, Any]) -> WorkerStep:
    return WorkerStep(
        protocol_version=row["protocol_version"],
        step_id=row["step_id"],
        assignment_id=row["assignment_id"],
        task_id=row["task_id"],
        task_revision=int(row["task_revision"]),
        created_epoch_sequence=int(row["created_epoch_sequence"]),
        step_key=row["step_key"],
        capability=row["capability"],
        reservation_ids=[UUID(str(value)) for value in row["reservation_ids"]],
        input_refs=list(row["input_refs"] or []),
        idempotency_key=row["idempotency_key"],
        effect_policy=WorkerEffectPolicy(row["effect_policy"]),
    )


def _row_to_claim(row: dict[str, Any]) -> WorkerClaim:
    return WorkerClaim(
        claim_id=row["claim_id"],
        step_id=row["step_id"],
        assignment_id=row["assignment_id"],
        task_id=row["task_id"],
        worker_id=row["worker_id"],
        attempt=int(row["attempt"]),
        checkpoint_revision=int(row["checkpoint_revision"]),
        observation_id=row["observation_id"],
        claimed_at=row["claimed_at"],
        lease_expires_at=row["lease_expires_at"],
        last_heartbeat_at=row["last_heartbeat_at"],
        status=WorkerClaimStatus(row["status"]),
    )


def _row_to_checkpoint(row: dict[str, Any]) -> WorkerCheckpoint:
    return WorkerCheckpoint(
        checkpoint_id=row["checkpoint_id"],
        step_id=row["step_id"],
        claim_id=row["claim_id"],
        revision=int(row["revision"]),
        state=dict(row["state"] or {}),
        output_refs=list(row["output_refs"] or []),
        created_at=row["created_at"],
    )


def _row_to_result(row: dict[str, Any]) -> WorkerResult:
    return WorkerResult(
        result_id=row["result_id"],
        step_id=row["step_id"],
        claim_id=row["claim_id"],
        checkpoint_revision=int(row["checkpoint_revision"]),
        idempotency_key=row["idempotency_key"],
        output=dict(row["output"] or {}),
        output_refs=list(row["output_refs"] or []),
        completed_at=row["completed_at"],
    )


def _row_to_resource(row: dict[str, Any]) -> ExecutionResource:
    return ExecutionResource(
        resource_id=row["resource_id"],
        resource_class=ExecutionResourceClass(row["resource_class"]),
        capacity=int(row["capacity"]),
        system_headroom=int(row["system_headroom"]),
        enabled=bool(row["enabled"]),
        metadata=dict(row["metadata"] or {}),
    )


def _row_to_reservation(row: dict[str, Any]) -> ResourceReservation:
    return ResourceReservation(
        reservation_id=row["reservation_id"],
        task_id=row["task_id"],
        resource_id=row["resource_id"],
        resource_class=ExecutionResourceClass(row["resource_class"]),
        units=int(row["units"]),
    )


def _require_worker(claim: WorkerClaim, worker_id: str) -> None:
    if claim.worker_id != worker_id.strip():
        raise PermissionError("worker id does not own this claim")


def _validate_lease_seconds(value: int) -> None:
    if value < 1 or value > MAX_WORKER_LEASE_SECONDS:
        raise ValueError(
            f"lease_seconds must be between 1 and {MAX_WORKER_LEASE_SECONDS}"
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("worker-store timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)
