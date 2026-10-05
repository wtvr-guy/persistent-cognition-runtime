"""Exclusive Prometheist-owned process-launch boundary for Increment F."""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg

from prometheist.attention_observation import HostResourceProbe, ResourceSafetyPolicy
from prometheist.attention_store import DEFAULT_SCHEDULER_KEY
from prometheist.worker_protocol import (
    WorkerClaimAttempt,
    WorkerClaimDecision,
    WorkerClaimEnvelope,
    WorkerClaimResourceObservation,
)
from prometheist.worker_store import (
    DEFAULT_WORKER_LEASE_SECONDS,
    WorkerProtocolError,
    guarded_claim_worker_step,
    release_worker_claim,
)


DEFAULT_CLAIM_RETRY_ATTEMPTS = 4
DEFAULT_CLAIM_RETRY_DELAY_SECONDS = 1.0


class WorkerLaunchDenied(WorkerProtocolError):
    """Raised when the durable claim-time gate refuses process launch."""

    def __init__(self, observation: WorkerClaimResourceObservation) -> None:
        super().__init__(observation.reason)
        self.observation = observation.model_copy(deep=True)


@dataclass(frozen=True)
class LaunchedWorker:
    """A process handle paired with the durable lease that authorized it."""

    envelope: WorkerClaimEnvelope
    process: Any


ProcessFactory = Callable[..., Any]
SleepFunction = Callable[[float], None]


def _is_transient_resource_denial(
    conn: psycopg.Connection,
    *,
    observation: WorkerClaimResourceObservation,
    scheduler_key: str,
) -> bool:
    """Classify retryability from structured resource state, never reason text.

    Probe failure and a currently insufficient/unobserved reserved resource are
    transient capacity conditions. Structural denials (stale task/epoch,
    policy mismatch, live claim, terminal result, at-most-once ambiguity, etc.)
    do not satisfy this predicate and therefore fail immediately.
    """

    if observation.decision is not WorkerClaimDecision.DENIED:
        return False
    snapshot = observation.resource_snapshot
    if not snapshot.healthy:
        return True
    reservation_ids = list(observation.target_reservation_ids)
    if not reservation_ids:
        return False

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT reservation_id, resource_id, units
            FROM attention_resource_reservations
            WHERE scheduler_key = %s AND reservation_id = ANY(%s)
            ORDER BY resource_id ASC, reservation_id ASC
            """,
            (scheduler_key, reservation_ids),
        )
        rows = cur.fetchall()
    if len(rows) != len(reservation_ids):
        return False

    capacity_by_id = snapshot.capacity_by_resource_id()
    for _reservation_id, resource_id, units in rows:
        capacity = capacity_by_id.get(str(resource_id))
        if capacity is None:
            return True
        if float(capacity.available_for_new_work) < float(units):
            return True
    return False


class GuardedWorkerLauncher:
    """Claim, commit, and only then start a disposable worker process.

    Production code that starts a Prometheist worker belongs behind this class.
    The launcher does not select queue work: callers must register a step from
    an assignment already chosen by the deterministic Attention Fabric.

    Transient resource pressure is not a terminal task failure. The launcher
    re-observes and retries a small bounded number of times while preserving the
    exact same fail-closed resource gate and configured headroom. Structural
    denials are never retried.
    """

    def __init__(
        self,
        connection_factory: Callable[[], psycopg.Connection],
        *,
        probe: HostResourceProbe | None = None,
        policy: ResourceSafetyPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        process_factory: ProcessFactory | None = None,
        scheduler_key: str = DEFAULT_SCHEDULER_KEY,
        claim_retry_attempts: int = DEFAULT_CLAIM_RETRY_ATTEMPTS,
        claim_retry_delay_seconds: float = DEFAULT_CLAIM_RETRY_DELAY_SECONDS,
        sleep: SleepFunction = time.sleep,
    ) -> None:
        if claim_retry_attempts < 1:
            raise ValueError("claim_retry_attempts must be >= 1")
        if claim_retry_delay_seconds < 0:
            raise ValueError("claim_retry_delay_seconds must be >= 0")
        self.connection_factory = connection_factory
        self.probe = probe
        self.policy = policy
        self.clock = clock
        self.process_factory = process_factory or subprocess.Popen
        self.scheduler_key = scheduler_key
        self.claim_retry_attempts = claim_retry_attempts
        self.claim_retry_delay_seconds = claim_retry_delay_seconds
        self.sleep = sleep

    def claim(
        self,
        *,
        step_id: UUID,
        worker_id: str,
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
    ) -> WorkerClaimEnvelope:
        """Return an executable lease or fail after bounded safe re-observation."""

        last_observation: WorkerClaimResourceObservation | None = None
        for retry_index in range(self.claim_retry_attempts):
            conn = self.connection_factory()
            try:
                attempt: WorkerClaimAttempt = guarded_claim_worker_step(
                    conn,
                    step_id=step_id,
                    worker_id=worker_id,
                    probe=self.probe,
                    policy=self.policy,
                    clock=self.clock,
                    lease_seconds=lease_seconds,
                    scheduler_key=self.scheduler_key,
                )
                if attempt.envelope is not None:
                    return attempt.envelope.model_copy(deep=True)
                last_observation = attempt.observation.model_copy(deep=True)
                retryable = _is_transient_resource_denial(
                    conn,
                    observation=attempt.observation,
                    scheduler_key=self.scheduler_key,
                )
            finally:
                conn.close()

            if not retryable or retry_index + 1 >= self.claim_retry_attempts:
                if last_observation is None:
                    raise WorkerProtocolError("guarded claim returned no decision")
                from prometheist.reflexes import record_admission_denial
                with self.connection_factory() as denial_conn:
                    record_admission_denial(denial_conn, last_observation)
                raise WorkerLaunchDenied(last_observation)
            self.sleep(self.claim_retry_delay_seconds)

        raise WorkerProtocolError("guarded claim retry loop exited without a decision")

    def launch(
        self,
        *,
        step_id: UUID,
        worker_id: str,
        command: Sequence[str],
        lease_seconds: int = DEFAULT_WORKER_LEASE_SECONDS,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> LaunchedWorker:
        """Create the durable guarded claim before invoking ``Popen``.

        Shell execution is deliberately unavailable. A failed process spawn
        releases the just-created lease because no worker began executing it.
        """

        if isinstance(command, (str, bytes)):
            raise TypeError("worker command must be a sequence of arguments")
        normalized_command = [str(value) for value in command]
        if not normalized_command or any(not value for value in normalized_command):
            raise ValueError("worker command must contain nonempty arguments")
        envelope = self.claim(
            step_id=step_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        child_env = dict(os.environ if env is None else env)
        child_env.update(
            {
                "PROMETHEIST_WORKER_CLAIM_ID": str(envelope.claim.claim_id),
                "PROMETHEIST_WORKER_STEP_ID": str(envelope.step.step_id),
                "PROMETHEIST_WORKER_ID": envelope.claim.worker_id,
                "PROMETHEIST_WORKER_TASK_ID": str(envelope.step.task_id),
                "PROMETHEIST_WORKER_ASSIGNMENT_ID": str(
                    envelope.step.assignment_id
                ),
                "PROMETHEIST_WORKER_SCHEDULER_KEY": self.scheduler_key,
                "PROMETHEIST_WORKER_IDEMPOTENCY_KEY": (
                    envelope.step.idempotency_key
                ),
                "PROMETHEIST_WORKER_CHECKPOINT_REVISION": str(
                    envelope.claim.checkpoint_revision
                ),
            }
        )
        try:
            process = self.process_factory(
                normalized_command,
                env=child_env,
                cwd=(str(cwd) if cwd is not None else None),
                shell=False,
            )
        except Exception:
            conn = self.connection_factory()
            try:
                release_worker_claim(
                    conn,
                    claim_id=envelope.claim.claim_id,
                    worker_id=worker_id,
                    clock=self.clock,
                    scheduler_key=self.scheduler_key,
                )
            except Exception as release_error:
                raise WorkerProtocolError(
                    "worker process failed to start and its claim could not be released"
                ) from release_error
            finally:
                conn.close()
            raise
        return LaunchedWorker(envelope=envelope, process=process)
