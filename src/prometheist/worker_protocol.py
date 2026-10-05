"""Durable, agent-neutral worker lifecycle primitives for Increment F.

The Attention Fabric owns assignments and resources.  A disposable worker may
borrow one committed assignment only through a guarded lease, advance one
bounded step, persist a checkpoint or result, and disappear.  None of the
models in this module grants scheduling authority to a worker process.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, field_validator, model_validator

from prometheist.attention_observation import ResourceObservationSnapshot


WORKER_PROTOCOL_VERSION = "v0.7-f-worker-v1"
WORKER_CLAIM_POLICY_VERSION = "v0.7-f-guarded-claim-v1"
_WORKER_STEP_NAMESPACE = UUID("28f0ad6c-dd5c-4a7e-9e55-7f6de9231599")
_WORKER_OBSERVATION_NAMESPACE = UUID("0dccacfa-95f6-455c-9d18-6bbdfbd5102b")


class WorkerEffectPolicy(str, Enum):
    """How an abandoned claim may be recovered without duplicating effects."""

    NO_EXTERNAL_EFFECT = "NO_EXTERNAL_EFFECT"
    IDEMPOTENT_WITH_KEY = "IDEMPOTENT_WITH_KEY"
    AT_MOST_ONCE = "AT_MOST_ONCE"

    @property
    def abandoned_retry_is_safe(self) -> bool:
        return self in {
            WorkerEffectPolicy.NO_EXTERNAL_EFFECT,
            WorkerEffectPolicy.IDEMPOTENT_WITH_KEY,
        }


class WorkerClaimStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"
    RELEASED = "RELEASED"


class WorkerClaimDecision(str, Enum):
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class WorkerStep(BaseModel):
    """Immutable contract for one bounded capability step.

    ``idempotency_key`` is stable across every lease attempt for this step.  A
    capability declaring ``IDEMPOTENT_WITH_KEY`` must pass that exact key to
    the external effect boundary rather than minting an attempt-local key.
    """

    protocol_version: str = Field(default=WORKER_PROTOCOL_VERSION, min_length=1)
    step_id: UUID
    assignment_id: UUID
    task_id: UUID
    task_revision: int = Field(ge=0)
    created_epoch_sequence: int = Field(ge=1)
    step_key: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    reservation_ids: list[UUID] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    idempotency_key: str = Field(min_length=1)
    effect_policy: WorkerEffectPolicy = WorkerEffectPolicy.NO_EXTERNAL_EFFECT

    @field_validator("step_key", "capability", "idempotency_key")
    @classmethod
    def normalize_nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("worker-step text fields must not be empty")
        return normalized

    @field_validator("input_refs")
    @classmethod
    def normalize_input_refs(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("input_refs must not contain empty values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("input_refs must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_deterministic_contract(self) -> "WorkerStep":
        if self.protocol_version != WORKER_PROTOCOL_VERSION:
            raise ValueError("worker step protocol version is unsupported")
        if len(self.reservation_ids) != len(set(self.reservation_ids)):
            raise ValueError("reservation_ids must not contain duplicates")
        if self.step_id != deterministic_worker_step_id(
            self.assignment_id,
            self.step_key,
        ):
            raise ValueError("worker step id is not deterministic")
        if self.idempotency_key != deterministic_worker_idempotency_key(
            self.step_id
        ):
            raise ValueError("worker idempotency key is not deterministic")
        return self


class WorkerClaim(BaseModel):
    """One durable, time-bounded attempt to execute a worker step."""

    claim_id: UUID
    step_id: UUID
    assignment_id: UUID
    task_id: UUID
    worker_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    checkpoint_revision: int = Field(ge=0)
    observation_id: UUID
    claimed_at: datetime
    lease_expires_at: datetime
    last_heartbeat_at: datetime
    status: WorkerClaimStatus = WorkerClaimStatus.ACTIVE

    @field_validator("worker_id")
    @classmethod
    def normalize_worker_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("worker_id must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_claim(self) -> "WorkerClaim":
        self.claimed_at = _as_utc(self.claimed_at)
        self.lease_expires_at = _as_utc(self.lease_expires_at)
        self.last_heartbeat_at = _as_utc(self.last_heartbeat_at)
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("worker lease must have a positive lifetime")
        if not (self.claimed_at <= self.last_heartbeat_at < self.lease_expires_at):
            raise ValueError("worker heartbeat must fall inside the lease")
        if self.claim_id != deterministic_worker_claim_id(
            self.step_id,
            self.attempt,
        ):
            raise ValueError("worker claim id is not deterministic")
        return self


class WorkerCheckpoint(BaseModel):
    """Append-only safe-resume state produced by one leased worker."""

    checkpoint_id: UUID
    step_id: UUID
    claim_id: UUID
    revision: int = Field(ge=1)
    state: dict[str, Any] = Field(default_factory=dict)
    output_refs: list[str] = Field(default_factory=list)
    created_at: datetime

    @field_validator("state")
    @classmethod
    def validate_json_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _require_json_object(value, field_name="checkpoint state")

    @field_validator("output_refs")
    @classmethod
    def normalize_output_refs(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("output_refs must not contain empty values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("output_refs must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "WorkerCheckpoint":
        self.created_at = _as_utc(self.created_at)
        if self.checkpoint_id != deterministic_worker_checkpoint_id(
            self.step_id,
            self.revision,
        ):
            raise ValueError("worker checkpoint id is not deterministic")
        return self


class WorkerResult(BaseModel):
    """Exactly one terminal result for one durable worker step."""

    result_id: UUID
    step_id: UUID
    claim_id: UUID
    checkpoint_revision: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1)
    output: dict[str, Any] = Field(default_factory=dict)
    output_refs: list[str] = Field(default_factory=list)
    completed_at: datetime

    @field_validator("output")
    @classmethod
    def validate_json_output(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _require_json_object(value, field_name="worker result output")

    @field_validator("idempotency_key")
    @classmethod
    def normalize_result_idempotency_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("idempotency_key must not be empty")
        return normalized

    @field_validator("output_refs")
    @classmethod
    def normalize_result_refs(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("output_refs must not contain empty values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("output_refs must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_result(self) -> "WorkerResult":
        self.completed_at = _as_utc(self.completed_at)
        if self.result_id != deterministic_worker_result_id(self.step_id):
            raise ValueError("worker result id is not deterministic")
        if self.idempotency_key != deterministic_worker_idempotency_key(
            self.step_id
        ):
            raise ValueError("worker result uses the wrong idempotency key")
        return self


class WorkerClaimResourceObservation(BaseModel):
    """Exact, immutable resource input and decision consumed by a claim."""

    observation_id: UUID
    step_id: UUID
    assignment_id: UUID
    captured_at: datetime
    evaluated_at: datetime
    valid_until: datetime
    claim_policy_version: str = Field(
        default=WORKER_CLAIM_POLICY_VERSION,
        min_length=1,
    )
    resource_safety_policy_version: str = Field(min_length=1)
    decision: WorkerClaimDecision
    reason: str = Field(min_length=1)
    active_claim_ids: list[UUID] = Field(default_factory=list)
    target_reservation_ids: list[UUID] = Field(default_factory=list)
    resource_snapshot: ResourceObservationSnapshot

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("claim observation reason must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_observation(self) -> "WorkerClaimResourceObservation":
        if self.claim_policy_version != WORKER_CLAIM_POLICY_VERSION:
            raise ValueError("worker claim policy version is unsupported")
        self.captured_at = _as_utc(self.captured_at)
        self.evaluated_at = _as_utc(self.evaluated_at)
        self.valid_until = _as_utc(self.valid_until)
        if self.valid_until <= self.captured_at:
            raise ValueError("claim observation must have a positive lifetime")
        if (
            self.resource_snapshot.captured_at != self.captured_at
            or self.resource_snapshot.valid_until != self.valid_until
            or self.resource_snapshot.safety_policy_version
            != self.resource_safety_policy_version
        ):
            raise ValueError("claim observation must bind its exact resource snapshot")
        if self.decision is WorkerClaimDecision.GRANTED and not (
            self.captured_at <= self.evaluated_at <= self.valid_until
        ):
            raise ValueError("a granted claim observation must be fresh")
        for field_name, values in (
            ("active_claim_ids", self.active_claim_ids),
            ("target_reservation_ids", self.target_reservation_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if self.observation_id != deterministic_worker_claim_observation_id(self):
            raise ValueError("worker claim observation id is not deterministic")
        return self

    def assert_fresh(self, *, at: datetime) -> None:
        evaluated_at = _as_utc(at)
        if evaluated_at < self.captured_at or evaluated_at > self.valid_until:
            raise ValueError("worker claim resource observation is stale")


class WorkerClaimEnvelope(BaseModel):
    """The complete bounded input handed to a freshly started worker."""

    step: WorkerStep
    claim: WorkerClaim
    checkpoint: WorkerCheckpoint | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> "WorkerClaimEnvelope":
        if (
            self.claim.step_id != self.step.step_id
            or self.claim.assignment_id != self.step.assignment_id
            or self.claim.task_id != self.step.task_id
        ):
            raise ValueError("worker claim does not match its step")
        if self.checkpoint is None:
            if self.claim.checkpoint_revision != 0:
                raise ValueError("claim references a missing checkpoint")
        elif (
            self.checkpoint.step_id != self.step.step_id
            or self.checkpoint.revision != self.claim.checkpoint_revision
        ):
            raise ValueError("claim does not reference the latest checkpoint")
        return self


class WorkerClaimAttempt(BaseModel):
    """Persisted guarded-claim decision and optional executable lease."""

    observation: WorkerClaimResourceObservation
    envelope: WorkerClaimEnvelope | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "WorkerClaimAttempt":
        granted = self.observation.decision is WorkerClaimDecision.GRANTED
        if granted != (self.envelope is not None):
            raise ValueError("only a granted observation may carry a worker lease")
        if self.envelope is not None and (
            self.envelope.step.step_id != self.observation.step_id
            or self.envelope.step.assignment_id != self.observation.assignment_id
            or self.envelope.claim.observation_id != self.observation.observation_id
        ):
            raise ValueError("claim envelope does not match its observation")
        return self


def deterministic_worker_step_id(assignment_id: UUID, step_key: str) -> UUID:
    normalized = step_key.strip()
    if not normalized:
        raise ValueError("step_key must not be empty")
    return uuid5(
        _WORKER_STEP_NAMESPACE,
        f"{assignment_id}:prometheist-worker-step:{normalized}",
    )


def deterministic_worker_idempotency_key(step_id: UUID) -> str:
    return f"prometheist:{WORKER_PROTOCOL_VERSION}:{step_id}"


def deterministic_worker_claim_id(step_id: UUID, attempt: int) -> UUID:
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return uuid5(step_id, f"prometheist-worker-claim:{attempt}")


def deterministic_worker_checkpoint_id(step_id: UUID, revision: int) -> UUID:
    if revision < 1:
        raise ValueError("checkpoint revision must be >= 1")
    return uuid5(step_id, f"prometheist-worker-checkpoint:{revision}")


def deterministic_worker_result_id(step_id: UUID) -> UUID:
    return uuid5(step_id, "prometheist-worker-result")


def deterministic_worker_claim_observation_id(
    observation: WorkerClaimResourceObservation,
) -> UUID:
    payload = observation.model_dump(mode="json", exclude={"observation_id"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return uuid5(_WORKER_OBSERVATION_NAMESPACE, canonical)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("worker protocol timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _require_json_object(
    value: dict[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        restored = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only JSON values") from exc
    if restored != value:
        raise ValueError(f"{field_name} must round-trip through JSON exactly")
    return value
