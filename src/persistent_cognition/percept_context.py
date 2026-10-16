"""Bounded adapter assertions; data and references never grant execution rights."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

MAX_CONTEXT_ITEMS = 16
MAX_REFERENCE_CHARS = 256
MAX_VALUE_CHARS = 1024
Scalar = StrictBool | StrictInt | StrictFloat | Annotated[StrictStr, Field(max_length=MAX_VALUE_CHARS)]
Reference = Annotated[str, Field(min_length=1, max_length=MAX_REFERENCE_CHARS)]


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value


class Observation(FrozenRecord):
    subject: Reference
    property: Reference
    value: Scalar
    unit: Reference | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PerceptContext(FrozenRecord):
    entity_refs: tuple[Reference, ...] = Field(default=(), max_length=MAX_CONTEXT_ITEMS)
    active_goal_refs: tuple[Reference, ...] = Field(default=(), max_length=MAX_CONTEXT_ITEMS)
    task_refs: tuple[UUID, ...] = Field(default=(), max_length=MAX_CONTEXT_ITEMS)
    causal_percept_refs: tuple[UUID, ...] = Field(default=(), max_length=MAX_CONTEXT_ITEMS)
    expectation_refs: tuple[UUID, ...] = Field(default=(), max_length=MAX_CONTEXT_ITEMS)
    observations: tuple[Observation, ...] = Field(default=(), max_length=MAX_CONTEXT_ITEMS)
    # These are adapter assertions, recorded with source provenance. Reflexes
    # independently verify live facts; no score authorizes an effect.
    threat: int = Field(default=0, ge=0, le=3)
    opportunity: int = Field(default=0, ge=0, le=3)
    integrity: int = Field(default=0, ge=0, le=3)
    uncertainty: int = Field(default=0, ge=0, le=3)

    @field_validator("observations")
    @classmethod
    def unique_observations(cls, values: tuple[Observation, ...]) -> tuple[Observation, ...]:
        if len({(value.subject, value.property) for value in values}) != len(values):
            raise ValueError("one observed value per subject/property per percept")
        return values
