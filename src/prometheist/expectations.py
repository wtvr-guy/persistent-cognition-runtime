"""Typed expectations and explicit comparisons, independent of language models."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TypeGuard
from uuid import UUID, uuid5

from pydantic import Field, field_validator, model_validator

from prometheist.percept_context import FrozenRecord, Observation, Reference, Scalar, aware


class Expectation(FrozenRecord):
    expectation_id: UUID
    subject: Reference
    property: Reference
    expected_value: Scalar | None = None
    expected_range: tuple[float, float] | None = None
    unit: Reference | None = None
    normalization_scale: float = Field(default=1.0, gt=0.0)
    source: Reference
    provenance: tuple[UUID, ...] = Field(min_length=1, max_length=16)
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: datetime
    valid_until: datetime
    supersedes: UUID | None = None

    _aware = field_validator("valid_from", "valid_until")(aware)

    @model_validator(mode="after")
    def valid_contract(self) -> "Expectation":
        if (self.expected_value is None) == (self.expected_range is None):
            raise ValueError("provide exactly one expected value or range")
        if self.expected_range and self.expected_range[0] > self.expected_range[1]:
            raise ValueError("expected range must be ascending")
        if self.valid_until <= self.valid_from:
            raise ValueError("expectation validity interval must be positive")
        return self


class SemanticDelta(str, Enum):
    MATCH = "MATCH"
    CONTRADICTION = "CONTRADICTION"
    OUTSIDE_RANGE = "OUTSIDE_RANGE"
    INCOMPARABLE = "INCOMPARABLE"
    EXPIRED = "EXPIRED"
    NOT_YET_VALID = "NOT_YET_VALID"


class PredictionError(FrozenRecord):
    prediction_error_id: UUID
    expectation_id: UUID
    percept_id: UUID
    subject: Reference
    property: Reference
    magnitude: float | None = Field(default=None, ge=0.0)
    direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    semantic_delta: SemanticDelta


def _is_numeric(value: object) -> TypeGuard[int | float]:
    """Narrow to a real number; ``bool`` is excluded although it subclasses ``int``."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_expectation(
    expectation: Expectation, observation: Observation, *, percept_id: UUID, observed_at: datetime,
) -> PredictionError:
    aware(observed_at)
    magnitude = None
    direction = "unknown"
    delta = SemanticDelta.INCOMPARABLE
    if observed_at >= expectation.valid_until:
        delta = SemanticDelta.EXPIRED
    elif observed_at < expectation.valid_from:
        delta = SemanticDelta.NOT_YET_VALID
    elif (expectation.subject, expectation.property, expectation.unit) == (
        observation.subject, observation.property, observation.unit,
    ):
        value = observation.value
        if expectation.expected_range is not None and _is_numeric(value):
            low, high = expectation.expected_range
            distance = value - high if value > high else value - low if value < low else 0.0
            magnitude = abs(distance) / expectation.normalization_scale
            direction = "above" if distance > 0 else "below" if distance < 0 else "equal"
            delta = SemanticDelta.OUTSIDE_RANGE if distance else SemanticDelta.MATCH
        elif expectation.expected_range is None:
            expected = expectation.expected_value
            if _is_numeric(value) and _is_numeric(expected):
                distance = value - expected
                magnitude = abs(distance) / expectation.normalization_scale
                direction = "above" if distance > 0 else "below" if distance < 0 else "equal"
            elif type(value) is type(expected):
                magnitude = 0.0 if value == expected else 1.0
                direction = "equal" if value == expected else "changed"
            if magnitude is not None:
                delta = SemanticDelta.CONTRADICTION if magnitude else SemanticDelta.MATCH
    return PredictionError(
        prediction_error_id=uuid5(expectation.expectation_id, str(percept_id)),
        expectation_id=expectation.expectation_id, percept_id=percept_id,
        subject=expectation.subject, property=expectation.property,
        magnitude=magnitude, direction=direction,
        confidence=min(expectation.confidence, observation.confidence), semantic_delta=delta,
    )
