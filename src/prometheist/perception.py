"""Deterministic percept normalization and salience assessment for v0.8."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prometheist.percept_context import PerceptContext, aware

PERCEPTION_POLICY_VERSION = "v0.8-perception-v2"
SALIENCE_POLICY_VERSION = "v0.8-salience-v2"
_INPUT_BUFFER_SEGMENT_CHARS = 160
_INPUT_BUFFER_MAX_SEGMENTS = 4
_PERCEPT_NAMESPACE = UUID("d0b7d62c-47d4-5a0f-99d2-0cc58d7357d3")


class PerceptKind(str, Enum):
    USER_INTERACTION = "USER_INTERACTION"
    SCHEDULED_EVENT = "SCHEDULED_EVENT"
    ANOMALY_ALERT = "ANOMALY_ALERT"
    EXTERNAL_OBSERVATION = "EXTERNAL_OBSERVATION"
    SYSTEM_OBSERVATION = "SYSTEM_OBSERVATION"
    ACTION_OUTCOME = "ACTION_OUTCOME"


class PerceptModality(str, Enum):
    TEXT = "TEXT"
    STRUCTURED = "STRUCTURED"
    METRIC = "METRIC"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    FILE = "FILE"
    DOCUMENT = "DOCUMENT"
    EVENT_STREAM = "EVENT_STREAM"


class SalienceDisposition(str, Enum):
    REFLEX = "REFLEX"
    ORIENT = "ORIENT"
    DELIBERATE = "DELIBERATE"
    IGNORE = "IGNORE"


class PerceptSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    kind: PerceptKind
    modality: PerceptModality
    interface: str | None = None

    @field_validator("source_id")
    @classmethod
    def normalize_source_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("source_id must not be blank")
        return normalized

    @field_validator("interface")
    @classmethod
    def normalize_interface(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class PerceptBuffer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_char_limit: int = Field(default=_INPUT_BUFFER_SEGMENT_CHARS, ge=1)
    max_segments: int = Field(default=_INPUT_BUFFER_MAX_SEGMENTS, ge=1)
    retained_segment_indices: tuple[int, ...] = Field(default_factory=tuple)
    segments: tuple[str, ...] = Field(default_factory=tuple)
    total_segments: int = Field(ge=0)
    total_characters: int = Field(ge=0)
    truncated: bool = False

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value for value in values if value)
        if len(normalized) != len(values):
            raise ValueError("buffer segments must not be blank")
        return normalized


class PerceptFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    normalized_characters: int = Field(ge=0)
    token_count: int = Field(ge=0)
    line_count: int = Field(ge=0)
    contains_question: bool = False
    contains_url: bool = False
    contains_code_block: bool = False
    contains_structured_payload: bool = False
    uppercase_ratio: float = Field(ge=0.0, le=1.0)


class Percept(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_version: str = PERCEPTION_POLICY_VERSION
    percept_id: UUID
    source: PerceptSource
    observed_at: datetime
    source_event_id: UUID | None = None
    conversation_id: UUID | None = None
    correlation_id: UUID
    raw_value_sha256: str = Field(min_length=64, max_length=64)
    normalized_text: str = Field(min_length=1)
    input_buffer: PerceptBuffer
    features: PerceptFeatures
    response_required: bool = False
    context: PerceptContext = Field(default_factory=PerceptContext)

    _aware = field_validator("observed_at")(aware)

    @field_validator("normalized_text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("normalized_text must not be blank")
        return normalized


class AdvisorySemanticClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    source_model: str = Field(min_length=1)
    authoritative: Literal[False] = False

    @field_validator("label", "source_model")
    @classmethod
    def normalize_text_field(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("classification fields must not be blank")
        return normalized


class SalienceSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    anomaly_score: int = Field(ge=0, le=3)
    threat_score: int = Field(ge=0, le=3)
    opportunity_score: int = Field(ge=0, le=3)
    goal_relevance_score: int = Field(ge=0, le=3)
    novelty_score: int = Field(ge=0, le=3)
    uncertainty_score: int = Field(ge=0, le=3)
    system_integrity_score: int = Field(ge=0, le=3)
    prediction_error_score: int = Field(default=0, ge=0, le=3)
    task_relevance_score: int = Field(default=0, ge=0, le=3)


class SalienceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_version: str = SALIENCE_POLICY_VERSION
    percept_id: UUID
    disposition: SalienceDisposition
    signals: SalienceSignals
    trigger_terms: tuple[str, ...] = Field(default_factory=tuple)
    preauthorized_reflexes: tuple[str, ...] = Field(default_factory=tuple)
    advisory_classification: AdvisorySemanticClassification | None = None
    advisory_only: Literal[True] = True


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _is_json_compatible(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_compatible(item) for key, item in value.items())
    return False


def _observation_text(value: Any, modality: PerceptModality) -> str:
    if modality in {PerceptModality.IMAGE, PerceptModality.AUDIO, PerceptModality.VIDEO,
                     PerceptModality.FILE, PerceptModality.DOCUMENT}:
        from prometheist.percept_adapters import MediaReference, verify_media
        reference = MediaReference.model_validate(value)
        verify_media(reference)
        return _stable_json(reference.model_dump(mode="json"))
    if modality is PerceptModality.EVENT_STREAM:
        from prometheist.percept_adapters import MAX_EVENT_STREAM_RECORDS
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_EVENT_STREAM_RECORDS:
            raise ValueError("event streams require a bounded nonempty list of records")
        if not all(isinstance(item, dict) and _is_json_compatible(item) for item in value):
            raise ValueError("event stream records must be JSON objects")
        return _stable_json(value)
    if modality is PerceptModality.TEXT:
        if not isinstance(value, str):
            raise ValueError("text percepts require a string observation")
        return value
    if modality is PerceptModality.METRIC:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("metric percepts require a scalar numeric observation")
        return _stable_json(value)
    if isinstance(value, str):
        raise ValueError("structured percepts require a non-text structured payload")
    if not _is_json_compatible(value):
        raise ValueError("structured percepts require a JSON-compatible payload")
    return _stable_json(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_buffer(text: str) -> PerceptBuffer:
    total_segments = max(1, math.ceil(len(text) / _INPUT_BUFFER_SEGMENT_CHARS))
    if total_segments <= _INPUT_BUFFER_MAX_SEGMENTS:
        retained_indices = tuple(range(total_segments))
    else:
        head_count = _INPUT_BUFFER_MAX_SEGMENTS // 2
        tail_count = _INPUT_BUFFER_MAX_SEGMENTS - head_count
        retained_indices = tuple(range(head_count)) + tuple(
            range(total_segments - tail_count, total_segments)
        )
    return PerceptBuffer(
        retained_segment_indices=retained_indices,
        segments=tuple(text[index * _INPUT_BUFFER_SEGMENT_CHARS :
                            (index + 1) * _INPUT_BUFFER_SEGMENT_CHARS]
                       for index in retained_indices),
        total_segments=total_segments,
        total_characters=len(text),
        truncated=total_segments > _INPUT_BUFFER_MAX_SEGMENTS,
    )


def _uppercase_ratio(text: str) -> float:
    letter_count = sum(char.isalpha() for char in text)
    if not letter_count:
        return 0.0
    return sum(char.isupper() for char in text) / letter_count


def _feature_flags(text: str, modality: PerceptModality) -> PerceptFeatures:
    stripped = text.strip()
    return PerceptFeatures(
        normalized_characters=len(text),
        token_count=sum(1 for _ in re.finditer(r"\S+", text)),
        line_count=(text.count("\n") + 1) if text else 0,
        contains_question="?" in text,
        contains_url=("://" in text) or ("www." in text.casefold()),
        contains_code_block="```" in text,
        contains_structured_payload=modality is PerceptModality.STRUCTURED
        or stripped.startswith("{")
        or stripped.startswith("["),
        uppercase_ratio=_uppercase_ratio(text),
    )


def deterministic_percept_id(
    source_event_id: UUID | None,
    correlation_id: UUID,
    source: PerceptSource,
    *,
    observed_at: datetime,
    normalized_text: str,
) -> UUID:
    seed = (
        str(source_event_id)
        if source_event_id is not None
        else (
            f"{correlation_id}:{source.source_id}:{source.kind.value}:{source.modality.value}:"
            f"{source.interface or '-'}:{observed_at.isoformat()}:"
            f"{_sha256_text(normalized_text)}"
        )
    )
    return uuid5(_PERCEPT_NAMESPACE, seed)


def normalize_percept(
    *,
    source: PerceptSource,
    observation: Any,
    observed_at: datetime,
    correlation_id: UUID,
    source_event_id: UUID | None = None,
    conversation_id: UUID | None = None,
    response_required: bool = False,
    context: PerceptContext | None = None,
) -> Percept:
    raw_text = _observation_text(observation, source.modality)
    if not raw_text.strip():
        raise ValueError("percept observation must not be empty")
    normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return Percept(
        percept_id=deterministic_percept_id(
            source_event_id,
            correlation_id,
            source,
            observed_at=observed_at,
            normalized_text=normalized_text,
        ),
        source=source,
        observed_at=observed_at,
        source_event_id=source_event_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        raw_value_sha256=_sha256_text(raw_text),
        normalized_text=normalized_text,
        input_buffer=_build_buffer(normalized_text),
        features=_feature_flags(normalized_text, source.modality),
        response_required=response_required or source.kind is PerceptKind.USER_INTERACTION,
        context=context or PerceptContext(),
    )


def normalize_user_interaction_percept(
    *,
    user_text: str,
    observed_at: datetime,
    correlation_id: UUID,
    source_event_id: UUID,
    conversation_id: UUID,
) -> Percept:
    return normalize_percept(
        source=PerceptSource(
            source_id="user",
            kind=PerceptKind.USER_INTERACTION,
            modality=PerceptModality.TEXT,
            interface="chat",
        ),
        observation=user_text,
        observed_at=observed_at,
        source_event_id=source_event_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        response_required=True,
    )


def normalize_scheduled_percept(
    *,
    source_id: str,
    observation: Any,
    observed_at: datetime,
    correlation_id: UUID,
    source_event_id: UUID | None = None,
) -> Percept:
    return normalize_percept(
        source=PerceptSource(
            source_id=source_id,
            kind=PerceptKind.SCHEDULED_EVENT,
            modality=PerceptModality.STRUCTURED,
            interface="scheduler",
        ),
        observation=observation,
        observed_at=observed_at,
        source_event_id=source_event_id,
        correlation_id=correlation_id,
        response_required=False,
    )


def normalize_anomaly_percept(
    *,
    source_id: str,
    observation: Any,
    observed_at: datetime,
    correlation_id: UUID,
    source_event_id: UUID | None = None,
) -> Percept:
    return normalize_percept(
        source=PerceptSource(
            source_id=source_id,
            kind=PerceptKind.ANOMALY_ALERT,
            modality=PerceptModality.STRUCTURED,
            interface="anomaly-detector",
        ),
        observation=observation,
        observed_at=observed_at,
        source_event_id=source_event_id,
        correlation_id=correlation_id,
        response_required=False,
    )


def evaluate_salience(
    percept: Percept,
    *,
    advisory_classification: AdvisorySemanticClassification | None = None,
    prediction_error: float = 0.0,
    novelty: bool = False,
) -> SalienceAssessment:
    """Rank explicit observations, goals, and prediction error; never lexical cues.

    Prediction error is already normalized to a declared expectation scale.
    Scores are advisory ordinal bins; no score authorizes a reflex or model call.
    """
    if not math.isfinite(prediction_error) or prediction_error < 0:
        raise ValueError("prediction_error must be finite and non-negative")
    context = percept.context
    error_score = (3 if prediction_error >= 1.0 else 2 if prediction_error >= 0.25
                   else 1 if prediction_error > 0.0 else 0)
    signals = SalienceSignals(
        anomaly_score=error_score,
        prediction_error_score=error_score,
        threat_score=context.threat,
        opportunity_score=context.opportunity,
        goal_relevance_score=3 if context.active_goal_refs else 2 if percept.response_required else 0,
        novelty_score=int(novelty),
        uncertainty_score=context.uncertainty,
        system_integrity_score=context.integrity,
        task_relevance_score=3 if context.task_refs else 0,
    )
    if max(error_score, context.threat, context.integrity, context.uncertainty) >= 2:
        disposition = SalienceDisposition.ORIENT
    elif percept.response_required or any((context.active_goal_refs, context.task_refs,
                                          context.opportunity, novelty, error_score)):
        disposition = SalienceDisposition.DELIBERATE
    else:
        disposition = SalienceDisposition.IGNORE
    return SalienceAssessment(
        percept_id=percept.percept_id, disposition=disposition, signals=signals,
        advisory_classification=advisory_classification,
    )


def format_salience_context(assessment: SalienceAssessment) -> str:
    signals = assessment.signals
    advisory = "none"
    if assessment.advisory_classification is not None:
        advisory = (
            f"{assessment.advisory_classification.label} "
            f"(confidence={assessment.advisory_classification.confidence:.2f}, "
            f"authoritative=false)"
        )
    trigger_terms = ", ".join(assessment.trigger_terms) or "none"
    reflexes = ", ".join(assessment.preauthorized_reflexes) or "none"
    return (
        f"disposition={assessment.disposition.value}\n"
        f"anomaly_score={signals.anomaly_score}\n"
        f"threat_score={signals.threat_score}\n"
        f"opportunity_score={signals.opportunity_score}\n"
        f"goal_relevance_score={signals.goal_relevance_score}\n"
        f"novelty_score={signals.novelty_score}\n"
        f"uncertainty_score={signals.uncertainty_score}\n"
        f"system_integrity_score={signals.system_integrity_score}\n"
        f"trigger_terms={trigger_terms}\n"
        f"preauthorized_reflexes={reflexes}\n"
        f"advisory_semantic_classification={advisory}"
    )
