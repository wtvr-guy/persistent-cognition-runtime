"""Application-enforced response evidence and surface-form policy.

The semantic policy is inferred from the current user percept only. Retrieved
memory is never shown to that classifier, so persisted text cannot influence
which historical source roles become admissible for final answer synthesis.
"""
from __future__ import annotations

from enum import Enum
import re

from pydantic import BaseModel, ConfigDict, model_validator

from prometheist.models import EventType, MemoryPacket


RESPONSE_POLICY_VERSION = "response-source-authority-v6"

_EXPLICIT_PRIOR_ASSISTANT_REFERENCE = re.compile(
    r"\b(?:you|assistant|prometheist)\s+"
    r"(?:just\s+)?(?:said|answered|recommended|ruled\s+out|asked|mentioned)\b"
    r"|\bwhat\s+did\s+(?:you|the\s+assistant|prometheist)\b"
    r"|\bprevious\s+(?:answer|response|recommendation)\b"
    r"|\bthat\s+(?:was\s+)?ruled[- ]out\b",
    re.IGNORECASE,
)


class HistoricalEvidenceScope(str, Enum):
    """Historical source role allowed to establish the requested claim."""

    USER_AUTHORED = "USER_AUTHORED"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    EXTERNAL_TOOL = "EXTERNAL_TOOL"
    SYSTEM_RECORD = "SYSTEM_RECORD"
    DERIVED_INTERNAL = "DERIVED_INTERNAL"
    SELF_MODEL = "SELF_MODEL"
    MIXED_CONVERSATION = "MIXED_CONVERSATION"
    GENERAL_OR_CURRENT = "GENERAL_OR_CURRENT"


class ResponseSurfaceMode(str, Enum):
    """Whether final text may be generated freely or must be source-extractive."""

    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"
    EXACT_SOURCE_SUBSTRING = "EXACT_SOURCE_SUBSTRING"
    EXACT_SOURCE_COMPOSITION = "EXACT_SOURCE_COMPOSITION"


class ResponsePolicy(BaseModel):
    """Closed current-percept-only policy used before exposing history."""

    model_config = ConfigDict(extra="forbid")

    evidence_scope: HistoricalEvidenceScope
    surface_mode: ResponseSurfaceMode
    insufficient_literal: str | None = None

    @model_validator(mode="after")
    def validate_insufficient_literal(self) -> "ResponsePolicy":
        if self.insufficient_literal is not None and not self.insufficient_literal:
            raise ValueError("insufficient_literal must not be empty")
        return self


class CurrentFallbackSelection(BaseModel):
    """Verbatim unsupported-history fallback selected from the current percept."""

    model_config = ConfigDict(extra="forbid")

    verbatim_value: str | None = None

    @model_validator(mode="after")
    def validate_value(self) -> "CurrentFallbackSelection":
        if self.verbatim_value is not None and not self.verbatim_value:
            raise ValueError("verbatim_value must not be empty")
        return self


class ExactSourceSelection(BaseModel):
    """Constrained semantic selection of an exact admitted-source substring."""

    model_config = ConfigDict(extra="forbid")

    source_index: int
    verbatim_value: str

    @model_validator(mode="after")
    def validate_value(self) -> "ExactSourceSelection":
        if not self.verbatim_value:
            raise ValueError("verbatim_value must not be empty")
        return self


class ExactSourceComposition(BaseModel):
    """Ordered source-backed values joined only by current-authority formatting."""

    model_config = ConfigDict(extra="forbid")

    selections: list[ExactSourceSelection]
    separator: str

    @model_validator(mode="after")
    def validate_structure(self) -> "ExactSourceComposition":
        if not self.selections:
            raise ValueError("exact-source composition requires at least one selection")
        if any(character.isalnum() for character in self.separator):
            raise ValueError("exact-source composition separator must be formatting-only")
        return self


_MODEL_OUTPUT_TYPES = frozenset(
    {
        EventType.INTERACTION_RESPONSE,
        EventType.AGENT_RESPONSE,
        EventType.AGENT_RESULT,
    }
)
_EXTERNAL_TOOL_TYPES = frozenset({EventType.TOOL_RESULT, EventType.PERCEPT_OBSERVATION})
_SYSTEM_RECORD_TYPES = frozenset(
    {
        EventType.SYSTEM_EVENT,
        EventType.ERROR,
        EventType.INTERACTION_WORKING_STATE,
    }
)
_DERIVED_INTERNAL_TYPES = frozenset(
    {
        EventType.DERIVED_REPRESENTATION,
        EventType.AGENT_DECISION,
        EventType.RETRIEVAL_REQUEST,
        EventType.RETRIEVAL_RESULT,
        EventType.MEMORY_REQUEST,
        EventType.MEMORY_PACKET,
        EventType.AGENT_DELEGATION,
        EventType.TOOL_REQUEST,
        EventType.CAPABILITY_REQUEST,
        EventType.CAPABILITY_PACKET,
        EventType.CAPABILITY_RESULT,
    }
)
_SELF_MODEL_SUPPORT_TYPES = frozenset(
    {
        EventType.USER_PROMPT,
        EventType.TOOL_RESULT,
        EventType.PERCEPT_OBSERVATION,
        EventType.SYSTEM_EVENT,
    }
)

_GENERAL_OR_CURRENT_TYPES = frozenset(
    {
        EventType.USER_PROMPT,
        EventType.TOOL_RESULT,
        EventType.SYSTEM_EVENT,
    }
)


def allowed_event_types(scope: HistoricalEvidenceScope) -> frozenset[EventType]:
    """Map every semantic scope to a closed application-owned allowlist."""

    if scope is HistoricalEvidenceScope.USER_AUTHORED:
        return frozenset({EventType.USER_PROMPT})
    if scope is HistoricalEvidenceScope.MODEL_OUTPUT:
        return _MODEL_OUTPUT_TYPES
    if scope is HistoricalEvidenceScope.EXTERNAL_TOOL:
        return _EXTERNAL_TOOL_TYPES
    if scope is HistoricalEvidenceScope.SYSTEM_RECORD:
        return _SYSTEM_RECORD_TYPES
    if scope is HistoricalEvidenceScope.DERIVED_INTERNAL:
        return _DERIVED_INTERNAL_TYPES
    if scope is HistoricalEvidenceScope.SELF_MODEL:
        return _SELF_MODEL_SUPPORT_TYPES
    if scope is HistoricalEvidenceScope.MIXED_CONVERSATION:
        return frozenset({EventType.USER_PROMPT}) | _MODEL_OUTPUT_TYPES
    if scope is HistoricalEvidenceScope.GENERAL_OR_CURRENT:
        return _GENERAL_OR_CURRENT_TYPES
    raise ValueError(f"unsupported historical evidence scope: {scope.value}")


def source_types_for_scope(scope: HistoricalEvidenceScope) -> list[EventType]:
    """Return the event roles eligible for bounded retrieval under one scope."""

    return sorted(allowed_event_types(scope), key=lambda event_type: event_type.value)


def explicit_prior_assistant_reference(prompt: str) -> bool:
    """Recognize only unambiguous references to prior assistant output."""

    return bool(_EXPLICIT_PRIOR_ASSISTANT_REFERENCE.search(prompt))


def filter_memory_packet_for_scope(
    packet: MemoryPacket | None,
    scope: HistoricalEvidenceScope,
) -> MemoryPacket | None:
    """Physically remove event roles that cannot establish the current claim."""

    if packet is None:
        return None
    allowed = allowed_event_types(scope)
    items = [item.model_copy(deep=True) for item in packet.items if item.event_type in allowed]
    return packet.model_copy(update={"items": items, "supported": bool(items)}, deep=True)


def scope_requires_historical_support(scope: HistoricalEvidenceScope) -> bool:
    return scope is not HistoricalEvidenceScope.GENERAL_OR_CURRENT


def validate_current_literal(prompt: str, literal: str | None) -> str | None:
    """Accept a selected fallback only when it is verbatim current-user text."""

    if literal is None:
        return None
    if literal not in prompt:
        raise ValueError("fallback literal must be an exact substring of the current prompt")
    return literal


def validate_exact_source_selection(
    source_texts: tuple[str, ...],
    selection: ExactSourceSelection,
) -> str:
    """Return canonical source bytes after validating index and substring."""

    if selection.source_index not in range(len(source_texts)):
        raise ValueError("exact-source selection referenced an unknown candidate")
    source = source_texts[selection.source_index]
    if selection.verbatim_value not in source:
        raise ValueError("exact-source value is not a verbatim substring of admitted evidence")
    return selection.verbatim_value


def validate_exact_source_composition(
    prompt: str,
    source_texts: tuple[str, ...],
    composition: ExactSourceComposition,
) -> str:
    """Compose exact output from source bytes plus current-request formatting."""

    if composition.separator and composition.separator not in prompt:
        raise ValueError("exact-source composition separator is not current-prompt text")
    values = [
        validate_exact_source_selection(source_texts, selection)
        for selection in composition.selections
    ]
    return composition.separator.join(values)
