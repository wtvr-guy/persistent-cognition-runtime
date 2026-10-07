"""Deterministic display and field lookup over admitted canonical evidence.

No model interprets these sources. A trusted caller may bind a known structured
field to a source reference; otherwise code displays the complete source records
as quotations. Display does not declare one conflicting statement to be truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from persistent_cognition.epistemic_authority import authority_for_event_type
from persistent_cognition.models import MemoryPacket


EVIDENCE_RENDERER_VERSION = "canonical-evidence-response/v1"

_SOURCE_LABELS = {
    "DIRECT_USER_TESTIMONY": "Saved user statement",
    "MODEL_OUTPUT_ONLY": "Previous assistant output",
    "EXTERNAL_TOOL_EVIDENCE": "Tool result",
    "SYSTEM_RECORD": "System record",
    "DERIVED_INTERNAL_EVIDENCE": "Internal record",
    "CURRENT_USER_MESSAGE": "Current message",
}


class SourceValueBinding(BaseModel):
    """Application-supplied reference and typed field path, never model output.

    Source refs are event:<UUID>, work-result:<plan position>, or current-message.
    A path addresses dictionary keys or list indices without parsing English,
    inferring a fact, selecting text offsets, or evaluating attribute expressions.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_ref: str = Field(min_length=1)
    field_path: tuple[str | int, ...] = Field(min_length=1)


@dataclass(frozen=True)
class ResponseEvidenceSource:
    source_ref: str
    authority_class: str
    content: str

    def receipt(self) -> dict[str, Any]:
        return {"source_ref": self.source_ref, "authority_class": self.authority_class,
                "content": self.content}


def response_evidence_sources(
    packet: MemoryPacket | None,
    work_results: tuple[dict[str, Any], ...],
    *,
    current_percept: str | None = None,
) -> tuple[ResponseEvidenceSource, ...]:
    """Retain every admitted record with stable ordering and source identity.

    The caller owns role/cutoff and byte-budget admission. A JSON record is only
    traversed when an explicit binding is supplied; its shape alone never makes
    a field the answer to a natural-language question.
    """

    sources: dict[str, ResponseEvidenceSource] = {}

    def add(source: ResponseEvidenceSource) -> None:
        prior = sources.get(source.source_ref)
        if prior is not None and prior != source:
            raise ValueError("conflicting canonical records for one response source")
        sources[source.source_ref] = source

    if packet is not None:
        for item in sorted(packet.items, key=lambda item: (
            item.global_seq, item.conversation_seq, str(item.source_event_id),
        )):
            add(ResponseEvidenceSource(
                f"event:{item.source_event_id}",
                authority_for_event_type(item.event_type).authority_class, item.content,
            ))
    for index, result in enumerate(work_results):
        # plan_position survives filtering/reordering of admitted results. Direct
        # callers without a plan use the supplied tuple's application-owned order.
        position = result.get("plan_position", index)
        if type(position) is not int or position < 0:
            raise ValueError("work result has an invalid plan position")
        add(ResponseEvidenceSource(
            f"work-result:{position}", "EXTERNAL_TOOL_EVIDENCE",
            json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False),
        ))
    if current_percept is not None:
        add(ResponseEvidenceSource("current-message", "CURRENT_USER_MESSAGE", current_percept))
    return tuple(sources.values())


def resolve_bound_values(
    sources: tuple[ResponseEvidenceSource, ...], bindings: tuple[SourceValueBinding, ...],
) -> tuple[str, ...]:
    """Read explicitly bound scalar fields; never guess or fall back to a model."""

    by_ref = {source.source_ref: source for source in sources}
    values = []
    for binding in bindings:
        if binding.source_ref not in by_ref:
            raise ValueError("field binding references an unadmitted source")
        source = by_ref[binding.source_ref]
        try:
            data = json.loads(source.content, object_pairs_hook=_unique_fields)
        except (TypeError, ValueError) as exc:
            raise ValueError("field binding requires an unambiguous structured source record") from exc
        for component in binding.field_path:
            if isinstance(data, dict) and type(component) is str and component in data:
                data = data[component]
            elif isinstance(data, list) and type(component) is int and 0 <= component < len(data):
                data = data[component]
            else:
                raise ValueError("field binding references an absent or invalid field")
        if isinstance(data, (dict, list)):
            raise ValueError("field binding must resolve to a scalar value")
        values.append(data if isinstance(data, str) else json.dumps(
            data, ensure_ascii=False, allow_nan=False,
        ))
    return tuple(values)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate fields in a structured response source")
        fields[key] = value
    return fields


def render_source_evidence(sources: tuple[ResponseEvidenceSource, ...]) -> str:
    """Display exact quoted records without assigning semantic answerability."""

    if not sources:
        raise ValueError("evidence display requires admitted source records")
    blocks = ["Retrieved evidence (quoted source records):"]
    for source in sources:
        blocks.append(
            f"{_SOURCE_LABELS[source.authority_class]}:\n"
            + json.dumps(source.content, ensure_ascii=False)
        )
    return "\n\n".join(blocks)


def render_value_response(values: tuple[str, ...]) -> str:
    if not values:
        raise ValueError("a factual response requires bound source values")
    if len(values) == 1:
        sentence = f"The requested value is {values[0]}"
    else:
        sentence = "The requested values are " + "; ".join(values)
    return sentence if sentence.endswith((".", "?", "!")) else sentence + "."
