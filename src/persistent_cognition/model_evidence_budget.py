"""Fail-closed bounds for ephemeral model-facing evidence.

Canonical durable evidence is intentionally not size-limited here. These limits
apply only to disposable evidence views that are about to be supplied to an LLM.
The exact byte caps are safety/environment tunables under Constitution Article 30,
not constitutional constants.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from persistent_cognition.models import MemoryPacket


DEFAULT_MAX_EVIDENCE_ITEM_BYTES = 16384
DEFAULT_MAX_EVIDENCE_TOTAL_BYTES = 65536
DEFAULT_MAX_MODEL_INPUT_BYTES = 98304
_ITEM_ENV = "PCR_MAX_MODEL_EVIDENCE_ITEM_BYTES"
_TOTAL_ENV = "PCR_MAX_MODEL_EVIDENCE_TOTAL_BYTES"
_INPUT_ENV = "PCR_MAX_MODEL_INPUT_BYTES"


class ModelEvidenceBudgetExceeded(RuntimeError):
    """Model-facing evidence cannot be represented within the governed budget."""


class ModelEvidenceBudget(BaseModel):
    """Safety-tunable byte limits for one disposable model invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_item_bytes: int = Field(default=DEFAULT_MAX_EVIDENCE_ITEM_BYTES, gt=0)
    max_total_bytes: int = Field(default=DEFAULT_MAX_EVIDENCE_TOTAL_BYTES, gt=0)

    @model_validator(mode="after")
    def validate_ordering(self) -> "ModelEvidenceBudget":
        if self.max_item_bytes > self.max_total_bytes:
            raise ValueError("max_item_bytes must not exceed max_total_bytes")
        return self


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def configured_model_evidence_budget() -> ModelEvidenceBudget:
    """Read the current model-facing evidence safety tunables."""

    return ModelEvidenceBudget(
        max_item_bytes=_positive_env_int(
            _ITEM_ENV,
            DEFAULT_MAX_EVIDENCE_ITEM_BYTES,
        ),
        max_total_bytes=_positive_env_int(
            _TOTAL_ENV,
            DEFAULT_MAX_EVIDENCE_TOTAL_BYTES,
        ),
    )


def validate_model_input(request: dict[str, Any]) -> None:
    """Bound the complete rendered prompt, message roles and output schema.

    This is a deterministic byte bound, not a model-specific tokenizer count.
    It includes current input and instructions that evidence-only limits omit.
    """

    maximum = _positive_env_int(_INPUT_ENV, DEFAULT_MAX_MODEL_INPUT_BYTES)
    input_fields = {
        key: request[key]
        for key in ("prompt", "messages", "format", "input", "instructions", "text")
        if key in request
    }
    size = len(json.dumps(input_fields, ensure_ascii=False, default=str).encode("utf-8"))
    if size > maximum:
        raise ModelEvidenceBudgetExceeded(
            f"complete model input exceeds byte budget: bytes={size} max={maximum}"
        )


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def validate_memory_packet_content(
    packet: MemoryPacket | None,
    *,
    budget: ModelEvidenceBudget | None = None,
) -> None:
    """Reject an oversized packet before any formatter copies its full content."""

    if packet is None:
        return
    active_budget = budget or configured_model_evidence_budget()
    total = 0
    for index, item in enumerate(packet.items):
        item_bytes = _utf8_size(item.content)
        if item_bytes > active_budget.max_item_bytes:
            raise ModelEvidenceBudgetExceeded(
                "memory evidence item exceeds model-facing byte budget: "
                f"index={index} bytes={item_bytes} max={active_budget.max_item_bytes} "
                f"source_event_id={item.source_event_id}"
            )
        total += item_bytes
        if total > active_budget.max_total_bytes:
            raise ModelEvidenceBudgetExceeded(
                "memory evidence content exceeds aggregate model-facing byte budget: "
                f"bytes={total} max={active_budget.max_total_bytes}"
            )


def validate_capability_result_content(
    results: tuple[dict[str, Any], ...],
    *,
    budget: ModelEvidenceBudget | None = None,
    prior_evidence_bytes: int = 0,
) -> int:
    """Bound structured capability evidence using its deterministic JSON view."""

    active_budget = budget or configured_model_evidence_budget()
    total = prior_evidence_bytes
    for index, result in enumerate(results):
        rendered = json.dumps(
            result,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        result_bytes = _utf8_size(rendered)
        if result_bytes > active_budget.max_item_bytes:
            raise ModelEvidenceBudgetExceeded(
                "capability result exceeds model-facing byte budget: "
                f"index={index} bytes={result_bytes} max={active_budget.max_item_bytes}"
            )
        total += result_bytes
        if total > active_budget.max_total_bytes:
            raise ModelEvidenceBudgetExceeded(
                "combined evidence exceeds aggregate model-facing byte budget: "
                f"bytes={total} max={active_budget.max_total_bytes}"
            )
    return total


def memory_packet_content_bytes(packet: MemoryPacket | None) -> int:
    """Return UTF-8 content bytes after validation callers establish boundedness."""

    if packet is None:
        return 0
    return sum(_utf8_size(item.content) for item in packet.items)


def validate_rendered_evidence(
    parts: Iterable[str],
    *,
    budget: ModelEvidenceBudget | None = None,
) -> None:
    """Bound the final rendered evidence payload including metadata overhead."""

    active_budget = budget or configured_model_evidence_budget()
    total = sum(_utf8_size(part) for part in parts)
    if total > active_budget.max_total_bytes:
        raise ModelEvidenceBudgetExceeded(
            "rendered evidence exceeds aggregate model-facing byte budget: "
            f"bytes={total} max={active_budget.max_total_bytes}"
        )
