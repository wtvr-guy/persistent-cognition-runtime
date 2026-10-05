"""Deterministic epistemic-role labels for model-facing historical evidence.

Canonical events remain unchanged. This module classifies each persisted event
by what its source role can establish when a disposable worker reasons from
retrieved history. The classification is application policy, not a model-authored
truth judgment.
"""
from __future__ import annotations

from dataclasses import dataclass

from prometheist.models import EventType, MemoryPacket


@dataclass(frozen=True)
class EvidenceAuthority:
    """Application-owned statement of what one event role can establish."""

    authority_class: str
    may_establish: str
    cannot_establish: str


_DIRECT_USER = EvidenceAuthority(
    authority_class="DIRECT_USER_TESTIMONY",
    may_establish=(
        "what the user previously said, named, preferred, required, planned, "
        "reported, or instructed"
    ),
    cannot_establish="external-world truth beyond the user's own statement",
)

_MODEL_OUTPUT = EvidenceAuthority(
    authority_class="MODEL_OUTPUT_ONLY",
    may_establish="what Prometheist or another model previously emitted",
    cannot_establish=(
        "that the user said, preferred, required, planned, reported, instructed, "
        "or possesses any embedded fact merely because the model asserted it"
    ),
)

_EXTERNAL_TOOL = EvidenceAuthority(
    authority_class="EXTERNAL_TOOL_EVIDENCE",
    may_establish="what the named external tool returned at that time",
    cannot_establish="that the user personally stated the returned content",
)

_SYSTEM_RECORD = EvidenceAuthority(
    authority_class="SYSTEM_RECORD",
    may_establish="the recorded Prometheist system/runtime state or occurrence",
    cannot_establish="a user-authored fact absent supporting USER_PROMPT evidence",
)

_DERIVED_INTERNAL = EvidenceAuthority(
    authority_class="DERIVED_INTERNAL_EVIDENCE",
    may_establish="that Prometheist derived, selected, or packaged this internal result",
    cannot_establish=(
        "a user-authored fact unless its canonical support includes USER_PROMPT evidence"
    ),
)

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


def authority_for_event_type(event_type: EventType) -> EvidenceAuthority:
    """Return deterministic epistemic authority for one canonical event role."""

    if event_type is EventType.USER_PROMPT:
        return _DIRECT_USER
    if event_type in _MODEL_OUTPUT_TYPES:
        return _MODEL_OUTPUT
    if event_type in _EXTERNAL_TOOL_TYPES:
        return _EXTERNAL_TOOL
    if event_type in _SYSTEM_RECORD_TYPES:
        return _SYSTEM_RECORD
    if event_type in _DERIVED_INTERNAL_TYPES:
        return _DERIVED_INTERNAL
    raise ValueError(f"unclassified event type: {event_type.value}")


def _mask_literals(text: str, literal_to_placeholder: dict[str, str]) -> str:
    masked = text
    for literal, placeholder in literal_to_placeholder.items():
        masked = masked.replace(literal, placeholder)
    return masked


def format_authority_bound_memory_packet(
    packet: MemoryPacket | None,
    *,
    literal_to_placeholder: dict[str, str] | None = None,
) -> str:
    """Render bounded evidence with explicit source-role authority semantics."""

    if packet is None:
        return ""
    if not packet.items:
        return "\n\n[Evidence timeline: oldest to newest]\nsupported: false\nitems: []"

    literal_to_placeholder = literal_to_placeholder or {}
    ordered_items = sorted(
        packet.items,
        key=lambda item: (item.global_seq, item.conversation_seq, str(item.source_event_id)),
    )
    recent_conversation_id = ordered_items[-1].conversation_id
    direct_user_orders: list[str] = []
    blocks: list[str] = []
    for index, item in enumerate(ordered_items):
        authority = authority_for_event_type(item.event_type)
        if authority.authority_class == "DIRECT_USER_TESTIMONY":
            direct_user_orders.append(str(index))
        scope = (
            "recent_conversation"
            if item.conversation_id == recent_conversation_id
            else "historical_context"
        )
        content = _mask_literals(item.content, literal_to_placeholder)
        blocks.append(
            f"evidence_order: {index}\n"
            f"conversation_scope: {scope}\n"
            f"event_type: {item.event_type.value}\n"
            f"authority_class: {authority.authority_class}\n"
            f"may_establish: {authority.may_establish}\n"
            f"cannot_establish: {authority.cannot_establish}\n"
            f"content: {content}"
        )

    direct_user_inventory = ",".join(direct_user_orders) if direct_user_orders else "none"
    return (
        "\n\n[Evidence authority inventory]\n"
        f"direct_user_testimony_orders: {direct_user_inventory}\n"
        "MODEL_OUTPUT_ONLY is never a substitute for DIRECT_USER_TESTIMONY.\n"
        "[Evidence timeline: oldest to newest]\n"
        f"supported: {str(packet.supported).lower()}\n"
        + "\n\n".join(blocks)
    )
