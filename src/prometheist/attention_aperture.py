"""System-owned default memory exposure for every percept.

The attention aperture is deliberately not a capability and does not depend on
an LLM deciding whether unseen memory might matter. Every interaction receives
one bounded, provenance-bearing activation packet derived from the current
percept plus durable WorkingState. Valid active WorkingState is guaranteed
exposure; the aperture recall budget limits only additional baseline history.
Fixed deterministic retrieval then widens and deepens recall through its
bounded routes. Basic memory availability and every
retrieval-stage choice remain part of Prometheist's cognitive substrate.

The aperture uses JIT Memory's activation boundary rather than its stricter
evidence-admission boundary. An aperture item means "potentially relevant enough
to keep available now," not "proved sufficient support for a claim."
"""
from __future__ import annotations

from uuid import UUID

import psycopg

from prometheist import jit_memory
from prometheist.interaction_contracts import (
    deterministic_aperture_request_id,
    deterministic_interaction_event_id,
    deterministic_interaction_id,
)
from prometheist.interaction_working_state import (
    MAX_ACTIVE_EVENT_IDS,
    activate_working_state,
    load_working_state,
)
from prometheist.models import EventType, MemoryPacket


ATTENTION_APERTURE_VERSION = "v0.7-attention-aperture-v4"
# This is the budget for additional baseline recall beyond guaranteed active
# WorkingState, not a total MemoryPacket item limit.
DEFAULT_ATTENTION_APERTURE_LIMIT = 6
MAX_ATTENTION_APERTURE_ITEMS = MAX_ACTIVE_EVENT_IDS + DEFAULT_ATTENTION_APERTURE_LIMIT


def _working_state_activation_order(
    *,
    prompt_event_id: UUID,
    packet: MemoryPacket,
    prior_active_event_ids: list[UUID],
) -> list[UUID]:
    """Promote current and newly recalled evidence ahead of stale focus.

    The MemoryPacket remains unchanged. This ordering only determines which
    canonical event identifiers survive WorkingState's bounded activation.
    """

    prior_active = set(prior_active_event_ids)
    newly_recalled = [
        item.source_event_id
        for item in packet.items
        if item.source_event_id not in prior_active
    ]
    recalled_prior_active = [
        item.source_event_id
        for item in packet.items
        if item.source_event_id in prior_active
    ]
    return [
        prompt_event_id,
        *newly_recalled,
        *recalled_prior_active,
        *prior_active_event_ids,
    ]


def open_attention_aperture(
    conn: psycopg.Connection,
    *,
    conversation_id: UUID,
    correlation_id: UUID,
    requester_task_id: UUID,
    user_text: str,
    before_global_seq: int,
    source_types: list[EventType] | None = None,
) -> MemoryPacket:
    """Return the bounded default activation packet for one current percept.

    Retrieval policy is application-owned and deterministic for the same event
    ledger, WorkingState, current percept, and kernel version. No model-written
    query/entity/capability text is accepted at this boundary. The maximum
    default exposure is bounded WorkingState plus the baseline recall budget.
    """

    interaction_id = deterministic_interaction_id(conversation_id, correlation_id)
    working_state = load_working_state(conn, conversation_id)
    active_event_ids = (
        list(working_state.active_event_ids) if working_state is not None else []
    )
    need = jit_memory.build_memory_need(
        user_text,
        active_event_ids=active_event_ids,
        include_persisted_history=True,
        conversation_id=None,
        limit=DEFAULT_ATTENTION_APERTURE_LIMIT,
        source_types=source_types,
    )
    packet = jit_memory.request_attention_activation(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        requesting_component=(
            f"attention-aperture:{ATTENTION_APERTURE_VERSION}/task:{requester_task_id}"
        ),
        need=need,
        before_global_seq=before_global_seq,
        memory_request_id=deterministic_aperture_request_id(interaction_id),
    )

    prompt_event_id = deterministic_interaction_event_id(interaction_id, "user-prompt")
    activate_working_state(
        conn,
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        activated_event_ids=_working_state_activation_order(
            prompt_event_id=prompt_event_id,
            packet=packet,
            prior_active_event_ids=active_event_ids,
        ),
        activation_key="aperture",
    )
    return packet
