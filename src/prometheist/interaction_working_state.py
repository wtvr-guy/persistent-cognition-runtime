"""Minimal durable active-situation state for stateless interaction continuity.

Working state is deliberately simple: it records which canonical events are
currently active. It does not parse natural language into an ever-growing set
of phrase-specific referents, labels, entities, or conclusions. Fresh workers
rehydrate those canonical events through JIT Memory, and semantic interpretation
remains a stateless inference concern rather than hard-coded application logic.
"""
from __future__ import annotations

from uuid import UUID, uuid5

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from prometheist import event_store
from prometheist.models import EventType


WORKING_STATE_VERSION = "v0.7-working-state-v2"
SOURCE = "interaction_working_state"
MAX_ACTIVE_EVENT_IDS = 12


class InteractionWorkingState(BaseModel):
    """Bounded system-owned activation record for the current situation."""

    version: str = WORKING_STATE_VERSION
    state_id: UUID
    revision: int = Field(ge=1)
    conversation_ids: list[UUID] = Field(default_factory=list, max_length=16)
    active_event_ids: list[UUID] = Field(
        default_factory=list,
        max_length=MAX_ACTIVE_EVENT_IDS,
    )


def deterministic_working_state_id(conversation_id: UUID) -> UUID:
    return uuid5(conversation_id, "interaction-working-state")


def deterministic_working_state_event_id(interaction_id: UUID, activation_key: str) -> UUID:
    return uuid5(interaction_id, f"working-state-event:{activation_key}")


def load_working_state(
    conn: psycopg.Connection,
    conversation_id: UUID,
) -> InteractionWorkingState | None:
    """Load the latest active state bound to the current interaction stream."""

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT payload
            FROM events
            WHERE event_type = %s
              AND (
                    conversation_id = %s
                    OR payload -> 'state' -> 'conversation_ids' ? %s
              )
            ORDER BY global_seq DESC
            LIMIT 1
            """,
            (
                EventType.INTERACTION_WORKING_STATE.value,
                conversation_id,
                str(conversation_id),
            ),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return InteractionWorkingState.model_validate(row["payload"]["state"])


def activate_working_state(
    conn: psycopg.Connection,
    *,
    interaction_id: UUID,
    conversation_id: UUID,
    correlation_id: UUID,
    activated_event_ids: list[UUID],
    activation_key: str = "response",
) -> InteractionWorkingState:
    """Promote canonical events into bounded active state.

    Each activation phase has a deterministic event ID. A retry of the same
    phase therefore returns the already-persisted revision instead of advancing
    state again. Distinct phases (for example `memory` and `response`) remain
    append-only revisions within the same interaction.
    """

    event_id = deterministic_working_state_event_id(interaction_id, activation_key)
    existing = event_store.get_event_by_id(conn, event_id)
    if existing is not None:
        if existing.event_type is not EventType.INTERACTION_WORKING_STATE:
            raise RuntimeError("working-state activation id collides with another event type")
        return InteractionWorkingState.model_validate(existing.payload["state"])

    previous = load_working_state(conn, conversation_id)
    state_id = (
        previous.state_id
        if previous is not None
        else deterministic_working_state_id(conversation_id)
    )
    revision = 1 if previous is None else previous.revision + 1
    conversation_ids = _merge_uuid_lists(
        [conversation_id],
        previous.conversation_ids if previous is not None else [],
        16,
    )
    active_event_ids = _merge_uuid_lists(
        activated_event_ids,
        previous.active_event_ids if previous is not None else [],
        MAX_ACTIVE_EVENT_IDS,
    )

    state = InteractionWorkingState(
        state_id=state_id,
        revision=revision,
        conversation_ids=conversation_ids,
        active_event_ids=active_event_ids,
    )
    event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.INTERACTION_WORKING_STATE,
        source=SOURCE,
        payload={"activation_key": activation_key, "state": state.model_dump(mode="json")},
        event_id=event_id,
    )
    return state


def _merge_uuid_lists(first: list[UUID], second: list[UUID], limit: int) -> list[UUID]:
    merged: list[UUID] = []
    seen: set[UUID] = set()
    for value in [*first, *second]:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
        if len(merged) == limit:
            break
    return merged
