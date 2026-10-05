"""PostgreSQL bindings for durable Increment G interactions."""
from __future__ import annotations

from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from prometheist.attention_store import DEFAULT_SCHEDULER_KEY
from prometheist.db import require_row
from prometheist.interaction_contracts import DurableInteraction
from prometheist.perception import Percept, SalienceAssessment


def save_interaction(
    conn: psycopg.Connection,
    interaction: DurableInteraction,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> DurableInteraction:
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO attention_interactions (
                    scheduler_key, interaction_id, protocol_version,
                    conversation_id, correlation_id, user_prompt_event_id,
                    before_global_seq, task_id, assignment_id, user_text,
                    percept_payload, salience_assessment_payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scheduler_key, interaction_id) DO UPDATE SET
                    percept_payload = COALESCE(
                        attention_interactions.percept_payload,
                        EXCLUDED.percept_payload
                    ),
                    salience_assessment_payload = COALESCE(
                        attention_interactions.salience_assessment_payload,
                        EXCLUDED.salience_assessment_payload
                    )
                """,
                (
                    scheduler_key,
                    interaction.interaction_id,
                    interaction.protocol_version,
                    interaction.conversation_id,
                    interaction.correlation_id,
                    interaction.user_prompt_event_id,
                    interaction.before_global_seq,
                    interaction.task_id,
                    interaction.assignment_id,
                    interaction.user_text,
                    (
                        Json(interaction.percept.model_dump(mode="json"))
                        if interaction.percept is not None
                        else None
                    ),
                    (
                        Json(interaction.salience_assessment.model_dump(mode="json"))
                        if interaction.salience_assessment is not None
                        else None
                    ),
                ),
            )
            cur.execute(
                """
                SELECT * FROM attention_interactions
                WHERE scheduler_key = %s AND interaction_id = %s
                """,
                (scheduler_key, interaction.interaction_id),
            )
            stored = _row_to_interaction(
                require_row(cur.fetchone(), context="stored durable interaction")
            )
            if stored != interaction:
                raise ValueError("conflicting durable interaction retry")
        conn.commit()
        return stored
    except Exception:
        conn.rollback()
        raise


def load_interaction(
    conn: psycopg.Connection,
    interaction_id: UUID,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> DurableInteraction:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM attention_interactions
            WHERE scheduler_key = %s AND interaction_id = %s
            """,
            (scheduler_key, interaction_id),
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(interaction_id)
    return _row_to_interaction(row)


def load_interaction_by_task(
    conn: psycopg.Connection,
    task_id: UUID,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> DurableInteraction:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM attention_interactions
            WHERE scheduler_key = %s AND task_id = %s
            """,
            (scheduler_key, task_id),
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(task_id)
    return _row_to_interaction(row)


def _row_to_interaction(row: dict) -> DurableInteraction:
    return DurableInteraction(
        protocol_version=row["protocol_version"],
        interaction_id=row["interaction_id"],
        conversation_id=row["conversation_id"],
        correlation_id=row["correlation_id"],
        user_prompt_event_id=row["user_prompt_event_id"],
        before_global_seq=int(row["before_global_seq"]),
        task_id=row["task_id"],
        assignment_id=row["assignment_id"],
        user_text=row["user_text"],
        percept=(
            Percept.model_validate(row["percept_payload"])
            if row.get("percept_payload") is not None
            else None
        ),
        salience_assessment=(
            SalienceAssessment.model_validate(row["salience_assessment_payload"])
            if row.get("salience_assessment_payload") is not None
            else None
        ),
    )
