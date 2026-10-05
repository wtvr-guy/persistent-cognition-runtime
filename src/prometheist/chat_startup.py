"""Interactive CLI startup policy for a clean execution plane.

Canonical conversations/events, derived memory, configured execution resources,
and the independent JSON artifact journal are intentionally outside this reset.
The interactive CLI is allowed to discard only the default scheduler namespace's
replaceable execution machinery before accepting a new prompt.
"""
from __future__ import annotations

import psycopg
from psycopg import sql

from prometheist.attention_store import DEFAULT_SCHEDULER_KEY


_RESET_TABLES_IN_DELETE_ORDER = (
    "attention_interactions",
    "attention_worker_results",
    "attention_worker_checkpoints",
    "attention_worker_claims",
    "attention_worker_claim_observations",
    "attention_worker_steps",
    "attention_preemption_events",
    "attention_resource_reservations",
    "attention_scheduler_state",
    "attention_scheduling_epochs",
    "attention_assignments",
    "attention_resource_observations",
    "attention_task_transitions",
    "attention_tasks",
)


def reset_chat_execution_state(
    conn: psycopg.Connection,
    *,
    scheduler_key: str = DEFAULT_SCHEDULER_KEY,
) -> dict[str, int]:
    """Atomically clear one scheduler namespace's replaceable execution state.

    This is the startup contract for ``prometheist chat``.  It prevents a prior
    Ctrl+C, killed parent process, crashed worker, or stale reservation from
    blocking a new interactive session.  Canonical memory/history and artifact
    files are never touched.  The global task-created sequence is deliberately
    not rewound, preserving monotonic intake identity across CLI sessions.
    """

    counts: dict[str, int] = {}
    try:
        with conn.cursor() as cur:
            for table in _RESET_TABLES_IN_DELETE_ORDER:
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE scheduler_key = %s").format(
                        sql.Identifier(table)
                    ),
                    (scheduler_key,),
                )
                counts[table] = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return counts
