"""Single-owner PostgreSQL advisory locks over one scheduler namespace.

The interactive CLI chat, the headless job process, and the always-on percept
service all mutate replaceable scheduler/worker execution state for a given
``scheduler_key``. They share this session-scoped advisory lock so that at most
one owner can drain and reset a scheduler namespace at a time. The lock is held
for the life of the owning database session, so a crashed or exited owner
releases it automatically and a restart can re-acquire it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg

LOCK_NAMESPACE = "persistent_cognition.scheduler"


def scheduler_lock_name(scheduler_key: str) -> str:
    """Return the shared advisory-lock name for one scheduler namespace."""

    return f"{LOCK_NAMESPACE}:{scheduler_key}"


@contextmanager
def scheduler_ownership(conn: psycopg.Connection, scheduler_key: str) -> Iterator[None]:
    """Acquire exclusive ownership of ``scheduler_key`` for this session.

    Fails closed if another live session already owns the namespace, rather than
    silently sharing replaceable execution state.
    """

    name = scheduler_lock_name(scheduler_key)
    acquired = conn.execute(
        "SELECT pg_try_advisory_lock(hashtext(%s))", (name,)
    ).fetchone()[0]
    conn.commit()
    if not acquired:
        raise RuntimeError(
            f"Another owner already holds the {scheduler_key!r} scheduler lock"
        )
    try:
        yield
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (name,))
        conn.commit()
