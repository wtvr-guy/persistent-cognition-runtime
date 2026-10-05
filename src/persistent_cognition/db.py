"""Postgres connection helper."""
from __future__ import annotations

import os
from typing import TypeVar

import psycopg
from dotenv import load_dotenv

load_dotenv()

_RowT = TypeVar("_RowT")


def get_connection() -> psycopg.Connection:
    database_url = os.environ["DATABASE_URL"]
    from persistent_cognition.network_consent import require_database_destination
    require_database_destination(database_url)
    return psycopg.connect(database_url)


def require_row(row: _RowT | None, *, context: str) -> _RowT:
    """Return a row a query is required to produce, or fail loudly.

    ``cursor.fetchone()`` is optional by contract even when the surrounding
    statement guarantees exactly one row. Raising here keeps a violated
    expectation an explicit, attributable failure instead of an opaque
    ``NoneType`` error far from its cause.
    """

    if row is None:
        raise RuntimeError(f"expected exactly one row: {context}")
    return row
