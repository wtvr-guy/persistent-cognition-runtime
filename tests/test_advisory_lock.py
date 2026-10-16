"""Real PostgreSQL regressions for scheduler-ownership transaction cleanup."""
from uuid import uuid4

import psycopg
import pytest

from persistent_cognition import db
from persistent_cognition.advisory_lock import scheduler_ownership


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_python_failure_rolls_back_pending_writes_and_releases_ownership(failure):
    key = f"failed-operation-{uuid4()}"
    with db.get_connection() as conn:
        conn.execute("CREATE TEMP TABLE ownership_writes (value integer)")
        conn.commit()
        error = failure("body failed after successful SQL")
        with pytest.raises(failure) as raised:
            with scheduler_ownership(conn, key):
                conn.execute("INSERT INTO ownership_writes VALUES (1)")
                assert conn.info.transaction_status is psycopg.pq.TransactionStatus.INTRANS
                raise error
        assert raised.value is error
        assert conn.execute("SELECT count(*) FROM ownership_writes").fetchone()[0] == 0
        # Keep the original session alive: closing it would conceal a leaked lock.
        with db.get_connection() as other:
            with scheduler_ownership(other, key):
                assert other.execute("SELECT 1").fetchone()[0] == 1


def test_successful_ownership_keeps_existing_commit_behavior():
    with db.get_connection() as conn:
        conn.execute("CREATE TEMP TABLE ownership_writes (value integer)")
        conn.commit()
        with scheduler_ownership(conn, f"successful-operation-{uuid4()}"):
            conn.execute("INSERT INTO ownership_writes VALUES (1)")
        conn.rollback()
        assert conn.execute("SELECT value FROM ownership_writes").fetchone()[0] == 1
