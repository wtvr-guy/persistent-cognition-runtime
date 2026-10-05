import pytest

from prometheist import db
from prometheist.cognitive_store import get_record, put_record, record_history


@pytest.fixture
def conn():
    with db.get_connection() as connection:
        yield connection


def test_record_history_returns_every_revision_oldest_first(conn):
    put_record(conn, "test_kind", "a", {"value": 1}, revision="r1")
    put_record(conn, "test_kind", "a", {"value": 2}, revision="r2")
    put_record(conn, "test_kind", "a", {"value": 3}, revision="r3")

    history = record_history(conn, "test_kind", "a")

    assert history == [{"value": 1}, {"value": 2}, {"value": 3}]
    # The rebuildable head still only exposes the latest revision.
    assert get_record(conn, "test_kind", "a") == {"value": 3}


def test_record_history_is_scoped_to_one_exact_key(conn):
    put_record(conn, "test_kind", "a", {"value": "a1"}, revision="r1")
    put_record(conn, "test_kind", "b", {"value": "b1"}, revision="r1")
    put_record(conn, "other_kind", "a", {"value": "other"}, revision="r1")

    assert record_history(conn, "test_kind", "a") == [{"value": "a1"}]
    assert record_history(conn, "test_kind", "b") == [{"value": "b1"}]
    assert record_history(conn, "other_kind", "a") == [{"value": "other"}]


def test_record_history_is_empty_for_an_unknown_key(conn):
    assert record_history(conn, "test_kind", "never-written") == []
