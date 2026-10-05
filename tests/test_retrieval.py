import uuid

import pytest

from prometheist import db, event_store, retrieval
from prometheist.models import EventType, RetrievalRequest


@pytest.fixture
def conn():
    connection = db.get_connection()
    yield connection
    connection.close()


def _seed(conn, conversation_id, correlation_id, texts):
    for text in texts:
        event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            event_type=EventType.USER_PROMPT,
            source="test",
            payload={"text": text},
            payload_text=text,
        )


def test_structured_filter_scoped_to_current_conversation(conn):
    conv_a = event_store.start_conversation(conn)
    conv_b = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()

    _seed(conn, conv_a, correlation_id, ["alpha message"])
    _seed(conn, conv_b, correlation_id, ["beta message"])

    result = retrieval.search(
        conn, conv_a, RetrievalRequest(conversation_scope="CURRENT_CONVERSATION", limit=10)
    )
    contents = [item.content for item in result.items]
    assert "alpha message" in contents
    assert "beta message" not in contents


def test_full_text_search_matches_shared_keywords(conn):
    conversation_id = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()
    token = uuid.uuid4().hex[:8]
    _seed(
        conn,
        conversation_id,
        correlation_id,
        [
            f"The codename for Project {token} is Silver Birch.",
            "unrelated message about lunch plans",
        ],
    )

    result = retrieval.search(
        conn,
        conversation_id,
        RetrievalRequest(query_text=f"codename Project {token}", ordering="RELEVANCE", limit=5),
    )
    assert len(result.items) >= 1
    assert token in result.items[0].content
    assert result.items[0].rank is not None


def test_limit_is_positive_and_respected_without_arbitrary_global_ceiling(conn):
    conversation_id = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()
    _seed(conn, conversation_id, correlation_id, [f"message {i}" for i in range(10)])

    result = retrieval.search(conn, conversation_id, RetrievalRequest(limit=3))
    assert len(result.items) == 3

    # The request contract enforces only the structural positivity invariant.
    # Query/candidate policy owns performance bounds; a legacy magic maximum does not.
    request = RetrievalRequest(limit=1000)
    assert request.limit == 1000

    with pytest.raises(Exception):
        RetrievalRequest(limit=0)


def test_retrieval_items_include_source_event_id(conn):
    conversation_id = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()
    recorded = event_store.record_event(
        conn,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": "traceable message"},
        payload_text="traceable message",
    )

    result = retrieval.search(conn, conversation_id, RetrievalRequest(limit=1))
    assert result.items[0].source_event_id == recorded.event_id


def test_all_conversations_scope_finds_other_conversations(conn):
    """Conversation boundaries are organizational metadata, not memory walls:
    ALL_CONVERSATIONS must be able to surface content from a conversation
    other than the one the current request happens to be scoped from.
    """
    conv_a = event_store.start_conversation(conn)
    conv_b = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()
    token = uuid.uuid4().hex[:8]

    _seed(conn, conv_a, correlation_id, [f"The codename for Project {token} is Iron Wolf."])

    result = retrieval.search(
        conn,
        conv_b,
        RetrievalRequest(
            query_text=f"codename Project {token}",
            conversation_scope="ALL_CONVERSATIONS",
            ordering="RELEVANCE",
            limit=5,
        ),
    )
    assert any(token in item.content for item in result.items)
    matching = [item for item in result.items if token in item.content]
    assert all(item.conversation_id == conv_a for item in matching)


def test_global_seq_cutoff_is_meaningful_across_conversations(conn):
    """`conversation_seq` isn't comparable across conversations; a
    cross-conversation cutoff must use `global_seq` instead, or it will
    incorrectly filter unrelated conversations by an unrelated local number.
    """
    conv_a = event_store.start_conversation(conn)
    correlation_id = uuid.uuid4()
    token = uuid.uuid4().hex[:8]
    early = event_store.record_event(
        conn,
        conversation_id=conv_a,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": f"early {token} fact"},
        payload_text=f"early {token} fact",
    )

    conv_b = event_store.start_conversation(conn)
    later = event_store.record_event(
        conn,
        conversation_id=conv_b,
        correlation_id=correlation_id,
        event_type=EventType.USER_PROMPT,
        source="test",
        payload={"text": f"later {token} fact"},
        payload_text=f"later {token} fact",
    )
    assert later.global_seq > early.global_seq

    result = retrieval.search(
        conn,
        conv_b,
        RetrievalRequest(query_text=token, conversation_scope="ALL_CONVERSATIONS", limit=10),
        before_global_seq=later.global_seq,
    )
    contents = [item.content for item in result.items]
    assert f"early {token} fact" in contents
    assert f"later {token} fact" not in contents
