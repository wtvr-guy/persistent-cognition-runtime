from __future__ import annotations

from uuid import uuid4

from prometheist import artifact_recovery, db, event_store
from prometheist.models import EventType


def test_restore_event_store_reconstructs_deleted_canonical_history() -> None:
    conversation_id = uuid4()
    correlation_id = uuid4()
    event_id = uuid4()

    with db.get_connection() as conn:
        event_store.start_conversation(conn, conversation_id)
        original = event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            event_type=EventType.USER_PROMPT,
            source="user",
            payload={"text": "artifact recovery target"},
            payload_text="artifact recovery target",
            event_id=event_id,
        )
        with conn.cursor() as cur:
            cur.execute("DELETE FROM events WHERE event_id = %s", (event_id,))
            cur.execute(
                "DELETE FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            )
        conn.commit()

        assert event_store.get_event_by_id(conn, event_id) is None
        restored = artifact_recovery.restore_event_store_from_artifacts(conn)
        recovered = event_store.get_event_by_id(conn, event_id)

        assert restored == {
            "artifact_events": 1,
            "inserted_events": 1,
            "existing_events": 0,
        }
        assert recovered is not None
        assert recovered.event_id == original.event_id
        assert recovered.conversation_id == original.conversation_id
        assert recovered.correlation_id == original.correlation_id
        assert recovered.global_seq == original.global_seq
        assert recovered.conversation_seq == original.conversation_seq
        assert recovered.created_at == original.created_at
        assert recovered.payload == original.payload

        with conn.cursor() as cur:
            cur.execute(
                "SELECT next_event_seq FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            )
            next_event_seq = int(cur.fetchone()[0])
        assert next_event_seq == original.conversation_seq + 1
