from __future__ import annotations

import uuid

from psycopg.types.json import Json

from prometheist import cli, db, event_store
from prometheist.chat_startup import reset_chat_execution_state
from prometheist.models import EventType


def test_chat_execution_reset_preserves_memory_resources_and_other_scheduler() -> None:
    with db.get_connection() as conn:
        conversation_id = event_store.start_conversation(conn)
        event = event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=uuid.uuid4(),
            event_type=EventType.USER_PROMPT,
            source="user",
            payload={"text": "persistent evidence"},
            payload_text="persistent evidence",
        )
        default_task_id = uuid.uuid4()
        other_task_id = uuid.uuid4()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attention_execution_resources (
                    resource_id, resource_class, capacity, system_headroom, enabled, metadata
                )
                VALUES ('local-llm', 'LLM_INFERENCE', 1, 0, true, '{}'::jsonb)
                """
            )
            cur.execute("SELECT nextval('attention_task_created_seq')")
            default_created_seq = int(cur.fetchone()[0])
            cur.execute("SELECT nextval('attention_task_created_seq')")
            other_created_seq = int(cur.fetchone()[0])
            cur.execute(
                """
                INSERT INTO attention_tasks (
                    scheduler_key, task_id, task_key, created_seq,
                    criticality, service_class, interruption_policy, status
                )
                VALUES
                    ('default', %s, 'default-stale', %s,
                     'USER_BLOCKING', 'INTERACTIVE', 'PREEMPTIBLE', 'QUEUED'),
                    ('other', %s, 'other-live', %s,
                     'USER_REQUESTED', 'USER_WORK', 'PREEMPTIBLE', 'QUEUED')
                """,
                (default_task_id, default_created_seq, other_task_id, other_created_seq),
            )
            cur.execute(
                """
                INSERT INTO attention_scheduler_state (
                    scheduler_key, admitted_task_ids
                )
                VALUES
                    ('default', %s),
                    ('other', %s)
                """,
                (Json([str(default_task_id)]), Json([str(other_task_id)])),
            )
        conn.commit()

        counts = reset_chat_execution_state(conn)
        assert counts["attention_tasks"] == 1
        assert counts["attention_scheduler_state"] == 1

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            )
            assert int(cur.fetchone()[0]) == 1
            cur.execute("SELECT count(*) FROM events WHERE event_id = %s", (event.event_id,))
            assert int(cur.fetchone()[0]) == 1
            cur.execute(
                "SELECT count(*) FROM attention_execution_resources WHERE resource_id = 'local-llm'"
            )
            assert int(cur.fetchone()[0]) == 1
            cur.execute(
                "SELECT count(*) FROM attention_tasks WHERE scheduler_key = 'default'"
            )
            assert int(cur.fetchone()[0]) == 0
            cur.execute(
                "SELECT count(*) FROM attention_scheduler_state WHERE scheduler_key = 'default'"
            )
            assert int(cur.fetchone()[0]) == 0
            cur.execute(
                "SELECT count(*) FROM attention_tasks WHERE scheduler_key = 'other'"
            )
            assert int(cur.fetchone()[0]) == 1
            cur.execute(
                "SELECT count(*) FROM attention_scheduler_state WHERE scheduler_key = 'other'"
            )
            assert int(cur.fetchone()[0]) == 1


def test_run_chat_resets_execution_state_before_accepting_input(monkeypatch) -> None:
    calls: list[object] = []
    fake_conn = object()

    class ConnectionContext:
        def __enter__(self):
            calls.append("connection")
            return fake_conn

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cli.db, "get_connection", lambda: ConnectionContext())
    monkeypatch.setattr(
        cli,
        "reset_chat_execution_state",
        lambda conn: calls.append(("reset", conn)) or {},
    )
    monkeypatch.setattr(
        cli.artifact_journal,
        "latest_interaction_id",
        lambda *, complete: None,
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "/exit")

    cli._run_chat(uuid.uuid4())

    assert calls == ["connection", ("reset", fake_conn)]
