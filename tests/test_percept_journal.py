"""The percept work product is a single resumable file with typed records."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from uuid import uuid4, uuid5

import pytest

from persistent_cognition import artifact_journal, event_artifact_store, journal_signing, percept_journal
from persistent_cognition.llm import OllamaClient
from persistent_cognition.model_evidence_budget import (
    ModelEvidenceBudgetExceeded,
    validate_model_input,
)


def test_events_receipts_and_worker_checkpoints_share_one_percept_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    conversation_id, correlation_id = uuid4(), uuid4()
    interaction_id = uuid5(conversation_id, f"interaction:{correlation_id}")
    event_id = uuid4()
    arguments = dict(
        event_id=event_id, conversation_id=conversation_id, correlation_id=correlation_id,
        conversation_seq=1, event_type="USER_PROMPT", source="user",
        payload={"text": "hello"}, payload_text="hello", journal_id=interaction_id,
    )
    first = event_artifact_store.write_event_record(**arguments)
    assert event_artifact_store.write_event_record(**arguments) == first
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id, conversation_id=conversation_id,
        correlation_id=correlation_id, task_id=interaction_id,
        user_text="hello", user_prompt_event_id=event_id,
    )
    commit = event_artifact_store.write_event_commit(
        event_id=event_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=1, global_seq=7,
        created_at=datetime(2026, 9, 27, tzinfo=timezone.utc), schema_version=1,
    )
    assert event_artifact_store.write_event_commit(
        event_id=event_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=1, global_seq=7,
        created_at=datetime(2026, 9, 27, tzinfo=timezone.utc), schema_version=1,
    ) == commit
    second_id = uuid4()
    event_artifact_store.write_event_record(
        event_id=second_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=2,
        event_type="MEMORY_REQUEST", source="jit_memory",
        payload={"query": "hello"}, payload_text=None,
    )
    event_artifact_store.write_event_commit(
        event_id=second_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=2, global_seq=8,
        created_at=datetime(2026, 9, 27, tzinfo=timezone.utc), schema_version=1,
    )
    artifact_journal.write_stage_result_artifact(
        interaction_id=interaction_id, conversation_id=conversation_id,
        correlation_id=correlation_id, task_id=interaction_id,
        assignment_id=uuid4(), stage="V2_PRECOGNITIVE",
        output={"decision": "proceed"}, output_refs=[f"event:{event_id}"],
    )
    path = percept_journal.path_for(interaction_id)
    assert list(tmp_path.rglob("*.jsonl")) == [path]
    assert [item["artifact_type"] for item in percept_journal.read(path)] == [
        "EVENT_RECORD", "PERCEPT", "EVENT_DATABASE_COMMIT",
        "EVENT_RECORD", "EVENT_DATABASE_COMMIT", "STAGE_RESULT",
    ]
    assert percept_journal.inspect(path)["record_count"] == 6
    assert percept_journal.inspect(path)["uncommitted_events"] == 0
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"]
    assert len(event_artifact_store.iter_event_artifacts()) == 2
    assert artifact_journal.load_stage_result_artifact(interaction_id, "V2_PRECOGNITIVE") == {
        "output": {"decision": "proceed"}, "output_refs": [f"event:{event_id}"],
    }


def test_concurrent_worker_appends_keep_one_valid_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    interaction_id, conversation_id, correlation_id = uuid4(), uuid4(), uuid4()
    def append(index):
        return artifact_journal.write_interaction_artifact(
            artifact_key=f"step:{index}", artifact_type="STAGE_RESULT",
            interaction_id=interaction_id, conversation_id=conversation_id,
            correlation_id=correlation_id, payload={"index": index}, producer="test",
        )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(append, range(20)))
    assert len({result["artifact_id"] for result in results}) == 20
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"]
    assert len(list(tmp_path.rglob("*.jsonl"))) == 1


def test_torn_tail_blocks_retries_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    interaction_id, conversation_id, correlation_id = uuid4(), uuid4(), uuid4()
    artifact_journal.write_interaction_artifact(
        artifact_key="start", artifact_type="PERCEPT", interaction_id=interaction_id,
        conversation_id=conversation_id, correlation_id=correlation_id,
        payload={}, producer="test",
    )
    path = percept_journal.path_for(interaction_id)
    with path.open("ab") as handle:
        handle.write(b'{"artifact_type":"STAGE_RESULT"')
    with pytest.raises(RuntimeError, match="incomplete percept journal"):
        artifact_journal.interaction_artifacts(interaction_id)
    with pytest.raises(RuntimeError, match="incomplete percept journal"):
        artifact_journal.write_interaction_artifact(
            artifact_key="next", artifact_type="STAGE_RESULT", interaction_id=interaction_id,
            conversation_id=conversation_id, correlation_id=correlation_id,
            payload={}, producer="test",
        )


def test_retry_repairs_only_matching_partial_database_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    conversation_id, correlation_id, event_id = uuid4(), uuid4(), uuid4()
    interaction_id = uuid5(conversation_id, f"interaction:{correlation_id}")
    arguments = dict(
        event_id=event_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=1,
        event_type="USER_PROMPT", source="user", payload={"text": "hi"},
        payload_text="hi", journal_id=interaction_id,
    )
    record = event_artifact_store.write_event_record(**arguments)
    committed_at = datetime(2026, 9, 27, tzinfo=timezone.utc)
    semantic = {
        "artifact_schema_version": 1, "artifact_type": "EVENT_DATABASE_COMMIT",
        "event_id": str(event_id), "record_hash": record["record_hash"],
        "global_seq": 5, "conversation_seq": 1,
        "created_at": committed_at.isoformat(), "schema_version": 1,
    }
    stamp = {**semantic, "commit_hash": event_artifact_store._digest(semantic)}
    encoded = json.dumps(stamp, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path = percept_journal.path_for(interaction_id)
    with path.open("ab") as handle:
        handle.write(encoded[:38])
    assert event_artifact_store.write_event_record(**arguments) == record
    assert event_artifact_store.write_event_commit(
        event_id=event_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=1, global_seq=5,
        created_at=committed_at, schema_version=1,
    ) == stamp
    assert percept_journal.read(path)[-1] == stamp


def test_complete_input_budget_counts_current_prompt_and_schema(monkeypatch):
    monkeypatch.setenv("PCR_MAX_MODEL_INPUT_BYTES", "100")
    with pytest.raises(ModelEvidenceBudgetExceeded, match="complete model input"):
        validate_model_input({
            "messages": [{"role": "system", "content": "policy"},
                         {"role": "user", "content": "x" * 90}],
            "format": {"type": "object"},
        })
    monkeypatch.setenv("PCR_MAX_MODEL_INPUT_BYTES", "1000")
    validate_model_input({"prompt": "small", "format": {"type": "object"}})


def test_full_prompt_budget_blocks_transport(monkeypatch):
    monkeypatch.setenv("PCR_MAX_MODEL_INPUT_BYTES", "100")
    client = OllamaClient(base_url="http://ollama.test", model="model:test")
    attempted = []
    monkeypatch.setattr(client._client, "post", lambda *args, **kwargs: attempted.append(args))
    with pytest.raises(ModelEvidenceBudgetExceeded, match="complete model input"):
        client._structured("policy", "system", "x" * 200, {"type": "object"}, 16)
    assert attempted == []
    client._client.close()


def test_signed_percept_covers_event_records_as_well_as_worker_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    conversation_id, correlation_id, event_id = uuid4(), uuid4(), uuid4()
    interaction_id = uuid5(conversation_id, f"interaction:{correlation_id}")
    event_artifact_store.write_event_record(
        event_id=event_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=1,
        event_type="USER_PROMPT", source="user", payload={"text": "original"},
        payload_text="original", journal_id=interaction_id,
    )
    event_artifact_store.write_event_commit(
        event_id=event_id, conversation_id=conversation_id,
        correlation_id=correlation_id, conversation_seq=1, global_seq=1,
        created_at=datetime(2026, 9, 27, tzinfo=timezone.utc), schema_version=1,
    )
    artifact_journal.write_percept_artifact(
        interaction_id=interaction_id, conversation_id=conversation_id,
        correlation_id=correlation_id, task_id=interaction_id,
        user_text="original", user_prompt_event_id=event_id,
    )
    artifact_journal.write_final_disposition_artifact(
        interaction_id=interaction_id, conversation_id=conversation_id,
        correlation_id=correlation_id, task_id=interaction_id,
        assignment_id=uuid4(), response_required=True, response_text="done",
    )
    journal_signing.sign_journal_head(interaction_id)
    path = percept_journal.path_for(interaction_id)
    entries = percept_journal.read(path)
    entries[0]["payload_text"] = "changed"
    entries[0]["record_hash"] = event_artifact_store._digest({
        key: value for key, value in entries[0].items()
        if key not in {"record_hash", "journaled_at"}
    })
    entries[1]["record_hash"] = entries[0]["record_hash"]
    entries[1]["commit_hash"] = event_artifact_store._digest({
        key: value for key, value in entries[1].items() if key != "commit_hash"
    })
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    assert percept_journal.inspect(path)["event_count"] == 1
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"]
    signed = journal_signing.verify_signed_journal_head(interaction_id)
    assert signed["valid"] is False
    assert "no longer matches the percept journal" in signed["reason"]


pytestmark = pytest.mark.usefixtures("consented_mock_ollama")
