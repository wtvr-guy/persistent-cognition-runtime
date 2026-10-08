from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from persistent_cognition import artifact_journal, event_artifact_store, llm_artifact_store
from tests._native_artifact_assertions import assert_response_evidence_receipt


def test_interaction_artifacts_are_hash_linked_idempotent_and_complete(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    conversation_id = uuid4()
    correlation_id = uuid4()
    task_id = uuid4()
    assignment_id = uuid4()
    prompt_event_id = uuid4()

    percept = artifact_journal.write_percept_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        user_text="remember the phrase",
        user_prompt_event_id=prompt_event_id,
    )
    first_stage = artifact_journal.write_stage_result_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_PRECOGNITIVE",
        output={"disposition": {"response_required": True}},
        output_refs=["memory-request:example"],
    )
    retry = artifact_journal.write_stage_result_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_PRECOGNITIVE",
        output={"disposition": {"response_required": True}},
        output_refs=["memory-request:example"],
    )
    assert retry["artifact_id"] == first_stage["artifact_id"]
    assert len(artifact_journal.interaction_artifacts(interaction_id)) == 2

    loaded = artifact_journal.load_stage_result_artifact(
        interaction_id,
        "V2_PRECOGNITIVE",
    )
    assert loaded == {
        "output": {"disposition": {"response_required": True}},
        "output_refs": ["memory-request:example"],
    }

    artifact_journal.write_final_disposition_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        response_required=True,
        response_text="done",
    )
    verification = artifact_journal.verify_interaction_chain(interaction_id)
    assert verification["valid"] is True
    assert verification["complete"] is True
    assert verification["artifact_count"] == 3
    assert percept["previous_artifact_id"] is None


def test_llm_invocation_artifact_preserves_exact_stateless_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    conversation_id = uuid4()
    correlation_id = uuid4()
    task_id = uuid4()
    assignment_id = uuid4()
    claim_id = uuid4()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    artifact = llm_artifact_store.write_llm_invocation(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_RESPOND",
        claim_id=claim_id,
        invocation_index=0,
        kind="FINAL_RESPONSE",
        model="qwen3:4b",
        base_url="http://localhost:11434",
        system_prompt="exact system prompt",
        user_prompt="exact response context",
        schema=schema,
        max_tokens=256,
        temperature=0.65,
        output='{"answer":"hello"}',
        error_type=None,
        error_message=None,
    )

    assert artifact["artifact_type"] == "LLM_INVOCATION"
    assert artifact["producer"] == "percept_response_v2/ollama"
    assert artifact["payload"] == {
        "provider": "ollama",
        "claim_id": str(claim_id),
        "invocation_index": 0,
        "kind": "FINAL_RESPONSE",
        "model": "qwen3:4b",
        "base_url": "http://localhost:11434",
        "system_prompt": "exact system prompt",
        "user_prompt": "exact response context",
        "schema": schema,
        "max_tokens": 256,
        "temperature": 0.65,
        "output": '{"answer":"hello"}',
        "error_type": None,
        "error_message": None,
    }
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"] is True


def test_llm_validation_artifact_links_parse_outcome_to_invocation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    conversation_id = uuid4()
    correlation_id = uuid4()
    task_id = uuid4()
    assignment_id = uuid4()
    claim_id = uuid4()
    invocation = llm_artifact_store.write_llm_invocation(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_COMPOSE_MEMORY",
        claim_id=claim_id,
        invocation_index=0,
        kind="V2_MEMORY_SUFFICIENCY_USER_PROMPT",
        model="qwen3:4b",
        base_url="http://localhost:11434",
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object"},
        max_tokens=96,
        temperature=0.0,
        output="not-json",
        error_type=None,
        error_message=None,
    )

    validation = llm_artifact_store.write_llm_validation(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_COMPOSE_MEMORY",
        claim_id=claim_id,
        invocation_index=0,
        kind="V2_MEMORY_SUFFICIENCY_USER_PROMPT",
        invocation_artifact_id=str(invocation["artifact_id"]),
        invocation_artifact_hash=str(invocation["artifact_hash"]),
        status="INVALID",
        raw_output_sha256="f" * 64,
        parsed_output=None,
        error_type="ValidationError",
        error_message="invalid JSON",
    )

    assert validation["artifact_type"] == "LLM_VALIDATION"
    assert validation["payload"]["invocation_artifact_id"] == invocation["artifact_id"]
    assert validation["payload"]["invocation_artifact_hash"] == invocation["artifact_hash"]
    assert validation["payload"]["status"] == "INVALID"
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"] is True


def test_successful_llm_validation_does_not_create_a_second_artifact(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    conversation_id = uuid4()
    correlation_id = uuid4()
    task_id = uuid4()
    assignment_id = uuid4()
    claim_id = uuid4()
    invocation = llm_artifact_store.write_llm_invocation(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_RESPOND",
        claim_id=claim_id,
        invocation_index=0,
        kind="FINAL_RESPONSE_V2",
        model="model:test",
        base_url="http://localhost:11434",
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object"},
        max_tokens=96,
        temperature=0.0,
        output='{"answer":"yes"}',
        error_type=None,
        error_message=None,
    )
    validation = llm_artifact_store.write_llm_validation(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_RESPOND",
        claim_id=claim_id,
        invocation_index=0,
        kind="FINAL_RESPONSE_V2",
        invocation_artifact_id=str(invocation["artifact_id"]),
        invocation_artifact_hash=str(invocation["artifact_hash"]),
        status="VALID",
        raw_output_sha256="f" * 64,
        parsed_output={"answer": "yes"},
        error_type=None,
        error_message=None,
    )
    assert validation is None
    assert [item["artifact_type"] for item in artifact_journal.interaction_artifacts(
        interaction_id
    )] == ["LLM_INVOCATION"]
    artifact_journal.write_stage_result_artifact(
        interaction_id=interaction_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        task_id=task_id,
        assignment_id=assignment_id,
        stage="V2_RESPOND",
        output={"answer": "yes"},
        output_refs=(),
    )
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"] is True


def test_llm_invocation_filename_is_bounded_independently_of_semantic_key(
    tmp_path,
    monkeypatch,
) -> None:
    nested_root = tmp_path / "deep-path-segment" / "another-segment" / "artifacts"
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(nested_root))
    interaction_id = uuid4()
    claim_id = uuid4()

    artifact = llm_artifact_store.write_llm_invocation(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=uuid4(),
        assignment_id=uuid4(),
        stage="V2_COMPOSE_MEMORY",
        claim_id=claim_id,
        invocation_index=0,
        kind="V2_MEMORY_SUFFICIENCY_USER_PROMPT",
        model="qwen3:4b",
        base_url="http://localhost:11434",
        system_prompt="system",
        user_prompt="user",
        schema={"type": "object"},
        max_tokens=192,
        temperature=0.0,
        output='{"sufficient":true}',
        error_type=None,
        error_message=None,
    )

    filename = Path(artifact["_path"]).name
    assert len(filename) <= 44
    assert str(claim_id) not in filename
    assert "V2_MEMORY_SUFFICIENCY_USER_PROMPT" not in filename
    assert artifact["artifact_key"].endswith("V2_MEMORY_SUFFICIENCY_USER_PROMPT")
    assert artifact_journal.verify_interaction_chain(interaction_id)["valid"] is True


def test_evidence_bound_llm_artifact_records_separate_transport_channels(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    artifact = llm_artifact_store.write_llm_invocation(
        interaction_id=uuid4(),
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=uuid4(),
        assignment_id=uuid4(),
        stage="V2_RESPOND",
        claim_id=uuid4(),
        invocation_index=0,
        kind="V2_EXACT_SOURCE_SELECTION",
        model="qwen3:4b-instruct-2507-q4_K_M",
        base_url="http://localhost:11434",
        system_prompt="system policy",
        user_prompt="current user authority",
        evidence_prompt="quarantined historical evidence",
        transport_layout="raw-generate:system,evidence,current-user,assistant",
        schema={"type": "object"},
        max_tokens=256,
        temperature=0.0,
        output='{"source_index":0,"verbatim_value":"value"}',
        error_type=None,
        error_message=None,
        evidence_refs=("event:source-1", "event:source-2"),
    )

    assert artifact["payload"]["user_prompt"] == "current user authority"
    assert artifact["payload"]["evidence_prompt"] == "quarantined historical evidence"
    assert artifact["payload"]["transport_layout"] == (
        "raw-generate:system,evidence,current-user,assistant"
    )
    assert artifact["payload"]["evidence_refs"] == [
        "event:source-1",
        "event:source-2",
    ]


def test_native_artifact_oracle_verifies_response_evidence_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    interaction_id = uuid4()
    required_event_id = uuid4()
    forbidden_event_id = uuid4()
    llm_artifact_store.write_llm_invocation(
        interaction_id=interaction_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        task_id=uuid4(),
        assignment_id=uuid4(),
        stage="V2_RESPOND",
        claim_id=uuid4(),
        invocation_index=0,
        kind="FINAL_RESPONSE_V2",
        model="qwen3:4b",
        base_url="http://localhost:11434",
        system_prompt="system",
        user_prompt="current prompt",
        evidence_prompt="quarantined evidence",
        transport_layout="chat:system,tool-evidence,current-user",
        schema={"type": "object"},
        max_tokens=256,
        temperature=0.65,
        output='{"answer":"review me"}',
        error_type=None,
        error_message=None,
        evidence_refs=(f"event:{required_event_id}",),
    )

    receipt = assert_response_evidence_receipt(
        interaction_id=interaction_id,
        required_event_ids=(required_event_id,),
        forbidden_event_ids=(forbidden_event_id,),
        require_complete=False,
    )

    assert receipt["response_kind"] == "FINAL_RESPONSE_V2"
    assert receipt["evidence_refs"] == [f"event:{required_event_id}"]


def test_event_artifacts_are_semantically_idempotent_and_verifiable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    event_id = uuid4()
    conversation_id = uuid4()
    correlation_id = uuid4()

    first = event_artifact_store.write_event_record(
        event_id=event_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        conversation_seq=7,
        event_type="USER_PROMPT",
        source="user",
        payload={"text": "hello"},
        payload_text="hello",
    )
    second = event_artifact_store.write_event_record(
        event_id=event_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        conversation_seq=7,
        event_type="USER_PROMPT",
        source="user",
        payload={"text": "hello"},
        payload_text="hello",
    )
    assert first["record_hash"] == second["record_hash"]
    assert event_artifact_store.verify_event_record(first)
    event_files = list((tmp_path / "artifacts" / "events").iterdir())
    assert len(event_files) == 1
    assert event_files[0].suffix == ".jsonl"

    committed_at = datetime.now(timezone.utc)
    commit = event_artifact_store.write_event_commit(
        event_id=event_id,
        global_seq=42,
        conversation_seq=7,
        created_at=committed_at,
        schema_version=1,
    )
    retry = event_artifact_store.write_event_commit(
        event_id=event_id,
        global_seq=42,
        conversation_seq=7,
        created_at=committed_at,
        schema_version=1,
    )
    assert commit == retry
    assert event_artifact_store.verify_event_commit(commit)
    assert list((tmp_path / "artifacts" / "events").iterdir()) == event_files
    assert len(event_files[0].read_text(encoding="utf-8").splitlines()) == 2
    pairs = event_artifact_store.iter_event_artifacts()
    assert len(pairs) == 1
    assert pairs[0]["record"]["event_id"] == str(event_id)
    assert pairs[0]["commit"]["global_seq"] == 42


def test_event_stream_repairs_only_a_matching_interrupted_commit(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    event_id, conversation_id, correlation_id = uuid4(), uuid4(), uuid4()
    record = event_artifact_store.write_event_record(
        event_id=event_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        conversation_seq=1,
        event_type="USER_PROMPT",
        source="user",
        payload={"text": "test"},
        payload_text="test",
    )
    path = tmp_path / "events" / f"{event_id}.jsonl"
    created_at = datetime.now(timezone.utc)
    semantic = {
        "artifact_schema_version": 1,
        "artifact_type": "EVENT_DATABASE_COMMIT",
        "event_id": str(event_id),
        "record_hash": record["record_hash"],
        "global_seq": 5,
        "conversation_seq": 1,
        "created_at": created_at.isoformat(),
        "schema_version": 1,
    }
    expected = {**semantic, "commit_hash": event_artifact_store._digest(semantic)}
    encoded = event_artifact_store._line(expected)
    with path.open("ab") as handle:
        handle.write(encoded[:25])
    with pytest.raises(RuntimeError, match="incomplete"):
        event_artifact_store.iter_event_artifacts()

    commit = event_artifact_store.write_event_commit(
        event_id=event_id,
        global_seq=5,
        conversation_seq=1,
        created_at=created_at,
        schema_version=1,
    )
    assert commit == expected
    before_retry = path.read_bytes()
    assert event_artifact_store.write_event_commit(
        event_id=event_id,
        global_seq=5,
        conversation_seq=1,
        created_at=created_at,
        schema_version=1,
    ) == commit
    assert path.read_bytes() == before_retry
    with pytest.raises(ValueError, match="conflicting event commit"):
        event_artifact_store.write_event_commit(
            event_id=event_id,
            global_seq=6,
            conversation_seq=1,
            created_at=created_at,
            schema_version=1,
        )
    path.write_bytes(path.read_bytes().replace(b'"global_seq":5', b'"global_seq":6'))
    with pytest.raises(RuntimeError, match="invalid event commit"):
        event_artifact_store.read_event_file(path)


def test_event_stream_rejects_corrupt_tail_and_reads_legacy_pair(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    event_id, conversation_id, correlation_id = uuid4(), uuid4(), uuid4()
    record = event_artifact_store.write_event_record(
        event_id=event_id,
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        conversation_seq=1,
        event_type="USER_PROMPT",
        source="user",
        payload={"text": "test"},
        payload_text="test",
    )
    path = tmp_path / "events" / f"{event_id}.jsonl"
    with path.open("ab") as handle:
        handle.write(b"garbage")
    with pytest.raises(RuntimeError, match="conflicting incomplete"):
        event_artifact_store.write_event_commit(
            event_id=event_id,
            global_seq=1,
            conversation_seq=1,
            created_at=datetime.now(timezone.utc),
            schema_version=1,
        )

    # A pre-upgrade event remains a two-file pair on retry; no migration rewrites it.
    path.rename(tmp_path / "events" / f"{event_id}.jsonl.saved")
    legacy = tmp_path / "events" / f"{event_id}.json"
    event_artifact_store._atomic_write(legacy, record)
    committed_at = datetime.now(timezone.utc)
    commit = event_artifact_store.write_event_commit(
        event_id=event_id,
        global_seq=1,
        conversation_seq=1,
        created_at=committed_at,
        schema_version=1,
    )
    assert (tmp_path / "events" / f"{event_id}.commit.json").exists()
    assert event_artifact_store.read_event_file(legacy) == (record, commit)


def test_parallel_event_commit_retries_append_only_one_stamp(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    event_id = uuid4()
    event_artifact_store.write_event_record(
        event_id=event_id,
        conversation_id=uuid4(),
        correlation_id=uuid4(),
        conversation_seq=1,
        event_type="USER_PROMPT",
        source="user",
        payload={"text": "test"},
        payload_text="test",
    )
    committed_at = datetime.now(timezone.utc)
    start = Barrier(4)

    def commit_retry():
        start.wait(timeout=30)
        return event_artifact_store.write_event_commit(
            event_id=event_id,
            global_seq=1,
            conversation_seq=1,
            created_at=committed_at,
            schema_version=1,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _index: commit_retry(), range(8)))
    assert all(result == results[0] for result in results)
    path = tmp_path / "events" / f"{event_id}.jsonl"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.fixture
def stream_event(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path))
    arguments = dict(
        event_id=uuid4(), conversation_id=uuid4(), correlation_id=uuid4(),
        conversation_seq=1, event_type="USER_PROMPT", source="user",
        payload={"text": "test"}, payload_text="test",
    )
    record = event_artifact_store.write_event_record(**arguments)
    path = tmp_path / "events" / f"{arguments['event_id']}.jsonl"
    return arguments, record, path


def _stream_operation(operation, arguments, path):
    if operation == "read":
        return event_artifact_store.read_event_file(path)
    if operation == "record_retry":
        return event_artifact_store.write_event_record(**arguments)
    if operation == "commit_retry":
        return event_artifact_store.write_event_commit(
            event_id=arguments["event_id"], global_seq=1, conversation_seq=1,
            created_at=datetime.now(timezone.utc), schema_version=1,
        )
    return event_artifact_store.iter_event_artifacts()


@pytest.mark.parametrize("operation", ["read", "record_retry", "commit_retry", "inventory"])
def test_event_stream_reads_use_the_owned_handle(stream_event, monkeypatch, operation):
    """Model mandatory locks even on POSIX, where unlocked reads often succeed."""
    from contextlib import contextmanager

    arguments, record, path = stream_event
    owned = set()
    reads = []
    regular_file = event_artifact_store.regular_file
    locked = event_artifact_store._locked_event_file
    read_bytes = Path.read_bytes

    class CheckedHandle:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, *args):
            assert self.handle.fileno() in owned, "stream read preceded lock acquisition"
            reads.append(self.handle.fileno())
            return self.handle.read(*args)

    @contextmanager
    def checked_file(*args, **kwargs):
        with regular_file(*args, **kwargs) as handle:
            yield CheckedHandle(handle)

    @contextmanager
    def tracked_lock(handle):
        with locked(handle):
            owned.add(handle.fileno())
            try:
                yield
            finally:
                owned.remove(handle.fileno())

    def deny_unlocked_read(candidate):
        if candidate == path:
            raise PermissionError("a second handle cannot read the locked stream")
        return read_bytes(candidate)

    monkeypatch.setattr(event_artifact_store, "regular_file", checked_file)
    monkeypatch.setattr(event_artifact_store, "_locked_event_file", tracked_lock)
    monkeypatch.setattr(Path, "read_bytes", deny_unlocked_read)
    result = _stream_operation(operation, arguments, path)
    assert reads and not owned
    if operation == "read":
        assert result == (record, None)
    elif operation == "record_retry":
        assert result == record
    elif operation == "commit_retry":
        assert event_artifact_store.verify_event_commit(result)
        assert event_artifact_store.read_event_file(path) == (record, result)
    else:
        assert result == [{"record": record, "commit": None}]


@pytest.mark.parametrize("operation", ["read", "record_retry", "commit_retry", "inventory"])
def test_native_windows_event_reader_waits_for_writer(stream_event, monkeypatch, operation):
    import os
    if os.name != "nt":
        pytest.skip("requires native Windows mandatory byte-range locks")
    import msvcrt
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    arguments, record, path = stream_event
    busy = Event()
    native_locking = msvcrt.locking

    def observed_locking(fd, mode, size):
        try:
            return native_locking(fd, mode, size)
        except OSError:
            busy.set()
            raise

    monkeypatch.setattr(msvcrt, "locking", observed_locking)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with path.open("r+b") as writer, event_artifact_store._locked_event_file(writer):
            # Establish that this host really enforces locks on other handles.
            with pytest.raises(PermissionError):
                path.read_bytes()
            future = pool.submit(_stream_operation, operation, arguments, path)
            assert busy.wait(timeout=5), "reader did not wait for the writer's lock"
            assert not future.done()
        result = future.result(timeout=20)
    stored_record, commit = event_artifact_store.read_event_file(path)
    assert stored_record == record
    if operation == "commit_retry":
        assert commit == result
    else:
        assert commit is None
