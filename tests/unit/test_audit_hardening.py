import hashlib
import json
import os
import stat
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from persistent_cognition import api, blob_store, journal_signing, percept_journal
from persistent_cognition.diagnostics import exception_summary, sanitize_message
from persistent_cognition.private_storage import atomic_private_write, remediate_permissions
from persistent_cognition.resource_limits import (
    ResourceLimitExceeded, bounded_http_request, validate_json_intake, validate_text_intake,
)


@pytest.fixture(autouse=True)
def private_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PCR_ARTIFACT_ROOT", str(tmp_path / "artifacts"))


@pytest.mark.parametrize("suffix", ["g" * 64, "../" + "x" * 61, "/" + "a" * 63,
                                    "\\" + "a" * 63, "a" * 63 + "\0", "A" * 64])
def test_correctly_sized_bad_blob_digests_never_open_a_file(suffix, monkeypatch):
    monkeypatch.setattr(blob_store, "regular_file", lambda *_: pytest.fail("invalid digest reached I/O"))
    with pytest.raises(ValueError):
        blob_store.get_blob("sha256:" + suffix)


@pytest.mark.skipif(os.name != "posix", reason="POSIX links and FIFOs")
@pytest.mark.parametrize("kind", ["leaf", "ancestor", "fifo"])
def test_blob_reads_reject_links_and_nonregular_files(tmp_path, kind):
    data = b"outside the store"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    path = blob_store.blob_path(digest)
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(data)
    if kind == "leaf":
        path.symlink_to(outside)
    elif kind == "ancestor":
        path.parent.rmdir()
        path.parent.symlink_to(tmp_path, target_is_directory=True)
        (tmp_path / path.name).write_bytes(data)
    else:
        os.mkfifo(path)
    with pytest.raises((OSError, ValueError)):
        blob_store.get_blob(digest)
    assert not blob_store.verify_blob(digest)


def test_blob_cap_applies_before_write_and_read(monkeypatch):
    descriptor = blob_store.put_blob(b"123456", media_type="text/plain")
    monkeypatch.setenv("PCR_MAX_BLOB_BYTES", "5")
    with pytest.raises(ResourceLimitExceeded):
        blob_store.put_blob(b"123456", media_type="text/plain")
    with pytest.raises(ResourceLimitExceeded):
        blob_store.get_blob(descriptor.digest)


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes; Windows requires ACL provisioning")
def test_private_creation_and_existing_permission_remediation(tmp_path):
    old_umask = os.umask(0o022)
    try:
        descriptor = blob_store.put_blob(b"private", media_type="text/plain")
        key_id = journal_signing.ensure_signing_key()
        path = percept_journal.path_for(uuid4())
        with percept_journal.locked(path, create=True) as handle:
            percept_journal.append_unlocked(handle, path, {"private": "history"})
    finally:
        os.umask(old_umask)
    root = tmp_path / "artifacts"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for file in [blob_store.blob_path(descriptor.digest), root / "keys" / f"{key_id}.private", path]:
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        file.chmod(0o644)
    report = remediate_permissions(root)
    assert report["files"] >= 3
    assert all(stat.S_IMODE(file.stat().st_mode) == 0o600 for file in root.rglob("*") if file.is_file())
    # The operator's unrelated parent directory is never chmod'ed.
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


def test_quota_and_free_space_admission_reject_without_replacing_history(tmp_path, monkeypatch):
    path = tmp_path / "artifacts" / "history"
    atomic_private_write(path, b"previous")
    monkeypatch.setenv("PCR_ARTIFACT_QUOTA_BYTES", "10")
    with pytest.raises(ResourceLimitExceeded, match="QUOTA"):
        atomic_private_write(path, b"new record")
    assert path.read_bytes() == b"previous"
    monkeypatch.setenv("PCR_ARTIFACT_QUOTA_BYTES", "1000000")
    monkeypatch.setattr("persistent_cognition.private_storage.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=0))
    with pytest.raises(ResourceLimitExceeded, match="free space"):
        atomic_private_write(path, b"new")
    assert path.read_bytes() == b"previous"


def test_intake_checks_utf8_depth_cycles_nodes_and_string_keys(monkeypatch):
    monkeypatch.setenv("PCR_MAX_INTAKE_BYTES", "4")
    validate_text_intake("éé")
    with pytest.raises(ResourceLimitExceeded):
        validate_text_intake("ééé")
    monkeypatch.setenv("PCR_MAX_INTAKE_BYTES", "1000")
    monkeypatch.setenv("PCR_MAX_JSON_DEPTH", "2")
    for value in [[[[0]]]], {1: "value"}, {"float": float("nan")}:
        with pytest.raises(ValueError):
            validate_json_intake(value)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ResourceLimitExceeded):
        validate_json_intake(cycle)
    monkeypatch.setenv("PCR_MAX_JSON_NODES", "3")
    with pytest.raises(ResourceLimitExceeded):
        validate_json_intake([1, 2, 3])


def test_chat_rejects_oversized_input_before_any_database_effect(monkeypatch):
    from persistent_cognition.percept_response_runtime import begin_percept
    monkeypatch.setenv("PCR_MAX_INTAKE_BYTES", "5")
    class NoDatabase:
        def __getattr__(self, name):
            pytest.fail(f"intake touched database: {name}")
    with pytest.raises(ResourceLimitExceeded):
        begin_percept(NoDatabase(), "123456", uuid4())


def test_http_stream_stops_at_cap_and_closes_the_response(monkeypatch):
    monkeypatch.setenv("PCR_MAX_HTTP_RESPONSE_BYTES", "4")
    consumed = []
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for chunk in [b"123", b"456", b"private tail"]:
                consumed.append(chunk)
                yield chunk
        def close(self):
            consumed.append("closed")
    with httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=Stream()))) as client:
        with pytest.raises(ResourceLimitExceeded):
            bounded_http_request(client, "GET", "https://model.test")
    assert consumed == [b"123", b"456", "closed"]


def test_http_compression_is_rejected_before_decoding():
    class NoRead(httpx.SyncByteStream):
        def __iter__(self):
            pytest.fail("compressed payload decoded before admission")
            yield b""
    def handler(request):
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=NoRead())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ResourceLimitExceeded, match="compressed"):
            bounded_http_request(client, "GET", "https://model.test")


def test_journal_pagination_partial_line_and_append_limits(monkeypatch):
    path = percept_journal.path_for(uuid4())
    with percept_journal.locked(path, create=True) as handle:
        for index in range(4):
            percept_journal.append_unlocked(handle, path, {"index": index})
    assert percept_journal.read_page(path, offset=1, page_size=2) == [{"index": 1}, {"index": 2}]
    original = path.read_bytes()
    monkeypatch.setenv("PCR_MAX_JOURNAL_BYTES", str(len(original)))
    with percept_journal.locked(path) as handle, pytest.raises(ResourceLimitExceeded):
        percept_journal.append_unlocked(handle, path, {"new": 1})
    assert path.read_bytes() == original
    monkeypatch.setenv("PCR_MAX_JOURNAL_BYTES", "1000")
    path.write_bytes(original + b'{"torn":')
    with pytest.raises(RuntimeError, match="incomplete"):
        percept_journal.read(path)
    with percept_journal.locked(path) as handle:
        records, tail = percept_journal.parse_unlocked(handle, path, allow_partial=True)
    assert len(records) == 4 and tail == b'{"torn":'


def test_exception_diagnostics_exclude_rejected_inputs_and_redact_before_bounding(monkeypatch):
    class Record(BaseModel):
        number: int
    with pytest.raises(ValidationError) as failure:
        Record(number="private autobiography")
    assert "private autobiography" not in exception_summary(failure.value)
    monkeypatch.setenv("APP_TOKEN", "sensitive-token")
    monkeypatch.setenv("PCR_MAX_DIAGNOSTIC_CHARS", "12")
    assert sanitize_message("sensitive-token repeated") == "[redacted] r"
    assert "autobiography" not in exception_summary(RuntimeError("private autobiography"))


def test_missing_job_settings_fails_before_database_or_model_io(tmp_path, monkeypatch):
    from persistent_cognition import runtime_job
    monkeypatch.delenv("PCR_GUI_JOB_CONFIG", raising=False)
    monkeypatch.delenv("PCR_GUI_JOB_CONFIG_FILE", raising=False)
    (tmp_path / "job.json").write_text(json.dumps({"action": "chat", "payload": {"text": "hello"}}))
    monkeypatch.setattr("persistent_cognition.db.get_connection", lambda: pytest.fail("database opened"))
    with pytest.raises(ValueError, match="require PCR_GUI"):
        runtime_job.run(tmp_path)


def test_public_api_provisions_the_packaged_schema():
    from importlib.resources import files
    captured = []
    conn = SimpleNamespace(execute=captured.append, commit=lambda: captured.append("commit"))
    api.initialize_schema(conn)
    assert "CREATE TABLE IF NOT EXISTS events" in captured[0]
    assert captured[-1] == "commit"
    assert files("persistent_cognition").joinpath("schema.sql").read_text() == (
        files("persistent_cognition").parent.parent.joinpath("schema.sql").read_text())
