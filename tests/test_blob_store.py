from __future__ import annotations

import pytest

from prometheist import blob_store


def test_put_blob_is_content_addressed_and_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    data = b"exact stateless llm generation payload"

    first = blob_store.put_blob(data, media_type="text/plain")
    second = blob_store.put_blob(data, media_type="text/plain")

    assert first == second
    assert first.digest.startswith("sha256:")
    assert first.size == len(data)
    assert blob_store.blob_exists(first.digest)
    assert blob_store.get_blob(first.digest) == data
    assert blob_store.verify_blob(first.digest) is True
    assert blob_store.verify_descriptor(first) is True


def test_different_bytes_produce_different_digests(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    first = blob_store.put_blob(b"payload one", media_type="text/plain")
    second = blob_store.put_blob(b"payload two", media_type="text/plain")

    assert first.digest != second.digest
    assert blob_store.get_blob(first.digest) == b"payload one"
    assert blob_store.get_blob(second.digest) == b"payload two"


def test_blob_path_uses_sha256_two_character_shard(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    descriptor = blob_store.put_blob(b"shard me", media_type="text/plain")
    hex_digest = descriptor.digest.split(":", 1)[1]

    path = blob_store.blob_path(descriptor.digest)

    assert path.parent.name == hex_digest[:2]
    assert path.name == hex_digest
    assert path.parent.parent.name == "sha256"
    assert path.parent.parent.parent == blob_store.blob_root()


def test_get_blob_missing_raises_file_not_found(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    missing_digest = "sha256:" + "0" * 64

    with pytest.raises(FileNotFoundError):
        blob_store.get_blob(missing_digest)
    assert blob_store.blob_exists(missing_digest) is False
    assert blob_store.verify_blob(missing_digest) is False


def test_get_blob_detects_corrupted_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    descriptor = blob_store.put_blob(b"original bytes", media_type="text/plain")

    path = blob_store.blob_path(descriptor.digest)
    path.write_bytes(b"tampered bytes!!")

    with pytest.raises(blob_store.BlobIntegrityError):
        blob_store.get_blob(descriptor.digest)
    assert blob_store.verify_blob(descriptor.digest) is False
    assert blob_store.verify_descriptor(descriptor) is False


def test_verify_descriptor_rejects_size_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    descriptor = blob_store.put_blob(b"some content", media_type="text/plain")
    wrong_size = blob_store.BlobDescriptor(
        media_type=descriptor.media_type,
        digest=descriptor.digest,
        size=descriptor.size + 1,
    )

    assert blob_store.verify_descriptor(wrong_size) is False


def test_put_blob_rejects_path_collision_with_mismatched_size(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    data = b"legitimate content"
    descriptor = blob_store.put_blob(data, media_type="text/plain")

    # Simulate corruption: the stored bytes were truncated/extended out from
    # under the digest that names them.
    blob_store.blob_path(descriptor.digest).write_bytes(data + b"extra")

    with pytest.raises(blob_store.BlobIntegrityError):
        blob_store.put_blob(data, media_type="text/plain")


@pytest.mark.parametrize(
    "malformed",
    ["not-a-digest", "md5:deadbeef", "sha256:tooshort", "sha256:"],
)
def test_blob_path_rejects_malformed_digest(malformed, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROMETHEIST_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    with pytest.raises(ValueError):
        blob_store.blob_path(malformed)
    assert blob_store.blob_exists(malformed) is False


def test_blob_descriptor_round_trips_oci_style_dict() -> None:
    descriptor = blob_store.BlobDescriptor(
        media_type="application/json", digest="sha256:" + "ab" * 32, size=18442
    )

    as_dict = descriptor.to_dict()

    assert as_dict == {
        "mediaType": "application/json",
        "digest": "sha256:" + "ab" * 32,
        "size": 18442,
    }
    assert blob_store.BlobDescriptor.from_dict(as_dict) == descriptor


def test_iter_blob_descriptors_finds_nested_references() -> None:
    descriptor_dict = {
        "mediaType": "audio/flac",
        "digest": "sha256:" + "11" * 32,
        "size": 48192114,
    }
    payload = {
        "output": {
            "response": descriptor_dict,
            "unrelated": {"nested": {"digest": "not-a-real-digest"}},
        },
        "evidence_refs": ["event:1", "event:2"],
        "attachments": [descriptor_dict, {"note": "no blob here"}],
    }

    found = list(blob_store.iter_blob_descriptors(payload))

    assert found == [
        blob_store.BlobDescriptor.from_dict(descriptor_dict),
        blob_store.BlobDescriptor.from_dict(descriptor_dict),
    ]


def test_iter_blob_descriptors_ignores_dicts_missing_required_fields() -> None:
    payload = {
        "mediaType": "text/plain",
        "digest": "sha256:" + "22" * 32,
        # size intentionally omitted: not a valid descriptor.
    }

    assert list(blob_store.iter_blob_descriptors(payload)) == []
