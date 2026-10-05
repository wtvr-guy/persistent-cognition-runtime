"""Signed finalized journal heads: an optional audit-strengthening layer.

Every mechanism in ``artifact_journal.py`` already makes ordinary local
corruption or truncation *detectable* (hash-chained records, sequence-gap
checks). What it cannot do alone is defend against a single actor with full
filesystem access rewriting both an interaction's artifacts *and* whatever
"proves" they were not rewritten -- the exact trap SuperLocalMemory's
sidecar-anchor design falls into, since its anchor lives right next to the
data it is meant to attest.

This module adds one additional, independent trust primitive: an Ed25519
digital signature over a finalized interaction's journal head, verifiable
with only the *public* key -- so a copy of a signed head can be handed to a
separate machine, a remote object store, or a human, and that party can
confirm the signature without needing write access to (or any trust in the
current state of) the local artifact root.

This is deliberately **not** wired into the live percept-response pipeline.
Signing is an explicit, opt-in operation over an already-finalized
interaction (``prometheist sign``), consistent with the research finding
that "the signature/remote anchor is not necessary for basic crash recovery;
it is an audit-strengthening layer." It is also, by construction, tamper
*evident* rather than tamper *proof*: a local actor who controls the signing
key can still sign a false statement. What it defends against is a *later*
rewrite of already-signed history being passed off as the original.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from prometheist.artifact_journal import artifact_root, verify_interaction_chain
from prometheist import percept_journal

SIGNATURE_ALGORITHM = "ed25519"


def _keys_root() -> Path:
    return artifact_root() / "keys"


def _anchors_root() -> Path:
    return artifact_root() / "anchors"


def _private_key_path(key_id: str) -> Path:
    return _keys_root() / f"{key_id}.private"


def _public_key_path(key_id: str) -> Path:
    return _keys_root() / f"{key_id}.public"


def _anchor_path(interaction_id: UUID) -> Path:
    return _anchors_root() / f"{interaction_id}.json"


def _fsync_parent(path: Path) -> None:
    """Best-effort directory fsync; Windows does not expose POSIX directory fsync."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path.parent)


def key_id_for_public_bytes(public_bytes: bytes) -> str:
    """A short, stable, public identifier for a key -- never the key itself."""

    return hashlib.sha256(public_bytes).hexdigest()[:16]


def ensure_signing_key() -> str:
    """Return the local signing key's id, generating a keypair on first use.

    The private key is written once and never rotated automatically; a
    deliberate key rotation is an explicit operator decision (a new key id
    naturally coexists with older ones since every anchor records the
    ``signing_key_id`` that produced it).
    """

    existing = list(_keys_root().glob("*.private")) if _keys_root().exists() else []
    if existing:
        return existing[0].stem

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes_raw()
    key_id = key_id_for_public_bytes(public_bytes)
    _atomic_write_bytes(_private_key_path(key_id), private_key.private_bytes_raw())
    _atomic_write_bytes(_public_key_path(key_id), public_bytes)
    return key_id


def _load_private_key(key_id: str) -> Ed25519PrivateKey:
    path = _private_key_path(key_id)
    if not path.exists():
        raise FileNotFoundError(f"unknown signing key id: {key_id}")
    return Ed25519PrivateKey.from_private_bytes(path.read_bytes())


def _load_public_key(key_id: str) -> Ed25519PublicKey:
    path = _public_key_path(key_id)
    if not path.exists():
        raise FileNotFoundError(f"unknown signing key id: {key_id}")
    return Ed25519PublicKey.from_public_bytes(path.read_bytes())


def _canonical_anchor_bytes(anchor: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in anchor.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_journal_head(interaction_id: UUID) -> dict[str, Any]:
    """Sign an already-finalized interaction's current journal head.

    Fails closed if the chain does not verify or is not yet finalized: a
    signature must never assert integrity/completeness the journal itself
    cannot already demonstrate.
    """

    verification = verify_interaction_chain(interaction_id)
    if not verification["valid"]:
        raise ValueError(f"refusing to sign an invalid artifact chain: {verification['errors']}")
    if not verification["complete"]:
        raise ValueError("refusing to sign an interaction with no FINAL_DISPOSITION artifact")

    key_id = ensure_signing_key()
    private_key = _load_private_key(key_id)
    anchor: dict[str, Any] = {
        "schema": "prometheist.journal-anchor/v1",
        "interaction_id": str(interaction_id),
        "journal_head": verification["last_artifact"]["artifact_hash"],
        "record_count": verification["artifact_count"],
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "signing_key_id": key_id,
        "algorithm": SIGNATURE_ALGORITHM,
    }
    stream = percept_journal.path_for(interaction_id)
    if stream.exists():
        contents = percept_journal.inspect(stream)
        if contents["uncommitted_events"]:
            raise ValueError("refusing to sign a percept with missing database receipts")
        anchor["percept_journal_sha256"] = hashlib.sha256(stream.read_bytes()).hexdigest()
    signature = private_key.sign(_canonical_anchor_bytes(anchor))
    anchor["signature"] = signature.hex()
    _atomic_write_bytes(_anchor_path(interaction_id), (json.dumps(anchor, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return anchor


def load_signed_journal_head(interaction_id: UUID) -> dict[str, Any] | None:
    path = _anchor_path(interaction_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_signed_journal_head(interaction_id: UUID) -> dict[str, Any]:
    """Verify a previously written anchor against the current journal state.

    Uses only the public key: this check never needs, and never touches,
    the private signing key, so it can run anywhere the anchor and the
    public key file have been copied.
    """

    anchor = load_signed_journal_head(interaction_id)
    if anchor is None:
        return {"interaction_id": str(interaction_id), "signed": False, "valid": False, "reason": "no signed anchor found"}

    try:
        signature = bytes.fromhex(anchor["signature"])
        public_key = _load_public_key(anchor["signing_key_id"])
        public_key.verify(signature, _canonical_anchor_bytes(anchor))
    except (InvalidSignature, KeyError, ValueError, FileNotFoundError) as exc:
        return {
            "interaction_id": str(interaction_id), "signed": True, "valid": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    verification = verify_interaction_chain(interaction_id)
    if not verification["valid"]:
        return {
            "interaction_id": str(interaction_id), "signed": True, "valid": False,
            "reason": f"signature is valid but the artifact chain no longer is: {verification['errors']}",
        }
    current_head = verification["last_artifact"]["artifact_hash"] if verification["last_artifact"] else None
    if anchor["journal_head"] != current_head:
        return {
            "interaction_id": str(interaction_id), "signed": True, "valid": False,
            "reason": "signature is valid but no longer matches the current journal head",
        }
    if "percept_journal_sha256" in anchor:
        stream = percept_journal.path_for(interaction_id)
        try:
            percept_journal.inspect(stream)
            current_digest = hashlib.sha256(stream.read_bytes()).hexdigest()
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "interaction_id": str(interaction_id), "signed": True, "valid": False,
                "reason": f"percept journal cannot be verified: {exc}",
            }
        if current_digest != anchor["percept_journal_sha256"]:
            return {
                "interaction_id": str(interaction_id), "signed": True, "valid": False,
                "reason": "signature is valid but no longer matches the percept journal",
            }
    return {
        "interaction_id": str(interaction_id), "signed": True, "valid": True,
        "signing_key_id": anchor["signing_key_id"], "journal_head": anchor["journal_head"],
        "signed_at": anchor["signed_at"],
    }
