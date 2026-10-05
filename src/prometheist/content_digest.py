"""Canonical hashing shared by engine policy records."""
import hashlib
import json


def content_digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()
