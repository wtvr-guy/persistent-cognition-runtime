"""Operator-tunable bounds applied before persistence or unbounded buffering.

These are deployment safety defaults, not measured optimal cognitive budgets.
Rejected evidence is never silently truncated or partially persisted.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

DEFAULT_MAX_INTAKE_BYTES = 1_048_576
DEFAULT_MAX_JSON_DEPTH = 32
DEFAULT_MAX_JSON_NODES = 65_536
DEFAULT_MAX_HTTP_RESPONSE_BYTES = 4_194_304
DEFAULT_MAX_JOURNAL_BYTES = 67_108_864
DEFAULT_MAX_JOURNAL_ENTRY_BYTES = 8_388_608
DEFAULT_MAX_JOURNAL_RECORDS = 65_536
DEFAULT_MAX_BLOB_BYTES = 67_108_864
DEFAULT_ARTIFACT_QUOTA_BYTES = 1_073_741_824
DEFAULT_MIN_FREE_DISK_BYTES = 67_108_864


class ResourceLimitExceeded(ValueError):
    """An exact operation exceeds its configured safety budget."""


def limit(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_text_intake(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("text intake requires a string")
    maximum = limit("PCR_MAX_INTAKE_BYTES", DEFAULT_MAX_INTAKE_BYTES)
    # Check characters first so an enormous string is not copied just to reject it.
    if len(value) > maximum or len(value.encode("utf-8")) > maximum:
        raise ResourceLimitExceeded("text intake exceeds PCR_MAX_INTAKE_BYTES")


def validate_json_intake(value: Any) -> None:
    """Iteratively bound depth, node count and bytes before JSON serialization.

    Iterator frames avoid copying a wide container. Cycles are rejected through
    the depth bound; no Python recursion or permissive JSON key coercion is used.
    """
    maximum = limit("PCR_MAX_INTAKE_BYTES", DEFAULT_MAX_INTAKE_BYTES)
    max_depth = limit("PCR_MAX_JSON_DEPTH", DEFAULT_MAX_JSON_DEPTH)
    max_nodes = limit("PCR_MAX_JSON_NODES", DEFAULT_MAX_JSON_NODES)
    stack = [(iter((value,)), 0)]
    nodes = byte_count = 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ResourceLimitExceeded("JSON intake exceeds depth or node budget")
        if isinstance(item, str):
            if len(item) > maximum:
                raise ResourceLimitExceeded("JSON intake exceeds byte budget")
            byte_count += len(item.encode("utf-8"))
        elif isinstance(item, dict):
            if len(item) > max_nodes:
                raise ResourceLimitExceeded("JSON intake exceeds node budget")
            def fields(mapping):
                for key, content in mapping.items():
                    if not isinstance(key, str):
                        raise ValueError("JSON object keys must be strings")
                    yield key
                    yield content
            stack.append((fields(item), depth + 1))
        elif isinstance(item, (list, tuple)):
            if len(item) > max_nodes:
                raise ResourceLimitExceeded("JSON intake exceeds node budget")
            stack.append((iter(item), depth + 1))
        elif item is None or isinstance(item, (bool, int)):
            pass
        elif isinstance(item, float) and math.isfinite(item):
            pass
        else:
            raise ValueError("intake must contain finite JSON-compatible values")
        if byte_count > maximum:
            raise ResourceLimitExceeded("JSON intake exceeds byte budget")
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode(value):
        size += len(chunk.encode("utf-8"))
        if size > maximum:
            raise ResourceLimitExceeded("JSON intake exceeds byte budget")


def bounded_http_request(client, method: str, url: str, **kwargs):
    """Read under a cap; refuse compression before any potentially huge decode."""
    maximum = limit("PCR_MAX_HTTP_RESPONSE_BYTES", DEFAULT_MAX_HTTP_RESPONSE_BYTES)
    headers = dict(kwargs.pop("headers", {}))
    headers["Accept-Encoding"] = "identity"
    kwargs["headers"] = headers
    with client.stream(method, url, **kwargs) as response:
        if response.headers.get("content-encoding", "identity").casefold() != "identity":
            raise ResourceLimitExceeded("compressed HTTP responses are not admitted; require identity encoding")
        chunks = []
        size = 0
        for chunk in response.iter_bytes(chunk_size=min(maximum + 1, 65_536)):
            size += len(chunk)
            if size > maximum:
                raise ResourceLimitExceeded("HTTP response exceeds PCR_MAX_HTTP_RESPONSE_BYTES")
            chunks.append(chunk)
        import httpx
        headers = dict(response.headers)
        headers.pop("content-encoding", None)
        headers.pop("content-length", None)
        return httpx.Response(response.status_code, content=b"".join(chunks),
                              headers=headers, request=response.request)


def read_json_file(path):
    from persistent_cognition.private_storage import regular_file
    maximum = limit("PCR_MAX_INTAKE_BYTES", DEFAULT_MAX_INTAKE_BYTES)
    with regular_file(path) as handle:
        if os.fstat(handle.fileno()).st_size > maximum:
            raise ResourceLimitExceeded("JSON file exceeds PCR_MAX_INTAKE_BYTES")
        raw = handle.read(maximum + 1)
        if len(raw) > maximum:
            raise ResourceLimitExceeded("JSON file exceeds PCR_MAX_INTAKE_BYTES")
    try:
        value = json.loads(raw)
    except RecursionError:
        raise ResourceLimitExceeded("JSON file exceeds supported nesting depth") from None
    validate_json_intake(value)
    return value
