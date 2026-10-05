"""Disposable, versioned projections derived from immutable evidence."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Protocol

from prometheist.memory_integrity import canonical_event_bytes
from prometheist.memory_kernel import MemoryEvent, normalize_text, tokenize


@dataclass(frozen=True, slots=True)
class ProjectionEntry:
    projection_name: str
    projection_version: str
    source_event_id: str
    source_global_seq: int
    source_hash: str
    data: Mapping[str, Any]


class Projection(Protocol):
    name: str
    version: str

    def project(self, event: MemoryEvent) -> Mapping[str, Any]: ...


class LexicalProjection:
    """Small deterministic v1 index: terms plus explicitly supplied entities."""

    name = "lexical"
    version = "1"

    def project(self, event: MemoryEvent) -> Mapping[str, Any]:
        explicit_entities = event.payload.get("entities", [])
        entities = sorted(
            {
                normalize_text(str(entity))
                for entity in explicit_entities
                if isinstance(entity, (str, int, float)) and str(entity).strip()
            }
        )
        return {
            "terms": sorted(set(tokenize(event.text))),
            "entities": entities,
            "event_type": event.event_type,
            "source": event.source,
        }


def build_projection(events: Iterable[MemoryEvent], projection: Projection) -> tuple[ProjectionEntry, ...]:
    entries: list[ProjectionEntry] = []
    for event in sorted(events, key=lambda item: (item.global_seq, item.event_id)):
        source_hash = hashlib.sha256(canonical_event_bytes(event)).hexdigest()
        entries.append(
            ProjectionEntry(
                projection_name=projection.name,
                projection_version=projection.version,
                source_event_id=event.event_id,
                source_global_seq=event.global_seq,
                source_hash=source_hash,
                data=projection.project(event),
            )
        )
    return tuple(entries)


def projection_digest(entries: Iterable[ProjectionEntry]) -> str:
    canonical = [
        {
            "projection_name": entry.projection_name,
            "projection_version": entry.projection_version,
            "source_event_id": entry.source_event_id,
            "source_global_seq": entry.source_global_seq,
            "source_hash": entry.source_hash,
            "data": entry.data,
        }
        for entry in sorted(entries, key=lambda item: (item.source_global_seq, item.source_event_id))
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
