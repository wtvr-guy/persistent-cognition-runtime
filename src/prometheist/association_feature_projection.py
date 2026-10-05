"""Disposable features used for incremental association maintenance.

The canonical event ledger remains authoritative. These features exist only so
ordinary projection catch-up can find the bounded predecessor evidence required
by deterministic association rules without reloading lifetime history into
Python.
"""
from __future__ import annotations

from typing import Any, Mapping

from prometheist.association_projection import (
    ASSOCIATION_PROJECTION_VERSION,
    _CHANGE_PHRASES,
    _RESOLVED_PHRASES,
    _UNRESOLVED_PHRASES,
    _concepts,
    _contains_any,
    _entity_terms,
)
from prometheist.memory_kernel import MemoryEvent


class AssociationFeatureProjection:
    """Index only event-local facts needed by association predecessor lookup."""

    name = "association_features"
    version = ASSOCIATION_PROJECTION_VERSION

    def project(self, event: MemoryEvent) -> Mapping[str, Any]:
        return {
            "source": event.source,
            "concepts": sorted(_concepts(event)),
            "entity_terms": sorted(_entity_terms(event)),
            "is_change": _contains_any(event.text, _CHANGE_PHRASES),
            "is_unresolved": _contains_any(event.text, _UNRESOLVED_PHRASES),
            "is_resolved": _contains_any(event.text, _RESOLVED_PHRASES),
        }
