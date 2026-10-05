"""Deterministic evidence selection; never a claim that memory is sufficient."""
from collections import deque
from itertools import zip_longest
from uuid import uuid5

from prometheist.model_evidence_budget import configured_model_evidence_budget
from prometheist.models import MemoryPacket

FIXED_RETRIEVAL_VERSION = "fixed-retrieval/v1"


def _temporal_order(items):
    """Alternate recent and older evidence within each bounded retrieval route."""
    queue = deque(sorted(items, key=lambda i: (i.global_seq, str(i.source_event_id))))
    while queue:
        yield queue.pop()
        if queue:
            yield queue.popleft()


def merge_evidence(interaction, packets, source_types, item_limit):
    """Round-robin routes and evidence roles, retaining exact bytes and lineage.

    Bounds apply to the view only. Excluded canonical events remain untouched.
    Duplicate discoveries merge their navigation provenance, not their content.
    """
    if item_limit < 1 or not packets:
        raise ValueError("retrieval requires a positive item limit and an initial packet")
    budget = configured_model_evidence_budget()
    by_id = {}
    groups = []
    rejected = {}
    for packet in packets:
        roles = {}
        for item in packet.items:
            key = item.source_event_id
            if item.event_type not in source_types or item.global_seq >= interaction.before_global_seq:
                rejected[str(key)] = "source_or_cutoff"
                continue
            if key in by_id:
                prior = by_id[key]
                if prior.content != item.content or prior.global_seq != item.global_seq:
                    raise ValueError("conflicting canonical evidence for one event ID")
                prior.retrieval_reasons = sorted(set(prior.retrieval_reasons + item.retrieval_reasons))
                prior.provenance_event_ids = sorted(set(prior.provenance_event_ids + item.provenance_event_ids), key=str)
            else:
                by_id[key] = item.model_copy(deep=True)
            # Separate known opposition so supporting evidence cannot crowd it out.
            relation = "OPPOSES" if "OPPOSES" in item.retrieval_reasons else "OTHER"
            roles.setdefault((relation, item.event_type.value), []).append(item)
        for role in sorted(roles):
            groups.append(tuple(_temporal_order(roles[role])))
    selected, seen, total = [], set(), 0
    for row in zip_longest(*groups):
        for candidate in row:
            if candidate is None or candidate.source_event_id in seen:
                continue
            seen.add(candidate.source_event_id)
            item = by_id[candidate.source_event_id]
            size = len(item.content.encode("utf-8"))
            if size > budget.max_item_bytes or total + size > budget.max_total_bytes:
                rejected[str(item.source_event_id)] = "byte_budget"
                continue
            if len(selected) == item_limit:
                rejected[str(item.source_event_id)] = "item_budget"
                continue
            selected.append(item)
            total += size
    return MemoryPacket(
        memory_request_id=uuid5(interaction.interaction_id, f"{FIXED_RETRIEVAL_VERSION}:merge:{len(packets)}"),
        need=packets[0].need.model_copy(update={"limit": item_limit}),
        supported=bool(selected), items=selected,
        retrieval_trace={"policy": FIXED_RETRIEVAL_VERSION,
                         "input_requests": [str(p.memory_request_id) for p in packets],
                         "routes": [p.retrieval_trace for p in packets],
                         "excluded": rejected, "content_bytes": total},
    )
