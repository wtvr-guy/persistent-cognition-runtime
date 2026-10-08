"""Generic deterministic dependency ordering for Attention-owned work.

This module owns ordering mechanics that should not be delegated to an LLM.
It is intentionally generic: capability execution is the first live consumer,
but any bounded Attention work set can use the same dependency/priority policy.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_ATTENTION_WORK_PRIORITY = 10_000


class AttentionWorkItem(BaseModel):
    """One bounded work item with application-owned dependency metadata."""

    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1)
    priority: int = Field(default=100, ge=0, le=MAX_ATTENTION_WORK_PRIORITY)
    dependency_ids: list[str] = Field(default_factory=list)

    @field_validator("item_id")
    @classmethod
    def normalize_item_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("attention work item_id must not be empty")
        return normalized

    @field_validator("dependency_ids")
    @classmethod
    def normalize_dependency_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("attention work dependencies must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("attention work dependencies must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "AttentionWorkItem":
        if self.item_id in self.dependency_ids:
            raise ValueError("attention work item must not depend on itself")
        return self


def deterministic_dependency_order(items: list[AttentionWorkItem]) -> list[str]:
    """Return a stable topological priority order or fail closed.

    Dependencies always precede dependents. At each decision point, lower
    numeric priority wins among *currently runnable* items and canonical
    ``item_id`` is the final deterministic tie-breaker. Readiness is recomputed
    after every item so newly unblocked higher-priority work is not delayed
    behind a batch of previously runnable lower-priority work. Input list order
    has no effect.
    """

    by_id = {item.item_id: item.model_copy(deep=True) for item in items}
    if len(by_id) != len(items):
        raise ValueError("attention work set must not contain duplicate item ids")

    known_ids = set(by_id)
    for item in by_id.values():
        missing = set(item.dependency_ids) - known_ids
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"attention work item {item.item_id!r} has missing dependencies: "
                f"{missing_text}"
            )

    remaining = set(known_ids)
    ordered: list[str] = []
    while remaining:
        ready = [
            item_id
            for item_id in remaining
            if not (set(by_id[item_id].dependency_ids) & remaining)
        ]
        if not ready:
            raise ValueError("attention work dependencies contain a cycle")
        next_item = min(
            ready,
            key=lambda item_id: (by_id[item_id].priority, item_id),
        )
        ordered.append(next_item)
        remaining.remove(next_item)
    return ordered
