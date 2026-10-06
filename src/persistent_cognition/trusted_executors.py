"""Process-local registry of application-owned capability executors.

Reacting to a percept (an error, a long-running tool result, a scheduled event)
may require running application-owned work. That work is dispatched only through
executors an application registers explicitly in its own trusted process
bootstrap. Registrations are never derived from percept data, model output, or a
persisted record, so a fresh worker process re-establishes them from trusted
configuration rather than importing arbitrary code paths named by observed data.

No entry point here evaluates code, imports a module named by data, or treats
percept text as an instruction. Dispatch is keyed only on a pre-authorized
executor name taken from the application-configured capability registration.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# A reserved executor handled directly by the deterministic runtime.
RESERVED_EXECUTORS = frozenset({"jit_memory"})
_EXECUTOR_NAME = re.compile(r"\A[a-z][a-z0-9_]{2,63}\Z")


class TrustedExecutionRequest(BaseModel):
    """Structured, validated input handed to a trusted executor.

    The request carries only typed routing/identity fields. ``task_text`` is the
    observed percept text; a trusted executor may read it as data but the runtime
    never treats it as an instruction to select or run anything.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    capability_id: str = Field(min_length=1)
    executor: str = Field(min_length=1)
    requester_task_id: UUID
    requester_step_id: UUID
    plan_position: int = Field(ge=0)
    conversation_id: UUID
    correlation_id: UUID
    task_text: str
    before_global_seq: int = Field(ge=0)
    execution_id: UUID | None = None


TrustedExecutor = Callable[[TrustedExecutionRequest], Mapping[str, Any]]


class TrustedExecutorError(RuntimeError):
    """A handler failed or returned invalid data, not a persistence failure."""

    def __init__(self, error_type: str, message: str | None = None) -> None:
        super().__init__(message or f"trusted executor failed: {error_type}")
        self.error_type = error_type


_REGISTRY: dict[str, TrustedExecutor] = {}


def validate_executor_name(name: str) -> str:
    if name in RESERVED_EXECUTORS:
        raise ValueError(f"executor name {name!r} is reserved by the runtime")
    if not _EXECUTOR_NAME.fullmatch(name):
        raise ValueError(
            "trusted executor name must be a lowercase identifier "
            "(letters, digits, underscore), 3-64 chars"
        )
    return name


def register_trusted_executor(name: str, handler: TrustedExecutor) -> None:
    """Bind an application-owned executor from trusted startup, never observations."""
    validate_executor_name(name)
    if not callable(handler):
        raise TypeError("trusted executor handler must be callable")
    _REGISTRY[name] = handler


def unregister_trusted_executor(name: str) -> None:
    _REGISTRY.pop(name, None)


def clear_trusted_executors() -> None:
    _REGISTRY.clear()


def trusted_executor_names() -> frozenset[str]:
    return frozenset(_REGISTRY)


def resolve_trusted_executor(name: str) -> TrustedExecutor | None:
    return _REGISTRY.get(name)
