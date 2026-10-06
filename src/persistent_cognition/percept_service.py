"""Always-on, restart-safe governed percept service.

This is a thin, bounded supervisor around the existing deterministic scheduler
tick (:func:`persistent_cognition.percept_cli.tick`). It owns exactly one scheduler
namespace, consumes durable candidates/cursors produced by explicit percept
intake, drains governed situations, backs off when idle, and stops gracefully.

It adds no new cognition: every reaction still flows through the deterministic
situation pipeline and its finite action-feedback governance. The service only
decides *when* to run the next bounded tick, never *what* a tick may do. Because
all progress is recorded in durable cursors, a fresh process resumes exactly
where a stopped or crashed one left off, and the session-scoped ownership lock
prevents two owners from draining the same namespace at once.
"""
from __future__ import annotations

import threading
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

import psycopg

from persistent_cognition import db
from persistent_cognition.advisory_lock import scheduler_ownership
from persistent_cognition.attention_store import DEFAULT_SCHEDULER_KEY
from persistent_cognition.percept_cli import tick
from persistent_cognition.cognitive_store import put_record

logger = logging.getLogger(__name__)

# Idle backoff grows geometrically from this base to the cap, so a quiet runtime
# stops polling tightly without ever blocking new work for longer than the cap.
SERVICE_IDLE_BACKOFF_BASE_SECONDS = 1.0
SERVICE_MAX_BACKOFF_SECONDS = 30.0
SERVICE_MAX_BACKOFF_EXPONENT = 5
# After a tick raises, wait this long before the next bounded attempt instead of
# re-entering a failing tick immediately (which would be a busy failure loop).
SERVICE_ERROR_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class ServiceRunSummary:
    """Outcome of a bounded or signalled service run."""

    iterations: int
    completed_total: int
    error_count: int
    stopped: bool


class PerceptService:
    """Bounded continuous supervisor over one scheduler namespace."""

    def __init__(
        self,
        *,
        scheduler_key: str = DEFAULT_SCHEDULER_KEY,
        idle_backoff_base_seconds: float = SERVICE_IDLE_BACKOFF_BASE_SECONDS,
        max_backoff_seconds: float = SERVICE_MAX_BACKOFF_SECONDS,
        max_backoff_exponent: int = SERVICE_MAX_BACKOFF_EXPONENT,
        error_backoff_seconds: float = SERVICE_ERROR_BACKOFF_SECONDS,
        tick_fn: Callable[..., dict] = tick,
        sleeper: Callable[[float], None] | None = None,
        worker_command: Sequence[str] | None = None,
    ) -> None:
        from persistent_cognition.advisory_lock import scheduler_lock_name
        scheduler_lock_name(scheduler_key)
        for value in (idle_backoff_base_seconds, max_backoff_seconds, error_backoff_seconds):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("service delays must be finite and positive")
        if (isinstance(max_backoff_exponent, bool) or not isinstance(max_backoff_exponent, int)
                or not 0 <= max_backoff_exponent <= SERVICE_MAX_BACKOFF_EXPONENT):
            raise ValueError("max_backoff_exponent exceeds the supported service bound")
        if worker_command is not None and (isinstance(worker_command, (str, bytes)) or not worker_command or
                any(not isinstance(part, str) or not part for part in worker_command)):
            raise ValueError("worker_command must be a nonempty trusted argument sequence")
        self._scheduler_key = scheduler_key
        self._idle_backoff_base_seconds = idle_backoff_base_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._max_backoff_exponent = max_backoff_exponent
        self._error_backoff_seconds = error_backoff_seconds
        self._tick_fn = tick_fn
        self._stop = threading.Event()
        self._sleeper = sleeper or self._interruptible_sleep
        self._worker_command = tuple(worker_command) if worker_command is not None else None

    @property
    def scheduler_key(self) -> str:
        return self._scheduler_key

    def request_stop(self) -> None:
        """Interrupt sleep and stop at the next durable worker boundary."""

        self._stop.set()

    def _interruptible_sleep(self, seconds: float) -> None:
        # Wait on the stop event so a stop signal interrupts an idle delay
        # promptly instead of after the full backoff interval elapses.
        if seconds:
            self._stop.wait(timeout=seconds)

    def _idle_delay(self, idle_steps: int) -> float:
        exponent = min(idle_steps, self._max_backoff_exponent)
        return min(
            self._idle_backoff_base_seconds * (2**exponent),
            self._max_backoff_seconds,
        )

    def run(
        self,
        conn: psycopg.Connection | None = None,
        *,
        max_iterations: int | None = None,
    ) -> ServiceRunSummary:
        """Run until stopped or ``max_iterations`` bounded ticks complete.

        A caller may pass an owned connection (tests, embedding apps). When none
        is given the service opens and owns one for its lifetime so the ownership
        advisory lock is released on exit.
        """

        if max_iterations is not None and (
                isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
                or max_iterations < 0):
            raise ValueError("max_iterations must be a non-negative integer")
        if conn is not None:
            with scheduler_ownership(conn, self._scheduler_key):
                return self._loop(conn, max_iterations=max_iterations)
        with db.get_connection() as owned:
            with scheduler_ownership(owned, self._scheduler_key):
                return self._loop(owned, max_iterations=max_iterations)

    def _loop(
        self,
        conn: psycopg.Connection,
        *,
        max_iterations: int | None,
    ) -> ServiceRunSummary:
        iterations = 0
        completed_total = 0
        errors = 0
        idle_steps = 0
        while not self._stop.is_set():
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1
            try:
                options = {"scheduler_key": self._scheduler_key, "should_stop": self._stop.is_set}
                if self._worker_command is not None:
                    options["worker_command"] = self._worker_command
                result = self._tick_fn(conn, **options)
                completed = result.get("completed") or []
            except Exception as exc:
                logger.exception("Percept tick failed for scheduler %r", self._scheduler_key)
                # A lost session also loses ownership. Exit, never continue work
                # on a reconnected session without reacquiring its lock.
                if conn.closed or conn.broken:
                    raise
                conn.rollback()
                errors += 1
                put_record(conn, "service_error", self._scheduler_key,
                           {"error_type": type(exc).__name__, "iteration": iterations,
                            "observed_at": datetime.now(timezone.utc).isoformat()},
                           revision=str(uuid4()))
                idle_steps = 0
                self._sleeper(self._error_backoff_seconds)
                continue
            completed_total += len(completed)
            if completed:
                # Work remains or just arrived: keep draining without delay. The
                # tick itself is bounded, so this is not a busy spin.
                idle_steps = 0
                continue
            self._sleeper(self._idle_delay(idle_steps))
            idle_steps += 1
        return ServiceRunSummary(
            iterations=iterations,
            completed_total=completed_total,
            error_count=errors,
            stopped=self._stop.is_set(),
        )
