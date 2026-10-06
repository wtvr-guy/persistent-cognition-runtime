"""Explicit local administration and one bounded cognition scheduler tick."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import signal
from pathlib import Path
from typing import Any
from collections.abc import Sequence
from uuid import UUID, uuid4

from persistent_cognition import db
from persistent_cognition.advisory_lock import scheduler_ownership
from persistent_cognition.attention_store import DEFAULT_SCHEDULER_KEY
from persistent_cognition.cognitive_store import get_record, list_records, put_record, rebuild_heads
from persistent_cognition.consolidation import ConsolidationSchedule, emit_due_consolidations, schedule_consolidation
from persistent_cognition.expectations import Expectation
from persistent_cognition.percept_context import PerceptContext
from persistent_cognition.percept_intake import ingest_percept, install_source_policy
from persistent_cognition.perception import PerceptSource
from persistent_cognition.percept_triage import SourcePolicy
from persistent_cognition.reflexes import poll_reflexes
from persistent_cognition.situation_runtime import drain_situations
from persistent_cognition.situations import register_expectation


def tick(
    conn, *, scheduler_key: str = DEFAULT_SCHEDULER_KEY,
    worker_command: Sequence[str] | None = None,
    should_stop=None,
) -> dict:
    """Drain one bounded page under the same ownership boundary as chat/service."""
    with scheduler_ownership(conn, scheduler_key):
        poll_reflexes(conn, scheduler_key=scheduler_key)
        cursor = get_record(conn, "schedule_cursor", scheduler_key) or {"after_key": "", "revision": 0}
        after = emit_due_consolidations(conn, now=datetime.now(timezone.utc), after_key=cursor["after_key"])
        revision = cursor["revision"] + 1
        put_record(conn, "schedule_cursor", scheduler_key, {"after_key": after or "", "revision": revision}, revision=str(revision))
        return {"completed": drain_situations(conn, scheduler_key=scheduler_key, worker_command=worker_command,
                                             should_stop=should_stop),
                "schedule_cursor": after}


def _require_payload(data: Any, command: str) -> dict[str, Any]:
    """Report a missing or malformed payload explicitly rather than failing on None."""

    if not isinstance(data, dict):
        raise SystemExit(f"{command} requires a JSON object payload")
    return data


def _serve(scheduler_key: str) -> dict:
    """Run the always-on governed percept service until a stop signal arrives."""

    from persistent_cognition.percept_service import PerceptService

    service = PerceptService(scheduler_key=scheduler_key)

    def _request_stop(_signum, _frame) -> None:
        service.request_stop()

    previous = {
        number: signal.getsignal(number)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    for number in previous:
        signal.signal(number, _request_stop)
    try:
        summary = service.run()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return {
        "scheduler_key": scheduler_key,
        "iterations": summary.iterations,
        "completed_total": summary.completed_total,
        "error_count": summary.error_count,
        "stopped": summary.stopped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("install-source", "ingest", "expectation", "schedule-consolidation"):
        commands.add_parser(name).add_argument("json_file", type=Path)
    commands.add_parser("tick").add_argument("--scheduler-key", default=DEFAULT_SCHEDULER_KEY)
    serve = commands.add_parser("serve", help="Run the always-on governed percept service until SIGINT/SIGTERM")
    serve.add_argument("--scheduler-key", default="default")
    consolidate = commands.add_parser("consolidate", help="Schedule one bounded consolidation page and run one scheduler tick")
    consolidate.add_argument("--after-key", default="")
    consolidate.add_argument("--scheduler-key", default=DEFAULT_SCHEDULER_KEY)
    commands.add_parser("rebuild-heads")
    show = commands.add_parser("situations")
    show.add_argument("--after-key", default="")
    args = parser.parse_args()
    if args.command == "serve":
        print(json.dumps(_serve(args.scheduler_key), indent=2))
        return
    with db.get_connection() as conn:
        data = json.loads(args.json_file.read_text(encoding="utf-8")) if hasattr(args, "json_file") else None
        if args.command == "install-source":
            install_source_policy(conn, SourcePolicy.model_validate(_require_payload(data, args.command)))
        elif args.command == "expectation":
            register_expectation(conn, Expectation.model_validate(_require_payload(data, args.command)))
        elif args.command == "schedule-consolidation":
            schedule_consolidation(conn, ConsolidationSchedule.model_validate(_require_payload(data, args.command)))
        elif args.command == "ingest":
            payload = _require_payload(data, args.command)
            kwargs = {"conversation_id": UUID(payload["conversation_id"])} if payload.get("conversation_id") else {}
            percept = ingest_percept(
                conn, source=PerceptSource.model_validate(payload["source"]), observation=payload["observation"],
                observed_at=datetime.fromisoformat(payload["observed_at"]), delivery_id=payload["delivery_id"],
                context=PerceptContext.model_validate(payload.get("context", {})),
                correlation_id=UUID(payload["correlation_id"]) if payload.get("correlation_id") else None, **kwargs,
            )
            print(percept.model_dump_json(indent=2))
        elif args.command == "consolidate":
            schedule_consolidation(conn, ConsolidationSchedule(
                schedule_id=uuid4(), due_at=datetime.now(timezone.utc), after_key=args.after_key,
            ))
            print(json.dumps(tick(conn, scheduler_key=args.scheduler_key), indent=2))
        elif args.command == "tick":
            print(json.dumps(tick(conn, scheduler_key=args.scheduler_key), indent=2))
        elif args.command == "situations":
            print(json.dumps(list_records(conn, "situation", after_key=args.after_key), indent=2))
        elif args.command == "rebuild-heads":
            rebuild_heads(conn)


if __name__ == "__main__":
    main()
