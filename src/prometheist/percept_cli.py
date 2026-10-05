"""Explicit local administration and one bounded cognition scheduler tick."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from prometheist import db
from prometheist.cognitive_store import get_record, list_records, put_record, rebuild_heads
from prometheist.consolidation import ConsolidationSchedule, emit_due_consolidations, schedule_consolidation
from prometheist.expectations import Expectation
from prometheist.percept_context import PerceptContext
from prometheist.percept_intake import ingest_percept, install_source_policy
from prometheist.perception import PerceptSource
from prometheist.percept_triage import SourcePolicy
from prometheist.reflexes import poll_reflexes
from prometheist.situation_runtime import drain_situations
from prometheist.situations import register_expectation


def tick(conn) -> dict:
    poll_reflexes(conn)
    cursor = get_record(conn, "schedule_cursor", "consolidation") or {"after_key": "", "revision": 0}
    after = emit_due_consolidations(conn, now=datetime.now(timezone.utc), after_key=cursor["after_key"])
    revision = cursor["revision"] + 1
    put_record(conn, "schedule_cursor", "consolidation", {"after_key": after or "", "revision": revision}, revision=str(revision))
    return {"completed": drain_situations(conn), "schedule_cursor": after}


def _require_payload(data: Any, command: str) -> dict[str, Any]:
    """Report a missing or malformed payload explicitly rather than failing on None."""

    if not isinstance(data, dict):
        raise SystemExit(f"{command} requires a JSON object payload")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("install-source", "ingest", "expectation", "schedule-consolidation"):
        commands.add_parser(name).add_argument("json_file", type=Path)
    commands.add_parser("tick")
    consolidate = commands.add_parser("consolidate", help="Schedule one bounded consolidation page and run one scheduler tick")
    consolidate.add_argument("--after-key", default="")
    commands.add_parser("rebuild-heads")
    show = commands.add_parser("situations")
    show.add_argument("--after-key", default="")
    args = parser.parse_args()
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
            print(json.dumps(tick(conn), indent=2))
        elif args.command == "tick":
            print(json.dumps(tick(conn), indent=2))
        elif args.command == "situations":
            print(json.dumps(list_records(conn, "situation", after_key=args.after_key), indent=2))
        elif args.command == "rebuild-heads":
            rebuild_heads(conn)


if __name__ == "__main__":
    main()
