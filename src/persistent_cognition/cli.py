"""First-party Persistent Cognition CLI.

The chat REPL owns presentation only.  Artifact inspection and verification do not
require PostgreSQL; recovery and event restoration use the independent JSON
journal as durable checkpoints/reconstruction input.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid

from persistent_cognition import (
    artifact_journal,
    artifact_recovery,
    audit_report,
    blob_store,
    db,
    journal_signing,
    semantic_memory,
)
from persistent_cognition.admission_diagnostics import (
    RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX,
    build_resource_admission_diagnostics,
)
from persistent_cognition.attention_store import load_scheduler
from persistent_cognition.chat_startup import reset_chat_execution_state
from persistent_cognition.percept_response_runtime import handle_percept_in_worker_processes
from persistent_cognition.worker_runtime import WorkerLaunchDenied

_INTERACTION_ADMISSION_FAILURE = "interaction was not safely admitted to one assignment"
_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}


def _configure_utf8_streams() -> None:
    """Make Windows subprocess output deterministic before argument parsing."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _emit_admission_diagnostics(diagnostics: dict) -> None:
    print(
        RESOURCE_ADMISSION_DIAGNOSTIC_PREFIX
        + json.dumps(diagnostics, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
    )


def _handle_with_admission_diagnostics(
    conn,
    user_text: str,
    conversation_id: uuid.UUID,
    **runtime_options,
) -> str | None:
    """Run one percept and expose the authoritative denial envelope on failure."""

    try:
        return handle_percept_in_worker_processes(
            conn,
            user_text,
            conversation_id,
            **runtime_options,
        )
    except WorkerLaunchDenied as exc:
        _emit_admission_diagnostics(
            {
                "kind": "WORKER_CLAIM_DENIED",
                "worker_claim_observation": exc.observation.model_dump(mode="json"),
            }
        )
        raise
    except RuntimeError as exc:
        if str(exc) != _INTERACTION_ADMISSION_FAILURE:
            raise
        try:
            scheduler = load_scheduler(conn)
            diagnostics = build_resource_admission_diagnostics(scheduler)
            diagnostics["kind"] = "SCHEDULER_ADMISSION_DENIED"
        except Exception as diagnostic_exc:  # never mask the original runtime failure
            diagnostics = {
                "kind": "SCHEDULER_ADMISSION_DIAGNOSTIC_ERROR",
                "diagnostic_error": f"{type(diagnostic_exc).__name__}: {diagnostic_exc}",
            }
        _emit_admission_diagnostics(diagnostics)
        raise


def _resolve_artifact_interaction_id(
    interaction_id: uuid.UUID | None,
    *,
    latest: bool,
    complete: bool | None = None,
) -> uuid.UUID:
    if interaction_id is not None:
        return interaction_id
    if not latest:
        raise RuntimeError("specify --interaction-id or --latest")
    resolved = artifact_journal.latest_interaction_id(complete=complete)
    if resolved is None:
        qualifier = " incomplete" if complete is False else ""
        raise RuntimeError(f"no{qualifier} interaction artifacts found")
    return resolved


def _run_inspect(interaction_id: uuid.UUID | None, *, latest: bool) -> None:
    resolved = _resolve_artifact_interaction_id(interaction_id, latest=latest)
    print(
        json.dumps(
            artifact_journal.inspect_interaction(resolved),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def _run_verify(interaction_id: uuid.UUID | None, *, latest: bool) -> None:
    resolved = _resolve_artifact_interaction_id(interaction_id, latest=latest)
    report = artifact_journal.verify_interaction_chain(resolved)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["valid"]:
        raise RuntimeError("artifact chain verification failed")


def _run_audit(interaction_id: uuid.UUID | None, *, latest: bool) -> None:
    """Print a deterministic human-readable timeline; a projection, not authoritative."""

    resolved = _resolve_artifact_interaction_id(interaction_id, latest=latest)
    print(audit_report.render_interaction_audit(resolved))


def _run_sign(interaction_id: uuid.UUID | None, *, latest: bool) -> None:
    """Sign an already-finalized interaction's journal head. Opt-in; not automatic."""

    resolved = _resolve_artifact_interaction_id(interaction_id, latest=latest)
    anchor = journal_signing.sign_journal_head(resolved)
    print(json.dumps(anchor, indent=2, sort_keys=True))


def _run_verify_signature(interaction_id: uuid.UUID | None, *, latest: bool) -> None:
    resolved = _resolve_artifact_interaction_id(interaction_id, latest=latest)
    report = journal_signing.verify_signed_journal_head(resolved)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise RuntimeError("journal signature verification failed")


def _run_blob_verify(digest: str) -> None:
    valid = blob_store.verify_blob(digest)
    print(json.dumps({"digest": digest, "valid": valid}, sort_keys=True))
    if not valid:
        raise RuntimeError(f"blob verification failed: {digest}")


def _run_memory_fact(subject: str, property_name: str) -> None:
    """Print assertions, evidence, and resolution for one subject/property."""

    with db.get_connection() as conn:
        assertions = semantic_memory.semantic_assertions(conn, subject, property_name)
        evidence = semantic_memory.semantic_evidence(conn, subject, property_name)
        resolution = semantic_memory.current_semantic_resolution(
            conn, subject, property_name
        )
        resolution_history = semantic_memory.semantic_resolution_history(
            conn, subject, property_name
        )
        selected = (
            semantic_memory.selected_assertion(conn, resolution)
            if resolution is not None
            else None
        )
    payload = {
        "subject": subject,
        "property": property_name,
        "current_resolution": (
            resolution.model_dump(mode="json") if resolution else None
        ),
        "selected_assertion": (
            selected.model_dump(mode="json") if selected else None
        ),
        "assertions": [item.model_dump(mode="json") for item in assertions],
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "resolution_history": [
            item.model_dump(mode="json") for item in resolution_history
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _run_recover(interaction_id: uuid.UUID | None, *, latest: bool) -> None:
    resolved = _resolve_artifact_interaction_id(
        interaction_id,
        latest=latest,
        complete=False if interaction_id is None else None,
    )
    with db.get_connection() as conn:
        response = artifact_recovery.resume_interaction_from_artifacts(conn, resolved)
    if response is not None:
        print(response)


def _run_restore_events() -> None:
    with db.get_connection() as conn:
        result = artifact_recovery.restore_event_store_from_artifacts(conn)
    print(json.dumps(result, indent=2, sort_keys=True))


def _reset_failed_turn(conn, exc: BaseException) -> None:
    """Release replaceable execution state after one failed interactive turn."""

    try:
        reset_chat_execution_state(conn)
    except Exception as reset_exc:
        print(
            "PCR_TURN_RECOVERY_FAILED="
            f"{type(reset_exc).__name__}: {reset_exc}",
            file=sys.stderr,
        )
        raise exc from reset_exc


def _run_chat(conversation_id: uuid.UUID) -> None:
    """Run the lightweight terminal UI over the authoritative percept path.

    Starting an interactive chat is a clean execution boundary.  Replaceable
    scheduler/worker state from an earlier CLI process is cleared automatically
    before the first prompt, so Ctrl+C or a killed process cannot strand the
    single LLM reservation across sessions.  Canonical memory and JSON artifacts
    are preserved.

    A failed turn is also an execution boundary. The error remains durably
    inspectable in artifacts/events, but replaceable scheduler state is reset so
    one malformed model output or worker failure cannot terminate the REPL or
    poison the next prompt.
    """

    with db.get_connection() as conn:
        from persistent_cognition.advisory_lock import scheduler_ownership
        from persistent_cognition.attention_store import DEFAULT_SCHEDULER_KEY

        # Hold single-owner authority over the default scheduler namespace for the
        # whole interactive session, so resetting replaceable execution state here
        # cannot race an always-on service draining the same namespace.
        try:
            ownership = scheduler_ownership(conn, DEFAULT_SCHEDULER_KEY)
            ownership.__enter__()
        except RuntimeError:
            print(
                "Another owner is already running the default scheduler "
                "(service or chat). Stop it before starting an interactive chat.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        try:
            reset_chat_execution_state(conn)

            print("Persistent Cognition")
            print(f"conversation_id: {conversation_id}")
            print("Type /exit or /quit to stop.")
            incomplete = artifact_journal.latest_interaction_id(complete=False)
            if incomplete is not None:
                print(
                    "Recoverable incomplete interaction retained in artifacts: "
                    f"{incomplete} (optional: `pcr recover --latest`)."
                )

            while True:
                try:
                    user_text = input("\nYou > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if user_text.lower() in _EXIT_COMMANDS:
                    break
                if not user_text:
                    continue

                try:
                    response = _handle_with_admission_diagnostics(
                        conn,
                        user_text,
                        conversation_id,
                    )
                except KeyboardInterrupt as exc:
                    _reset_failed_turn(conn, exc)
                    print("\nTurn interrupted; execution state reset.", file=sys.stderr)
                    continue
                except Exception as exc:
                    traceback.print_exc()
                    _reset_failed_turn(conn, exc)
                    latest = artifact_journal.latest_interaction_id(complete=False)
                    suffix = f" interaction_id={latest}" if latest is not None else ""
                    print(
                        "PCR_TURN_FAILED="
                        f"{type(exc).__name__}: {exc}{suffix}\n"
                        "Execution state reset; chat remains available.",
                        file=sys.stderr,
                    )
                    continue

                if response is not None:
                    print(f"\nPersistent Cognition > {response}")
        finally:
            ownership.__exit__(None, None, None)


def main() -> None:
    _configure_utf8_streams()
    parser = argparse.ArgumentParser(prog="pcr")
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "chat",
            "inspect",
            "verify",
            "audit",
            "recover",
            "restore-events",
            "blob-verify",
            "memory-fact",
            "sign",
            "verify-signature",
        ),
        default="chat",
        help=(
            "Run chat (default), inspect/verify/audit artifact chains, resume an "
            "incomplete interaction, restore canonical events from artifacts, "
            "verify one content-addressed blob, show one subject/property's "
            "durable belief history, or sign/verify-signature a finalized "
            "journal head."
        ),
    )
    parser.add_argument(
        "--once",
        help="Handle a single percept non-interactively and print the response.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest applicable artifact interaction.",
    )
    parser.add_argument(
        "--interaction-id",
        type=uuid.UUID,
        help="Specific artifact interaction id for inspect/verify/audit/recover/sign/verify-signature.",
    )
    parser.add_argument(
        "--conversation-id",
        type=uuid.UUID,
        help="Conversation id to use/resume for chat/--once.",
    )
    parser.add_argument(
        "--digest",
        help="Content digest (sha256:<hex>) to check with blob-verify.",
    )
    parser.add_argument(
        "--subject",
        help="Subject reference to look up with memory-fact, e.g. 'person:mike'.",
    )
    parser.add_argument(
        "--property",
        dest="property_name",
        help="Property reference to look up with memory-fact, e.g. 'preferred_drink'.",
    )
    args = parser.parse_args()

    if args.command in {"inspect", "verify", "audit", "recover", "restore-events", "sign", "verify-signature"}:
        if args.once is not None:
            parser.error("--once is only valid with chat")
        if args.conversation_id is not None:
            parser.error("--conversation-id is only valid with chat")
        if args.digest is not None:
            parser.error("--digest is only valid with blob-verify")
        if args.subject is not None or args.property_name is not None:
            parser.error("--subject/--property are only valid with memory-fact")
        if args.command == "inspect":
            _run_inspect(args.interaction_id, latest=args.latest)
            return
        if args.command == "verify":
            _run_verify(args.interaction_id, latest=args.latest)
            return
        if args.command == "audit":
            _run_audit(args.interaction_id, latest=args.latest)
            return
        if args.command == "recover":
            _run_recover(args.interaction_id, latest=args.latest)
            return
        if args.command == "sign":
            _run_sign(args.interaction_id, latest=args.latest)
            return
        if args.command == "verify-signature":
            _run_verify_signature(args.interaction_id, latest=args.latest)
            return
        if args.interaction_id is not None or args.latest:
            parser.error("restore-events takes no interaction selector")
        _run_restore_events()
        return

    if args.command == "blob-verify":
        if args.once is not None:
            parser.error("--once is only valid with chat")
        if args.conversation_id is not None:
            parser.error("--conversation-id is only valid with chat")
        if args.interaction_id is not None or args.latest:
            parser.error("blob-verify takes --digest, not an interaction selector")
        if args.subject is not None or args.property_name is not None:
            parser.error("--subject/--property are only valid with memory-fact")
        if args.digest is None:
            parser.error("blob-verify requires --digest")
        _run_blob_verify(args.digest)
        return

    if args.command == "memory-fact":
        if args.once is not None:
            parser.error("--once is only valid with chat")
        if args.conversation_id is not None:
            parser.error("--conversation-id is only valid with chat")
        if args.interaction_id is not None or args.latest:
            parser.error("memory-fact takes --subject/--property, not an interaction selector")
        if args.digest is not None:
            parser.error("--digest is only valid with blob-verify")
        if args.subject is None or args.property_name is None:
            parser.error("memory-fact requires --subject and --property")
        _run_memory_fact(args.subject, args.property_name)
        return

    if args.interaction_id is not None or args.latest:
        parser.error("--interaction-id/--latest are not valid with chat")
    if args.digest is not None:
        parser.error("--digest is only valid with blob-verify")
    if args.subject is not None or args.property_name is not None:
        parser.error("--subject/--property are only valid with memory-fact")
    conversation_id = args.conversation_id or uuid.uuid4()
    if args.once is not None:
        with db.get_connection() as conn:
            response = _handle_with_admission_diagnostics(conn, args.once, conversation_id)
        if response is not None:
            print(response)
        return
    _run_chat(conversation_id)


if __name__ == "__main__":
    main()
