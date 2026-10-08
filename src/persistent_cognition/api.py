"""Supported embedding surface v1; PostgreSQL remains the operational backend.

Configuration is process-scoped. Applications establish worker environment once
at startup, then use separate database connections per thread/session. Multiple
security principals or configurations require separate processes and stores.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from uuid import UUID

import psycopg

from persistent_cognition.advisory_lock import scheduler_ownership
from persistent_cognition.capability_registry import CapabilityRegistry, DEFAULT_REGISTRY
from persistent_cognition.percept_intake import ingest_percept, install_source_policy
from persistent_cognition.percept_service import PerceptService
from persistent_cognition.percept_triage import SourcePolicy
from persistent_cognition.runtime_settings import AppSettings, ModelSelection
from persistent_cognition.trusted_executors import register_trusted_executor

API_VERSION = "persistent-cognition-api/v1"
__all__ = ["API_VERSION", "AppSettings", "ModelSelection", "SourcePolicy", "PerceptService",
           "CapabilityRegistry", "connect", "initialize_schema", "worker_environment", "respond",
           "tick", "ingest_percept", "install_source_policy", "register_trusted_executor"]


def connect(database_url: str) -> psycopg.Connection:
    from persistent_cognition.network_consent import require_database_destination
    require_database_destination(database_url)
    return psycopg.connect(database_url)


def initialize_schema(conn: psycopg.Connection) -> None:
    """Explicit provisioning, including from an installed wheel; never drops data."""
    conn.execute(files("persistent_cognition").joinpath("schema.sql").read_text(encoding="utf-8"))
    conn.commit()


def worker_environment(settings: AppSettings, *, database_url: str, artifact_root: Path) -> dict[str, str]:
    """Return child-inheritable configuration; the caller installs it at startup."""
    from persistent_cognition.network_consent import require_database_destination
    from persistent_cognition.private_storage import private_directory
    private_directory(artifact_root)
    require_database_destination(database_url, root=artifact_root)
    return {**settings.worker_environment(), "DATABASE_URL": database_url,
            "PCR_ARTIFACT_ROOT": str(artifact_root.absolute())}


def respond(conn: psycopg.Connection, text: str, conversation_id: UUID, *,
            scheduler_key: str = "embedding-chat", registry: CapabilityRegistry = DEFAULT_REGISTRY,
            worker_command: tuple[str, ...] | list[str] | None = None, **runtime_options) -> str | None:
    from persistent_cognition.percept_response_runtime import handle_percept_in_worker_processes
    with scheduler_ownership(conn, scheduler_key):
        return handle_percept_in_worker_processes(conn, text, conversation_id,
            scheduler_key=scheduler_key, registry=registry, worker_command=worker_command, **runtime_options)


def tick(conn: psycopg.Connection, *, scheduler_key: str = "automation", **options) -> dict:
    from persistent_cognition.percept_cli import tick as run_tick
    return run_tick(conn, scheduler_key=scheduler_key, **options)
