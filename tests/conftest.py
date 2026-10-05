"""Pytest database setup for an isolated, disposable PostgreSQL test store.

The test suite is redirected before application modules are imported so the
long-lived development database cannot be selected through the normal `.env`.
A database-name guard adds a second line of defense against destructive test
setup, and every test begins from an empty derived/authoritative store.

Each test also receives its own independent artifact root. Subprocesses inherit
that environment value, so restart/cross-process tests exercise real filesystem
artifacts without leaking immutable records into later test cases.
"""
from __future__ import annotations

import os
import pathlib
import platform

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://persistent_cognition_app@localhost:5432/persistent_cognition_test",
)

import pytest
import httpx

from persistent_cognition import db

SCHEMA_PATH = pathlib.Path(__file__).resolve().parent.parent / "schema.sql"


@pytest.fixture
def consented_mock_ollama(_isolated_artifact_root):
    """Transport fixtures explicitly authorize their reserved .test destination.

    This does not bypass the production guard or grant real remote endpoints.
    Tests using this fixture replace the HTTP transport with an in-memory fake.
    """
    from persistent_cognition.content_digest import content_digest
    from persistent_cognition.network_consent import NetworkPurpose, consent_proposal, grant_consent
    purpose = NetworkPurpose.MODEL
    url = "http://ollama.test"
    grant_consent(url, purpose, accepted_digest=content_digest(consent_proposal(url, purpose)))


def pytest_sessionstart(session) -> None:
    """Fail closed when an explicitly requested real-machine gate is unavailable."""

    require_ollama = os.environ.get("REQUIRE_OLLAMA_ACCEPTANCE") == "1"
    require_v07_local = os.environ.get("REQUIRE_V07_LOCAL_ACCEPTANCE") == "1"
    if require_v07_local and platform.system() != "Windows":
        raise pytest.UsageError(
            "v0.7 local acceptance must run on the intended native Windows host"
        )
    if require_ollama or require_v07_local:
        try:
            with httpx.Client(trust_env=False, timeout=2.0) as client:
                response = client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise pytest.UsageError(
                "Ollama acceptance was required but localhost:11434 is unavailable"
            ) from exc


def _require_disposable_database(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        database_name = str(cur.fetchone()[0])
    lowered = database_name.casefold()
    if "test" not in lowered and "benchmark" not in lowered:
        raise RuntimeError(
            "Refusing destructive pytest setup against database "
            f"{database_name!r}; TEST_DATABASE_URL must select a dedicated "
            "database whose name contains 'test' or 'benchmark'."
        )
    return database_name


@pytest.fixture(scope="session", autouse=True)
def _prepare_test_database():
    """Apply the idempotent schema once after verifying the database target."""
    conn = db.get_connection()
    try:
        _require_disposable_database(conn)
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        conn.close()
    yield


@pytest.fixture(autouse=True)
def _isolated_artifact_root(tmp_path, monkeypatch):
    """Give each test and its child processes an independent immutable journal."""

    monkeypatch.setenv(
        "PCR_ARTIFACT_ROOT",
        str(tmp_path / "persistent_cognition-artifacts"),
    )
    yield


@pytest.fixture(autouse=True)
def _reset_test_database(_prepare_test_database):
    """Give every test a clean authoritative ledger and derived-memory state."""
    conn = db.get_connection()
    try:
        _require_disposable_database(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE TABLE
                    cognitive_heads,
                    attention_interactions,
                    attention_worker_results,
                    attention_worker_checkpoints,
                    attention_worker_claims,
                    attention_worker_claim_observations,
                    attention_worker_steps,
                    attention_resource_reservations,
                    attention_preemption_events,
                    attention_scheduling_epochs,
                    attention_resource_observations,
                    attention_assignments,
                    attention_task_transitions,
                    attention_scheduler_state,
                    attention_tasks,
                    attention_execution_resources,
                    memory_association_entries,
                    memory_projection_entries,
                    memory_projection_runs,
                    event_integrity,
                    events,
                    conversations
                RESTART IDENTITY CASCADE
                """
            )
            cur.execute("ALTER SEQUENCE attention_task_created_seq RESTART WITH 1")
        conn.commit()
    finally:
        conn.close()
    yield
