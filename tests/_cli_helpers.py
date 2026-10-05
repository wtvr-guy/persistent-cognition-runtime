"""Shared helpers for subprocess-based CLI acceptance tests."""
from __future__ import annotations

import subprocess
import sys
import uuid

import httpx


def ollama_available() -> bool:
    try:
        with httpx.Client(trust_env=False, timeout=2.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
        return True
    except (httpx.HTTPError, OSError):
        return False


def print_transcript(value: str) -> None:
    print(value, flush=True)


def run_once(prompt: str, conversation_id: uuid.UUID, timeout: int = 900) -> str:
    """Runs the CLI's --once mode as a brand-new process with zero shared
    in-memory state, and returns its printed response.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prometheist.cli",
            "--once",
            prompt,
            "--conversation-id",
            str(conversation_id),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()
