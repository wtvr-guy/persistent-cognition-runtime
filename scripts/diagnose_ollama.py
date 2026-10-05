"""Standalone diagnostic: times progressively more complex Ollama requests
against the configured model to isolate whether latency comes from plain
generation, think=false, or constrained-schema complexity.

Run with: uv run python scripts/diagnose_ollama.py
"""
from __future__ import annotations

import os
import time

import httpx
from dotenv import load_dotenv
from prometheist.contract_registry import SEMANTIC_CONTRACTS
from prometheist.network_consent import NetworkPurpose, require_destination

load_dotenv()

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")


def _call(label: str, payload: dict) -> None:
    require_destination(BASE_URL, NetworkPurpose.MODEL)
    client = httpx.Client(base_url=BASE_URL, timeout=300.0, trust_env=False)
    t0 = time.monotonic()
    response = client.post("/api/chat", json=payload)
    elapsed = time.monotonic() - t0
    response.raise_for_status()
    body = response.json()
    content = body["message"]["content"]
    print(f"--- {label} ---")
    print(f"elapsed={elapsed:.2f}s")
    print(
        f"load_duration={body.get('load_duration')}ns "
        f"prompt_eval_duration={body.get('prompt_eval_duration')}ns "
        f"eval_duration={body.get('eval_duration')}ns "
        f"prompt_eval_count={body.get('prompt_eval_count')} "
        f"eval_count={body.get('eval_count')}"
    )
    print(f"content={content!r}")
    print()


def main() -> None:
    print(f"model={MODEL} base_url={BASE_URL}\n")

    _call(
        "A: plain generation",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
        },
    )

    _call(
        "B: plain generation + think=false",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "think": False,
            "stream": False,
        },
    )

    _call(
        "C: v2 pre-cognitive work-selection schema",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "The codename for Project Oriole is 12AB34CD.",
                }
            ],
            "format": SEMANTIC_CONTRACTS["PRECOGNITIVE_USER_PROMPT_WORK"].output_schema(),
            "think": False,
            "stream": False,
        },
    )

    _call(
        "D: evidence-policy schema",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "What was the codename I gave you for Project Oriole?",
                }
            ],
            "format": SEMANTIC_CONTRACTS["V2_RESPONSE_POLICY"].output_schema(),
            "think": False,
            "stream": False,
        },
    )

    _call(
        "E: v2 final-answer envelope",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "What was the codename I gave you for Project Oriole?",
                }
            ],
            "format": SEMANTIC_CONTRACTS["FINAL_RESPONSE_V2"].output_schema(),
            "think": False,
            "stream": False,
        },
    )


if __name__ == "__main__":
    main()
