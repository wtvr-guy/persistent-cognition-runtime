"""Scale/noise test: seed hundreds-to-thousands of unrelated events, plant a
target fact among them, then run one real end-to-end interaction (real
Ollama + real Postgres) and report:

- whether retrieval found the source event;
- how many candidate items were returned;
- whether the model's final answer contains the correct fact;
- latency of the interaction;
- approximate context size handed to the LLM (chars across retrieved items).

This is an empirical probe, not a pytest test -- it's meant to be run and
read, not asserted on in CI.

Run with: uv run python scripts/scale_test.py --count 500
"""
from __future__ import annotations

import argparse
import random
import time
import uuid

from prometheist import db, event_store
from prometheist.percept_response_runtime import handle_percept_in_worker_processes
from prometheist.models import EventType

_SUBJECTS = ["my neighbor", "the team", "our cat", "the intern", "my sister", "the vendor"]
_VERBS = ["mentioned", "forgot about", "asked about", "complained about", "joked about"]
_TOPICS = [
    "the parking situation",
    "a broken printer",
    "weekend plans",
    "a coffee order",
    "the office thermostat",
    "a late delivery",
    "a podcast episode",
    "the quarterly report",
    "a hiking trail",
    "a recipe for soup",
    "the gym schedule",
    "a flight delay",
]


def _noise_sentence(rng: random.Random) -> str:
    return f"{rng.choice(_SUBJECTS).capitalize()} {rng.choice(_VERBS)} {rng.choice(_TOPICS)}."


def _seed_noise(conn, conversation_id: uuid.UUID, count: int, rng: random.Random) -> None:
    correlation_id = uuid.uuid4()
    for _ in range(count):
        text = _noise_sentence(rng)
        event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            event_type=EventType.USER_PROMPT,
            source="user",
            payload={"text": text},
            payload_text=text,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="number of unrelated noise events to seed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fact = uuid.uuid4().hex[:8].upper()
    fact_sentence = f"The codename for Project Kestrel is {fact}."
    question = "What codename did I give Project Kestrel?"

    with db.get_connection() as conn:
        conversation_id = event_store.start_conversation(conn)

        half = args.count // 2
        _seed_noise(conn, conversation_id, half, rng)
        event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=uuid.uuid4(),
            event_type=EventType.USER_PROMPT,
            source="user",
            payload={"text": fact_sentence},
            payload_text=fact_sentence,
        )
        _seed_noise(conn, conversation_id, args.count - half, rng)

        print(f"seeded {args.count} noise events + 1 target fact in conversation {conversation_id}")

        t0 = time.monotonic()
        answer = handle_percept_in_worker_processes(conn, question, conversation_id)
        elapsed = time.monotonic() - t0

        events = event_store.get_events_by_conversation(conn, conversation_id)
        retrieval_results = [e for e in events if e.event_type == EventType.MEMORY_PACKET]
        last_result = retrieval_results[-1] if retrieval_results else None
        items = last_result.payload.get("packet", {}).get("items", []) if last_result else []
        context_chars = sum(len(item.get("content", "")) for item in items)
        found_target = any(fact in item.get("content", "") for item in items)

        print(f"total events in conversation: {len(events)}")
        print(f"retrieval items returned: {len(items)}")
        print(f"target event among retrieved items: {found_target}")
        print(f"context size handed to LLM (chars): {context_chars}")
        print(f"interaction latency: {elapsed:.2f}s")
        print(f"target fact: {fact}")
        print(f"final answer: {answer!r}")
        print(f"answer contains target fact: {answer is not None and fact in answer}")


if __name__ == "__main__":
    main()
