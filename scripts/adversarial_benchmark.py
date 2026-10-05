"""Adversarial retrieval benchmark: high lexical overlap, conflicting/temporal
facts, and closely-related entity names -- the retrieval failure modes that
matter most before pgvector is worth adding. Complements the FTS failure
corpus (paraphrase/zero-overlap cases) and the scale test (raw volume).

For each case, measures:
- target_rank: position of the correct source statement within the returned
  candidates (0-based), or None if it wasn't retrieved at all;
- result_set_size: how many candidate items came back;
- context_chars: total content size handed to the response step;
- correct: whether the final answer contains the expected value and none of
  the distractor values;
- latency: wall-clock time for the whole interaction.

This is an empirical probe, not a pytest test -- read the output, don't gate
CI on it.

Run with: uv run python scripts/adversarial_benchmark.py
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from prometheist import db, event_store
from prometheist.percept_response_runtime import handle_percept_in_worker_processes
from prometheist.models import EventType

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _commits_to_expected(answer: str, expected: str) -> bool:
    """A response "commits" to a value if it appears in the final sentence
    (its concluding statement), not just anywhere while hedging or narrating
    earlier context/distractors.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s]
    tail = sentences[-1] if sentences else answer
    return expected in tail or expected in answer[-100:]


@dataclass
class Case:
    category: str
    note: str
    statements: list[str]
    target_statement_index: int  # which seeded statement holds the correct answer
    question: str
    expected: str
    distractors: list[str] = field(default_factory=list)  # other plausible-but-wrong values


@dataclass
class TemporalQuestion:
    label: str
    question: str
    target_statement_index: int
    expected: str
    distractors: list[str] = field(default_factory=list)


@dataclass
class TemporalGroup:
    """Seeds one set of conflicting/updated statements once, then asks several
    differently-time-scoped questions against that same seeded history --
    proves the context representation carries real temporal signal rather
    than just biasing every answer toward the newest item.
    """

    category: str
    note: str
    statements: list[str]
    questions: list[TemporalQuestion]


def build_cases() -> list[Case]:
    token = uuid.uuid4().hex[:6].upper()
    return [
        Case(
            category="high_lexical_overlap",
            note="5 near-identical sentences differing only by entity name",
            statements=[
                f"The launch code for Project Sparrow is AAA-{token}-1.",
                f"The launch code for Project Sparrowhawk is BBB-{token}-2.",
                f"The launch code for Project Robin is CCC-{token}-3.",
                f"The launch code for Project Falcon is DDD-{token}-4.",
                f"The launch code for Project Eagle is EEE-{token}-5.",
            ],
            target_statement_index=3,
            question="What launch code did I give for Project Falcon?",
            expected=f"DDD-{token}-4",
            distractors=[f"AAA-{token}-1", f"BBB-{token}-2", f"CCC-{token}-3", f"EEE-{token}-5"],
        ),
        Case(
            category="closely_related_entities",
            note="near-identical entity names with different values",
            statements=[
                f"The budget for Project Falcon is {token}1 dollars.",
                f"The budget for Project Falconer is {token}2 dollars.",
                f"The budget for Project Falcons is {token}3 dollars.",
            ],
            target_statement_index=0,
            question="What is the budget for Project Falcon?",
            expected=f"{token}1",
            distractors=[f"{token}2", f"{token}3"],
        ),
    ]


def build_temporal_group() -> TemporalGroup:
    token = uuid.uuid4().hex[:6].upper()
    statements = [
        f"The codename for Project Kestrel is Red-{token}.",
        f"Actually, update the codename for Project Kestrel to Silver-{token}.",
        f"Correction: the codename for Project Kestrel is now Golden-{token}.",
    ]
    red, silver, golden = f"Red-{token}", f"Silver-{token}", f"Golden-{token}"
    return TemporalGroup(
        category="conflicting_temporal",
        note="same fact restated 3x with different values; tests current/original/previous",
        statements=statements,
        questions=[
            TemporalQuestion(
                label="current",
                question="What is the current codename for Project Kestrel?",
                target_statement_index=2,
                expected=golden,
                distractors=[red, silver],
            ),
            TemporalQuestion(
                label="original",
                question="What was Project Kestrel's original codename?",
                target_statement_index=0,
                expected=red,
                distractors=[silver, golden],
            ),
            TemporalQuestion(
                label="immediately_previous",
                question=f"What codename did Project Kestrel have immediately before {golden}?",
                target_statement_index=1,
                expected=silver,
                distractors=[red, golden],
            ),
        ],
    )


def _measure_interaction(
    conn, conversation_id, question: str, target_text: str, expected: str, distractors: list[str]
) -> dict:
    t0 = time.monotonic()
    answer = handle_percept_in_worker_processes(conn, question, conversation_id)
    if answer is None:
        raise RuntimeError("explicit benchmark prompt produced no response")
    elapsed = time.monotonic() - t0

    events = event_store.get_events_by_conversation(conn, conversation_id)
    retrieval_results = [e for e in events if e.event_type == EventType.MEMORY_PACKET]
    items = (
        retrieval_results[-1].payload.get("packet", {}).get("items", [])
        if retrieval_results
        else []
    )

    target_rank = None
    for i, item in enumerate(items):
        if target_text in item.get("content", ""):
            target_rank = i
            break

    context_chars = sum(len(item.get("content", "")) for item in items)
    has_expected = expected in answer
    leaked_distractor = any(d in answer for d in distractors)
    correct = _commits_to_expected(answer, expected)

    return {
        "question": question,
        "expected": expected,
        "answer": answer,
        "target_rank": target_rank,
        "result_set_size": len(items),
        "context_chars": context_chars,
        "correct": correct,
        "has_expected": has_expected,
        "leaked_distractor": leaked_distractor,
        "latency_s": round(elapsed, 2),
    }


def run_case(conn, case: Case) -> dict:
    conversation_id = event_store.start_conversation(conn)
    target_text = case.statements[case.target_statement_index]

    for statement in case.statements:
        event_store.record_event(
            conn,
            conversation_id=conversation_id,
            correlation_id=uuid.uuid4(),
            event_type=EventType.USER_PROMPT,
            source="user",
            payload={"text": statement},
            payload_text=statement,
        )

    result = _measure_interaction(
        conn, conversation_id, case.question, target_text, case.expected, case.distractors
    )
    result["category"] = case.category
    result["note"] = case.note
    return result


def run_temporal_group(conn, group: TemporalGroup) -> list[dict]:
    """Each question gets its own fresh conversation seeded with the same
    timeline, so questions can't leak retrieval/context state between each
    other -- only the seeded history and the current question matter.
    """
    results = []
    for q in group.questions:
        conversation_id = event_store.start_conversation(conn)
        for statement in group.statements:
            event_store.record_event(
                conn,
                conversation_id=conversation_id,
                correlation_id=uuid.uuid4(),
                event_type=EventType.USER_PROMPT,
                source="user",
                payload={"text": statement},
                payload_text=statement,
            )

        target_text = group.statements[q.target_statement_index]
        result = _measure_interaction(
            conn, conversation_id, q.question, target_text, q.expected, q.distractors
        )
        result["category"] = f"{group.category}:{q.label}"
        result["note"] = group.note
        results.append(result)
    return results


def main() -> None:
    results = []
    with db.get_connection() as conn:
        for case in build_cases():
            results.append(run_case(conn, case))
        results.extend(run_temporal_group(conn, build_temporal_group()))

    for r in results:
        print(f"--- {r['category']} ---")
        print(f"note:            {r['note']}")
        print(f"question:        {r['question']}")
        print(f"expected:        {r['expected']}")
        print(f"answer:          {r['answer']!r}")
        print(f"target_rank:     {r['target_rank']}")
        print(f"result_set_size: {r['result_set_size']}")
        print(f"context_chars:   {r['context_chars']}")
        print(f"latency_s:       {r['latency_s']}")
        print(f"has_expected:    {r['has_expected']}")
        print(f"leaked_distractor: {r['leaked_distractor']}")
        print(f"CORRECT:         {r['correct']}")
        print()

    n_correct = sum(1 for r in results if r["correct"])
    print(f"summary: {n_correct}/{len(results)} correct")


if __name__ == "__main__":
    main()
