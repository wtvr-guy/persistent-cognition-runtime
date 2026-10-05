# Persistent Cognition Memory Kernel v0.2

> **Historical milestone record.** This freezes the v0.2 memory kernel as accepted at
> that milestone. Later milestones supersede parts of it: deterministic derived
> associations now exist in
> [`association_projection.py`](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/src/jit_agent/association_projection.py), and v0.5
> added specificity-aware PostgreSQL candidate routing plus support-aware evidence
> admission. Current memory authority is
> [`LOSSLESS_PROGRESSIVE_MEMORY.md`](LOSSLESS_PROGRESSIVE_MEMORY.md) and
> [`COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md). Retained as milestone
> evidence, not as current architecture.

This branch adds the first deterministic Memory Kernel **beside** the existing MVP Retrieval Service. It does not replace the working MVP.

## Purpose

The kernel establishes one architectural invariant:

> **What happened is authoritative; how Persistent Cognition currently remembers it is disposable.**

The existing append-only `events` table remains the source of truth. The v0.2 kernel adds deterministic cue-based recall, versioned derived projections, tamper-evident integrity metadata, complete projection rebuilds, retrieval traces, and a synthetic-life benchmark.

No LLM is required for any Memory Kernel v0.2 operation.

## Components

- `memory_kernel.py` — pure deterministic cue scoring and bounded evidence packets.
- `memory_integrity.py` — SHA-256 hash-chain generation and verification over canonical event evidence.
- `memory_projection.py` — versioned, reproducible projections; v0.2 ships a small lexical projection.
- `postgres_memory_kernel.py` — PostgreSQL adapter for rebuilding projections/integrity and retrieving bounded candidate evidence.
- `synthetic_benchmark.py` — evaluator for a fictional persona whose oracle truth is kept separate from the event stream.
- `benchmarks/jordan_vale_v1.json` — the initial artificial lifetime and evidence-retrieval test set.

## Deterministic recall contract

For a fixed evidence set, cue state, and policy version:

```text
same evidence + same cues + same policy = same MemoryPacket
```

A `MemoryPacket` includes both the selected source events and an audit trace containing score components and the reasons each candidate activated.

The v0.2 policy deliberately remains simple. It combines lexical, explicit-entity, temporal, and conversation cues. It is a baseline socket for later episodic/associative mechanisms, not the final human-inspired recall algorithm.

## Rebuildability

The new PostgreSQL tables are derived state:

- `event_integrity`
- `memory_projection_entries`
- `memory_projection_runs`

`rebuild()` deletes and regenerates the current derived structures from `events`. It never updates or deletes authoritative events.

Run against the configured database:

```bash
python scripts/rebuild_memory_kernel.py
```

## Synthetic-life benchmark

The Jordan Vale benchmark deliberately includes changing preferences, two people with the same first name, stale facts, contradictory sources, weak lexical overlap, temporal questions, exact identifiers, and an unanswerable question.

Run it without PostgreSQL or an LLM:

```bash
python -m jit_agent.synthetic_benchmark
```

Current deterministic-cue baseline:

```text
questions: 15/18
question_success_rate: 0.833
evidence_recall: 0.833
mean_reciprocal_rank: 0.941
unknown_abstention_rate: 1.000
```

The three intentional baseline failures are useful research targets rather than hidden defects:

1. retrieving a superseded historical coffee preference from a change-oriented question;
2. recovering `security deposit returned` from the paraphrase `recover the money from the old apartment`;
3. mapping the general concept `vehicle` to a specific Toyota event with no shared vocabulary.

Those cases are intended to drive v0.3 associative/semantic recall work.

## What v0.2 intentionally does not add

- embeddings or pgvector;
- a graph database;
- LLM-based memory extraction;
- autonomous consolidation;
- an episodic event-segmentation algorithm;
- mobile/SQLite portability;
- multi-agent orchestration changes.

Those remain future layers. The v0.2 goal is to make the evidence/projection/recall boundary explicit, testable, deterministic, and rebuildable before adding more sophisticated recall mechanisms.
