# Persistent Cognition Memory Kernel v0.3 — Associative Recall Experiment

> **Historical milestone record.** This freezes the v0.3 associative-recall experiment
> and its deliberately curated fixture. Its "what comes next" section has since been
> acted on: deterministic derived association extraction exists in
> [`association_projection.py`](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/src/jit_agent/association_projection.py) and is
> measured by `derived_associative_benchmark.py`, and v0.5 accepted the
> specificity-aware candidate router on held-out scale corpora. Retained as milestone
> evidence, not as current architecture.

This branch tests one narrow hypothesis:

> Can a small deterministic association graph recover evidence that v0.2 lexical/entity/temporal cues miss, without weakening boundedness, provenance, determinism, or abstention?

It is an experiment on top of Memory Kernel v0.2, not a replacement for the authoritative event ledger.

## Invariant

Canonical events remain evidence. Associations are **derived routing hints**.

```text
association -> candidate activation -> canonical source event -> evidence packet
```

An association must never become a substitute for the event(s) that justify it.

## Association shape

```text
Association
├── association_id
├── source_kind: TERM | EVENT
├── source
├── target_kind: TERM | EVENT
├── target
├── relationship
├── strength: (0, 1]
├── provenance_event_ids
└── required_cue_terms
```

`required_cue_terms` gates relationships whose relevance depends on the present context. For example, a `PREVIOUS_STATE` edge should activate an old preference when the query asks about the state **before** a change, but should not boost that stale preference when the query asks what is true **now**.

## Bounded spreading activation

Present query/entity terms begin as activated term nodes. Events that already clear the v0.2 minimum score become activated event nodes. The algorithm then propagates activation over the association graph for a fixed number of hops.

```text
Present cues
    |
    +--> directly activated terms
    +--> v0.2 directly activated events
                  |
                  v
          association edges
                  |
                  v
      neighboring terms/events
                  |
                  v
       canonical evidence ranking
```

The default experiment uses at most two hops and deterministic decay. There is no graph database and no LLM in the retrieval loop.

## Control benchmark

The v0.2 retrieval algorithm remains unchanged and continues to provide the control baseline:

```text
15/18 successful questions
0.833 evidence recall
0.941 MRR
1.000 unknown abstention
```

The Jordan Vale fixture was corrected during v0.3 development so the unknown-fact case represents **open-world absence** rather than an explicit null field. The evaluator now asks `What is Jordan's blood type?`, while blood type is represented nowhere in the oracle/profile or event corpus. This correction does not change the v0.2 retrieval result: the control still reproduces 15/18.

The three associative-recall baseline failures remain:

1. historical-state recall for the earlier latte preference;
2. resolving the returned security deposit from the paraphrase about recovering money from the old apartment;
3. mapping the concept `vehicle` to the Toyota Corolla purchase event.

## Association fixture

`benchmarks/jordan_vale_associations_v1.json` contains three explicit relationships used to validate the mechanism:

- `PREVIOUS_STATE`: the coffee-change event can activate the earlier latte event when a past-state cue is present;
- `RESOLVED_BY`: the outstanding-deposit event can activate the later returned-deposit event when a resolution cue is present;
- `CONCEPT_INSTANCE`: the term `vehicle` can activate the Toyota Corolla purchase event.

This fixture is **curated**. It demonstrates that the retrieval mechanism can use associations correctly. It does not demonstrate that Persistent Cognition can yet derive those associations automatically from arbitrary lifetime data.

That distinction is deliberate. Extraction and retrieval are separate research problems and should not be entangled before the retrieval mechanism earns its place.

## Comparative benchmark

Run:

```powershell
uv run python -m jit_agent.associative_benchmark
```

The command prints the v0.2 algorithmic control and the v0.3 associative result side by side.

The regression test requires:

```text
v0.2 baseline:      15/18
v0.3 associative:   18/18
unknown abstention: 1.000 in both
```

This 18/18 result should be interpreted narrowly: the curated association graph repairs the three known benchmark failures while the canonical event stream remains unchanged. It is not evidence of general semantic-memory performance.

## What comes next if this passes

The next experiment should **not** add more hand-authored Jordan associations. That would overfit the benchmark.

Instead:

1. generate a second fictional persona and held-out questions;
2. define deterministic rules for deriving a limited set of association types from event metadata/content;
3. build those associations as versioned derived projections with provenance;
4. evaluate on the held-out persona without changing retrieval weights or hand-authoring answer-specific edges;
5. only then compare this approach against embeddings/vector retrieval.

The purpose of v0.3 is to prove that deterministic associative activation is a useful retrieval primitive before spending complexity on automatic association extraction, vector infrastructure, or a larger multi-agent architecture.
