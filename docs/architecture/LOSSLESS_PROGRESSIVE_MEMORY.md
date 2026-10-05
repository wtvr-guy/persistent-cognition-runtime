# Lossless Progressive Memory and Bounded Recall

**Status:** constitutional architecture deep dive, rebaselined 2026-09-03.

**Constitutional status:** primary architecture authority for Articles 9, 10, 11, 23, and 24 of [`../../CONSTITUTION.md`](../../CONSTITUTION.md). This document explains those constitutional rules in depth and is subordinate to the Constitution where wording conflicts.

This document freezes the memory-scaling constraints that future Persistent Cognition milestones must preserve. It supplements `COGNITIVE_ARCHITECTURE.md`, `MEMORY_KERNEL.md`, and `ASSOCIATIVE_MEMORY.md` and supersedes any older wording that would permit an already-admitted durable memory to be replaced by a summary, abstraction, embedding, aggregation, or other lossy derivative. [`SEMANTIC_FACT_PROVENANCE.md`](SEMANTIC_FACT_PROVENANCE.md) implements the append-only, non-destructive supersession model this document requires for one concrete kind of derived structure: a durable belief about a subject/property value.

## Core rule

> **Persistent Cognition keeps durable memory lossless. It may reduce the active working set, but it may not reduce a persisted memory into a substitute representation.**

Storage is treated as comparatively abundant. The scarce resources are attention, active RAM, CPU/GPU/VRAM, inference time, I/O bandwidth, and model context. Persistent Cognition therefore optimizes the movement and computation performed over memory rather than the existence of memory itself.

## Durable-memory admission boundary

External input may be ephemeral before a retention decision admits it into durable memory.

```text
external input
      |
      v
ephemeral observation
      |
retention/admission decision
   /                \
not admitted         admitted
                         |
                         v
                canonical durable memory
```

Before admission, deterministic retention policy may discard data that never became durable memory. After admission, ordinary retention, compaction, indexing, or storage-pressure policy must not replace or delete the canonical memory. Explicit policy-governed erasure, if later supported, is a separate governance operation and must never be confused with automatic compression or retention policy.

## Source authority

Canonical durable memories are authoritative evidence.

A durable memory must preserve its exact admitted content and provenance. Its logical identity must remain stable when the bytes move between storage devices or backends.

Derived structures may include:

- full-text indexes;
- vector embeddings used only as retrieval coordinates;
- entity mentions and identifiers;
- association edges;
- temporal or causal links;
- coreference candidates;
- situation/topic membership;
- salience and activation metadata;
- access statistics;
- storage manifests and availability metadata.

These structures are routing and navigation aids. They are not memories and may never substitute for the exact source evidence they reference.

The long-lived invariant remains:

```text
association/index -> candidate activation -> canonical source memory -> evidence
```

## Append-only history and correction

Canonical historical events are append-only under ordinary system operation.

If a prior observation, statement, belief, interpretation, or derived claim is later discovered to be wrong, stale, contradicted, or superseded, Persistent Cognition records the new evidence and the correction/supersession relationship. It does not edit the earlier record to make the historical state appear as though the correction had always been known.

Conceptually:

```text
original canonical event
        |
        +<---- correction / contradiction / supersession relation
        |
new provenance-bearing event
```

This preserves both current understanding and historical reconstructability: the system can distinguish “what is now believed/supported” from “what was actually present in the record when an earlier decision was made.” [`SEMANTIC_FACT_PROVENANCE.md`](SEMANTIC_FACT_PROVENANCE.md) implements this with separate immutable SemanticAssertion, SemanticEvidence, and SemanticResolution records. Assertions never overwrite one another; evidence accumulates exact provenance; and only resolution records supersede prior resolution records. Historical backfill, ambiguity, and bi-temporal knowledge reconstruction therefore remain explicit rather than being encoded as destructive fact replacement.

Explicit policy-governed erasure, if implemented, is a separate governance path and must define its audit/integrity semantics explicitly. It is not an ordinary correction mechanism.

## No memory summaries

Persistent Cognition must not create or persist summaries of durable memories as memory substitutes, retrieval substitutes, or supposedly equivalent compressed representations.

A later model invocation must not be asked to trust a prose condensation of earlier evidence when the factual answer depends on that evidence. Factual recall must ultimately dereference exact source memories.

This prohibition includes generational summarization in which one model summarizes source memories and later models reason from the summary instead of the source. Such a pipeline would introduce cumulative lossy distortion and violate the accuracy objective.

## Derived structures are disposable

Indexes and annotations should be rebuildable from canonical source memory wherever their derivation permits it.

If embeddings are deleted, they can be recomputed. If an entity linker improves, its mappings can be rebuilt. If a coreference model proves unreliable, its edges can be discarded and regenerated.

Destroying a derived index is an operational inconvenience. Destroying the only canonical source memory is actual memory loss.

Derived assertions that are not deterministically reconstructable must carry explicit provenance, derivation method/version, and non-authoritative status. They may guide retrieval, but supported factual recall must still dereference the source evidence.

## Progressive associative recall

Persistent Cognition must not interpret a memory request as a request to retrieve everything associated with a subject.

Ordinary recall is progressive:

1. activate the strongest present cues;
2. navigate a bounded associative neighborhood using inexpensive deterministic/indexed operations where possible;
3. rank candidate memory identifiers;
4. dereference a small exact evidence set;
5. invoke an LLM only when semantic interpretation or reasoning actually requires it;
6. expand another bounded layer only when the current evidence is insufficient for the requested recall depth.

A cue such as a person's name should normally activate a small set of distinctive and currently relevant exact memories rather than every recorded interaction involving that person. A request for greater detail may progressively widen the search. Exhaustive historical analysis is a different workload and is allowed to consume substantially more total compute.

## Activation, salience, and storage temperature are distinct

Persistent Cognition must not collapse these into one score.

**Salience** describes how significant or distinctive an observation or memory is under the current policy/history.

**Activation** describes how relevant a memory, entity, association, task, or situation appears to the present cognitive state.

**Storage temperature** describes physical retrieval cost and availability, not meaning.

A decades-old memory may have high salience, low current activation, and cold physical storage. A new cue may sharply raise activation without altering the source memory. Repeated use may then promote exact bytes into a faster cache without changing logical identity or content.

## Semantic identity is independent of storage location

Persistent Cognition must not make a physical volume semantically own a person, topic, or situation. One memory may participate in many associations.

Each durable memory should eventually have:

- a stable logical memory/event identifier;
- a cryptographic content hash or equivalent integrity identity;
- exact provenance;
- one or more physical storage locations;
- availability/integrity state for each replica.

A storage manifest may map the same canonical memory to internal NVMe, external SSD/HDD, removable media, NAS, cloud, or future archival backends. If a replica is unavailable, Persistent Cognition may know that the memory exists and where it lives, but must not pretend to have inspected unavailable source evidence.

## Bounded LLM context is a hard invariant

No ordinary memory mechanism may solve scale by increasing one model invocation's context in proportion to corpus size.

Every LLM invocation must eventually have a policy-bounded maximum working context. This applies to the whole invocation, not only a `MemoryPacket`. A worker may not defeat stateless-model architecture by accumulating an ever-growing hidden research transcript and replaying it into later calls.

Item-count bounds alone are insufficient. A single canonical event or structured work
result may be arbitrarily large. Model-facing views therefore enforce explicit
per-item and aggregate byte limits before transport, including rendered metadata
overhead. An oversized view fails closed; canonical evidence is not truncated or
rewritten to fit the model.

A worker that needs additional history must externalize compact structured task state and request exact evidence as needed rather than drag an unbounded transcript forward.

## Bounded ordinary compute is also a goal

Flattening context while exploding the number of model calls is not an acceptable ordinary-case solution.

For routine recall, total LLM calls and active compute should remain small as the durable corpus grows. Most navigation should rely on conventional deterministic/indexed operations such as exact identifiers, SQL, full-text search, temporal filtering, sparse association traversal, entity intersections, and other cheap routing mechanisms.

Derived-projection freshness is also ordinary compute. Steady-state freshness must
advance from durable high-water marks over the new event tail rather than materialize
the lifetime ledger for each percept. Incompatible or missing projection state may
trigger a full deterministic rebuild; that recovery path must not become routine.

Large forensic questions may legitimately create large durable workloads, but their work must be partitioned into bounded operations with exact provenance. They are exceptional analyses, not the normal recall path.

## Evidence navigation

JIT Memory should mature from a top-k retrieval service into an evidence-navigation capability.

Its target question is:

> **What is the smallest exact source-evidence set that satisfies the present information need, and which bounded association path should be explored next if that evidence is insufficient?**

The capability may use derived indexes to identify candidate evidence, but the packet consumed by factual reasoning must remain grounded in canonical source memories.

## Scaling hypothesis

Persistent Cognition's central memory-scaling hypothesis is:

> **Recall quality can remain accurate over an increasingly large lossless corpus without proportional growth in any individual LLM context window or in ordinary-case LLM inference count.**

Future memory benchmarks must therefore measure at least:

- exact source-evidence recall;
- provenance correctness;
- false-memory/unsupported-claim rate;
- maximum tokens in any LLM invocation;
- maximum memory tokens in any invocation;
- total LLM calls per recall task;
- candidate memories examined;
- association-hop depth;
- database/index operations;
- CPU time and peak RAM;
- latency;
- index/storage growth;
- archive reads and cache promotions.

Corpus-size experiments must freeze the query families while increasing corpus scale so that both the context-bloat curve and the ordinary-compute curve are visible.

## Milestone implications

### v0.7

The closure candidate implements the Attention Fabric, bounded aperture, Adaptive
Recall, incremental projection freshness, and byte-bounded model evidence. Native
acceptance on the frozen closure SHA remains required.

### v0.8

Epistemic WorkingState must remain a bounded derived projection over exact evidence.
Hypotheses and unresolved questions may guide attention but must not become canonical
truth or copied lifetime context.

### v0.9

Perception/salience may influence attention-worthiness without itself rewriting memory
or implying durable admission.

### v0.10

Retention must implement the explicit durable-admission boundary. Pre-admission
aggregation and disposable indexes must never replace admitted canonical evidence.

### v0.11

Memory generalization and episode routing must test progressive recall, activation,
exact dereferencing, interference, long-horizon navigation, and context/compute
scaling before any learned semantic route is accepted.

### v1.0

A defensible v1.0 must demonstrate lossless source-backed durable memory and bounded provenance-bearing recall whose ordinary per-invocation context and inference cost do not grow proportionally with corpus size.

## Constitutional relationship

Articles 9 and 10 govern canonical/derived memory fidelity; Article 11 governs bounded recall; Articles 23 and 24 govern epistemic separation and durable causal provenance. Explicit policy-governed erasure is governed separately by [`LOCAL_FIRST_PORTABILITY.md`](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/LOCAL_FIRST_PORTABILITY.md) and must never be implemented as ordinary compaction.
