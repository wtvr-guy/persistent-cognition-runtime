# System Determinism

**Status:** constitutional architecture deep dive.  
**Constitutional authority:** implements Articles 15, 16, and 35 of [`../../CONSTITUTION.md`](../../CONSTITUTION.md).

Persistent Cognition uses probabilistic models as disposable semantic compute inside a deterministic application envelope. The system does not require every physical execution detail to be deterministic; it requires every durable control-plane decision that can be made deterministically to be owned, recorded, and replayable by ordinary software.

## Core invariant

> **Given the same authoritative durable state, the same authoritative external/resource observations, and the same policy versions, Persistent Cognition must reconstruct the same durable system decision.**

This is the determinism boundary.

The operating system may schedule threads differently. A local model may sample different language if sampling is enabled. Network timing may vary. Those facts do not authorize a race, hidden model preference, or process-local state to decide which durable task owns attention, which resource reservation wins, which capability is legal, which event is authoritative, or which side effect is allowed.

## What ordinary software owns

Where the decision can be represented mechanically, application policy owns it. This includes at least:

- stable/deterministic identifiers where inputs permit them;
- authoritative event/task sequence numbers and total-order tie breakers;
- priority derivation and service-guarantee promotion;
- dependency graphs and dependency closure;
- capability IDs, availability, permissions, dependencies, and execution ordering;
- scheduling epochs, assignments, reservations, and preemption selection;
- resource-safety formulas and policy versions;
- retention/admission/deletion authority;
- retry, idempotency, lease, checkpoint, and terminal-result rules;
- schema validation and bounds checking;
- provenance relationships;
- test fixture generation and deterministic oracles where exact verification is possible.

A model may help interpret an ambiguous semantic question, but its free-form preference must not become the hidden owner of deterministic system policy.

## Authoritative inputs

A deterministic decision may legitimately depend on changing reality. When it does, the changing input becomes part of the decision record.

Examples include:

- current runnable-task state;
- persisted resource observations;
- policy/configuration versions;
- capability registry revision;
- canonical event/history state;
- explicitly admitted external observations;
- model result when a semantic classification is deliberately part of the contract.

If a measured value materially changes a durable decision, Persistent Cognition must retain enough information to explain which value was used. “The machine was busy at the time” is not sufficient if the exact resource observation decided whether a worker was launched.

## Ordering and identity

Persistent Cognition should prefer authoritative logical ordering over wall-clock inference when both are available.

Examples:

- event/task sequence is authoritative for total ordering inside a persistence namespace;
- deterministic tie breakers complete partial orderings;
- timestamps remain valuable observations/provenance but must not replace a stable sequence when race-free ordering is required;
- identifiers should be application-owned and derived deterministically from canonical inputs when idempotency/replay requires stable identity.

Random UUIDs may still be valid where stable replay identity is not required, but they must not become an accidental source of policy ordering when a deterministic total order is needed.

## Deterministic scheduling epochs

The Attention Fabric is the clearest example of the rule:

```text
authoritative runnable tasks
        +
authoritative resource snapshot
        +
policy versions
        |
        v
deterministic ranking/admission/preemption
        |
        v
atomic committed epoch + assignments + reservations
        |
        v
workers may execute only committed authority
```

Workers do not race to “grab the next task.” A worker race may affect which CPU core executes code first, but not which durable assignment exists.

## Capability determinism

Model-selected capability indices represent requirements, not an execution schedule.

Persistent Cognition resolves them through application-owned metadata:

```text
bounded model selection
    -> canonical capability IDs
    -> dependency closure
    -> reject missing dependency/cycle
    -> deterministic ready-item ordering
    -> persist/execute
```

The model does not write executor names, dependencies, durable IDs, or execution order.

## Deterministic evidence responses

Article 15 applies to evidence lookup and display as well as the executive control
plane. First use practical deterministic code for the operation. A fresh model
may interpret a natural-language request, but that interpretation is not a reason
to send already retrieved records through another model to rediscover their values.

For a trusted structured source/field binding, code performs direct field access.
For factual recall over prose, code displays the admitted source records with
attribution and exact quoted content. This avoids inventing a fact schema or a
phrase-specific English parser. It also preserves conflicts rather than assigning
truth to whichever record a small model happens to select. Evidence display does
not promise semantic relevance or complete answerability; retrieval and intent
classification remain separate obligations.

The source renderer is model-free, versioned and replayable from `RESPONSE_RENDER`
artifacts. An invalid binding fails without a model fallback. Requested free-form
explanations and raw semantic extraction from unstructured prose may still need a
model when no suitable deterministic contract exists.

## Memory determinism

Memory retrieval may include model-assisted semantic mechanisms in future versions, but canonical evidence identity and provenance remain system-owned. The deterministic baseline should be used wherever it can satisfy the information need.

A retrieval result should make clear which parts were produced by:

- exact/structured filtering;
- deterministic lexical/indexed routing;
- deterministic association traversal;
- model/embedding-assisted candidate generation or reranking;
- evidence-admission policy.

A probabilistic candidate generator may propose evidence. It does not become the authoritative source event.

## Model-output boundary

When a model must participate in system control, the system should constrain the model to the smallest mechanically verifiable semantic output that solves the problem.

Preferred representation:

```text
enum / boolean / bounded integer / application-owned ID or index
    before
verified extractive selection
    before
free-form natural language
```

Validation failure is an explicit failure, not permission to infer what the model “probably meant.”

## Fail-closed authority

The deterministic envelope must not fabricate authority when its required inputs are invalid or insufficient.

Examples include:

- invalid or unrecognized model-control output;
- stale/missing policy revisions;
- incomplete dependency authority;
- contradictory or malformed durable state;
- missing provenance needed for an irreversible decision.

The appropriate outcome may be rejection, abstention, waiting, retry under an explicit policy, or reconciliation. It is not an untracked guess.

## No race-based authority

The following patterns are unconstitutional:

- multiple workers independently selecting the next queued task and the database winner becoming policy;
- first-completing model call deciding system priority without an explicit policy that makes completion time an authoritative input;
- first process to acquire an ungoverned lock becoming the semantic owner of a durable decision;
- model-generated textual task/capability names being treated as canonical IDs;
- unordered database results becoming persistent ordering accidentally;
- wall-clock arrival replacing an existing authoritative sequence without an explicit design decision.

Concurrency is allowed. Race-based authority is not.

## Replay and reconstruction

A durable decision should be reconstructible from:

1. canonical durable state at the decision boundary;
2. referenced authoritative observations;
3. exact policy/configuration versions;
4. any deliberately admitted model/semantic result;
5. the deterministic algorithm version.

Replay need not reproduce incidental process timing. It must reproduce the durable decision.

## Policy changes

Deterministic behavior is versioned behavior. If a policy changes, Persistent Cognition should preserve enough version/provenance information to explain why the same underlying evidence produced a different decision under a later policy.

A policy migration must not rewrite historical decisions as if the newer policy had always been active.

## Testing requirements

Constitutional determinism should be tested through:

- repeated identical-input runs;
- restart/reconstruction equivalence;
- alternate insertion/worker timing that must yield the same durable outcome;
- deterministic total-order tests for ties;
- stale-policy/snapshot rejection;
- transaction rollback tests proving partial authority is not exposed;
- process-destruction tests proving decisions survive loss of process-local state;
- explicit assertions that unordered/racing worker behavior cannot choose durable authority.

See [`../engineering/TESTING_AND_ACCEPTANCE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/engineering/TESTING_AND_ACCEPTANCE.md).

## Relationship to probabilistic intelligence

This rule does not require Persistent Cognition to pretend semantic uncertainty is deterministic. A model may produce different interpretations when the architecture intentionally asks it to reason. The constitutional requirement is that the uncertainty is represented as an explicit input/result with provenance and bounded authority.

Persistent Cognition's design target is therefore:

> **Probabilistic cognition inside deterministic governance.**

That separation allows models to remain useful, replaceable, and improvable without making persistent system behavior unknowable.
