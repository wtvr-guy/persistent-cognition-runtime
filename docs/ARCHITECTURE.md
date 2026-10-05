# Architecture

## State ownership

Durable history, task state, policy versions, and execution records carry continuity. An LLM context and worker process are temporary computation. A controller may remain alive for efficiency, but restarting it must recover authoritative state.

Canonical evidence is distinct from derived interpretations. Corrections append new evidence and explicit supersession relationships; they do not silently replace prior events. Projections can be regenerated from their sources.

## Percept to disposition

1. Persist the accepted percept with deterministic identifiers, provenance, and causal links.
2. Reconstruct orientation from durable state and retrieve bounded initial evidence.
3. Invoke narrow semantic specialists where interpretation is required.
4. Persist task proposals and apply deterministic eligibility, dependency, and resource checks.
5. Execute authorized work through explicit capability contracts.
6. For required responses, assess persistent-memory sufficiency with a narrow Composer; perform bounded Adaptive Recall and fresh reassessment when needed.
7. Deliver authoritative capability results and admitted memory evidence to a fresh final responder.
8. Persist outputs, artifacts, and final or interrupted disposition.

These are responsibility boundaries, not an assertion that the source implementation has exactly eight workers. A non-response task may complete without a final responder.

## Memory boundary

A MemoryNeed specifies the semantic deficit and any explicit temporal, entity, or provenance constraints. A MemoryPacket returns admitted canonical evidence, source references, and uncertainty within a fixed budget.

Candidate generation may use lexical, entity, temporal, or association projections. Candidate activation alone is not evidence. Additional retrieval mechanisms require demonstrated failure cases and measured benefit.

Conversation and device identifiers provide provenance. They must not silently restrict recall when the task needs evidence across interactions.

## Working state and execution

Working state records outstanding questions, relevant canonical references, durable intentions, and completed steps. It must not become a hidden transcript or an unbounded substitute for model context.

Every capability invocation needs an explicit effect policy. Recovery must distinguish never executed, completed, failed, and uncertain external effects. Exactly-once behavior cannot be inferred merely from a durable checkpoint; use idempotency keys or reconciliation where the external service supports them.

## Concurrent events

Events sharing a timestamp remain distinct events. Preserve event IDs, source sequence and explicit causal dependencies. A deterministic processing order is an operational choice; it does not establish a unique real-world chronology for simultaneous independent events.

## Hardware

The initial resource policy may serialize model execution. Concurrent work is allowed only when measured resources and dependencies permit it. Model size, context budget, database costs, artifact I/O, and worker startup overhead must all be measured on representative hardware.
