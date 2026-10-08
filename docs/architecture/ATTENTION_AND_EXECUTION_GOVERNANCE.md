# Attention and Execution Governance

**Status:** constitutional architecture deep dive.  
**Constitutional authority:** implements Articles 17–20, 34, and 35 of [`../../CONSTITUTION.md`](../../CONSTITUTION.md).

This document defines Persistent Cognition's durable executive-control and worker-execution invariants independent of any one milestone implementation. The v0.7 scheduler, resource-admission, preemption, and worker-protocol documents are implementation evidence for these rules; this document is the long-lived normative authority beneath the Constitution.

## Core separation

Persistent Cognition asks two different questions before work runs:

1. **Attention:** Which durable runnable work deserves execution, and in what deterministic order?
2. **Resource admission:** Which compatible subset of that work can execute concurrently inside the current safe physical-resource envelope?

> **Priority determines entitlement and ordering. Resource state determines safe concurrency.**

Neither question is owned by a model or worker.

## Attention authority

The Attention Fabric owns one globally authoritative model of runnable durable work within the deployment's persistence/control domain.

Tasks carry structured scheduling metadata such as:

- stable task identity;
- authoritative creation/order sequence;
- criticality/effective priority;
- service class and service guarantees;
- dependencies;
- deadline when applicable;
- interruption policy;
- required capabilities;
- resource requirements;
- resumable/checkpoint state.

Priority is derived by application policy rather than model preference. If two items otherwise tie, the scheduler uses a stable deterministic total-order rule.

Workers may discover new work or propose task metadata through constrained contracts. They do not acquire executive authority merely by existing or completing first.

## Resource admission

Execution resources are authoritative system state. Different resource classes may use different capacity semantics.

Examples include:

- CPU/general compute;
- RAM;
- local LLM inference slots;
- GPU/VRAM when supported;
- database capacity;
- filesystem I/O;
- network I/O;
- device/actuator or other exclusive resources.

Logical CPU count is not a universal concurrency number. A host can expose many CPU threads yet still safely support only one local-model generation. Conversely, database, filesystem, network, and lightweight CPU work may be able to proceed concurrently with that generation.

Resource classes and dimensions should be introduced when a concrete safety or performance requirement demonstrates that the distinction matters. Do not create permanent hardware partitions merely to simplify scheduling.

## Safe capacity and headroom

Installed hardware is not equivalent to schedulable capacity.

Persistent Cognition must reserve capacity for the operating system, required services, Persistent Cognition's own control-plane work, and operator-authorized applications.

Conceptually:

```text
observed/configured physical capacity
    - required system/user headroom
    - uncertainty reserve
    - live Persistent Cognition reservations
    = currently admissible capacity
```

Headroom is a safety property. It is not spare capacity that the scheduler may silently reclaim because a queue is long.

The local-model baseline is conservative: **one concurrent local LLM inference slot unless calibrated evidence and the declared host policy justify more.** The number one is a safety default, not a universal performance truth; changing it is governed by empirical/environment calibration.

## Authoritative observations

Physical availability varies. Persistent Cognition handles that variability by converting relevant observations into explicit decision inputs.

A resource observation that materially controls admission should identify at least:

- the host/resource domain observed;
- measured capacity/pressure relevant to policy;
- observation health/status;
- freshness/time boundary;
- observation or policy version;
- enough provenance to identify the exact snapshot consumed by the decision.

Missing, stale, failed, mismatched, or otherwise unusable safety observations fail closed when live safety depends on them.

Sampling may occur outside deterministic policy. The decision made from the sampled authoritative input remains deterministic.

## Scheduling epochs

Workers must not independently race to claim arbitrary runnable tasks.

A scheduling epoch conceptually performs:

1. load dependency-satisfied durable work;
2. capture/load authoritative safe resource state;
3. derive deterministic effective order;
4. preserve still-valid committed work where policy permits;
5. admit compatible candidates whose complete reservations fit;
6. if required, evaluate legal preemption deterministically;
7. persist the complete resulting assignment/reservation authority atomically;
8. release only committed assignments to the worker boundary.

The exact persistence schema may evolve. The atomic-authority invariant may not.

A partially persisted epoch or reservation set is not worker-visible authority.

## Safe parallelism

Persistent Cognition should execute as much compatible work concurrently as safe capacity permits.

It must not serialize the machine simply because one task has higher priority or because the original implementation used one focus slot.

Likewise, concurrency is not a goal that overrides safety. If one task requires most of the safe RAM, an exclusive model slot, or another scarce resource, the correct assignment set may contain only that task.

> **Safe parallelism is the default; concentration is the consequence of actual resource demand, not a permanent lane topology.**

## Preemption

Priority alone does not justify interruption.

A higher-priority task first asks:

```text
Can I be admitted safely beside current work?
    yes -> admit; current compatible work continues
    no  -> identify the exact deficient resource classes
```

Only work occupying relevant deficient capacity is considered as a victim.

Persistent Cognition then respects declared interruption semantics, at minimum the concepts represented by:

- **PREEMPTIBLE** — may yield when selected by policy;
- **CHECKPOINT_ONLY** — may be asked to yield, but releases authority/resources only at a declared safe checkpoint;
- **ATOMIC** — normal priority policy does not interrupt the active atomic operation.

If legal preemption is necessary, choose victims with a deterministic total order and release only the minimum work/capacity required by the policy to admit the blocked higher-priority task.

Same-priority arrivals should not thrash stable running work without an explicit policy reason.

## Service guarantees and starvation

A deterministic priority queue must not equate “low priority” with “never.” Service guarantees or equivalent anti-starvation mechanisms are part of executive policy.

Their exact numeric values are empirical constraints rather than constitutional constants. Whatever policy is used must be:

- structured and versioned;
- deterministic from authoritative inputs;
- subordinate to interruption/resource safety;
- benchmarked under overload and mixed workloads.

See [`../engineering/EMPIRICAL_CONSTRAINT_GOVERNANCE.md`](../engineering/EMPIRICAL_CONSTRAINT_GOVERNANCE.md).

## Assignment is entitlement, not launch permission

A durable committed assignment establishes which work owns resource entitlement. It does **not** prove that the host is still safe at the instant a new process starts.

Before process launch, the worker boundary must perform a guarded claim appropriate to the deployment that:

- verifies the assignment/task is still authoritative and current;
- checks no conflicting live claim already owns the work;
- re-observes resource safety when live conditions matter;
- binds the decision to the committed resource-safety policy;
- records the admission/denial result;
- acquires durable ownership/lease semantics before executing.

A denied claim must not spawn the worker.

Transient resource pressure is a wait/re-observe condition when policy permits bounded retry. It is never permission to lower reservations, headroom, or thresholds simply to make progress.

## Disposable-worker contract

A worker is a bounded executor, not an agent with durable identity.

Conceptually:

```text
committed assignment
    -> guarded durable claim
    -> reconstruct bounded step/checkpoint from canonical state
    -> execute one bounded unit
    -> persist checkpoint/result/provenance
    -> release/complete claim
    -> process may disappear
```

A worker receives durable identifiers and bounded task-local references. It must not depend on inherited in-process cognition or a hidden prior LLM transcript.

## Durable step and recovery semantics

Externally visible work requires explicit crash semantics.

A bounded work step should have:

- stable durable step identity;
- assignment/task/revision binding;
- stable idempotency identity where retry is allowed;
- explicit side-effect policy;
- ownership/lease or equivalent claim state when concurrent execution is possible;
- append-only or otherwise auditable checkpoint revisions;
- at most one authoritative terminal result;
- provenance tying results to the exact work/claim/evidence that produced them.

The system should distinguish at least these effect classes conceptually:

- safe to retry because no external effect occurred;
- safe to retry only with the same idempotency key;
- ambiguous/at-most-once effect that fails closed after an unconfirmed crash until reconciliation can establish what occurred.

“Try it again and hope” is not a recovery strategy for irreversible effects.

## Lease expiry and abandonment

A dead worker must not hold resources forever. Conversely, a slow live worker must not be duplicated merely because another process assumes it is dead.

Lease/heartbeat/abandonment parameters are empirical safety tunables, but the protocol must ensure:

- only one live owner when exclusivity is required;
- ownership-checked heartbeat/renewal;
- deterministic abandonment after the declared recovery boundary;
- reservation protection while a valid live claim owns capacity;
- stable retry/idempotency semantics after abandonment;
- fail-closed treatment of ambiguous at-most-once effects.

## Capabilities inside admitted work

A model may identify that one or more application-owned capabilities are required. That selection does not supersede Attention governance.

Persistent Cognition owns:

- canonical capability identity;
- dependency closure;
- availability/permission checks;
- resource requirements;
- deterministic execution ordering;
- worker-step construction;
- idempotency/effect policy;
- final durable result authority.

In a given version, several capability substeps may execute inside one admitted task. A later version may promote them to independently scheduled child tasks. Either representation must preserve the same system-owned authority boundary.

## Failure modes that must fail closed

Examples include:

- stale task/assignment revisions;
- stale or unhealthy resource observations when live safety depends on them;
- policy-version mismatch;
- missing required resource class/capacity;
- oversubscription;
- conflicting live claims;
- already-completed work;
- invalid checkpoint ownership;
- unknown/ambiguous external effect after a crash;
- partial scheduler persistence;
- dependency cycles or missing dependencies.

Failing closed may leave work queued or require reconciliation. It must not fabricate authority or weaken safety.

## Testing requirements

This governance layer requires deterministic and native evidence.

Deterministic tests should cover:

- total ordering and service guarantees;
- dependency gating;
- synthetic admission/rejection and no oversubscription;
- deterministic reservation identity;
- atomic epoch publication/rollback;
- safe concurrent assignments;
- contention-specific deterministic preemption;
- PREEMPTIBLE/CHECKPOINT_ONLY/ATOMIC semantics;
- worker claim exclusion/lease/checkpoint/result invariants;
- restart/reconstruction equivalence;
- idempotent retry and ambiguous-effect failure;
- forced process destruction.

Native acceptance should cover actual host CPU/RAM pressure, local model startup/inference behavior, database/process interaction, resource re-observation, and failure/restart paths where simulation is insufficient.

See [`../engineering/TESTING_AND_ACCEPTANCE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/engineering/TESTING_AND_ACCEPTANCE.md).

## Historical implementation evidence

The following remain useful implementation records, but are subordinate to this constitutional deep dive and the Constitution:

- [`../milestones/v0.7/JIT_ATTENTION_DESIGN.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/milestones/v0.7/JIT_ATTENTION_DESIGN.md)
- [`../milestones/v0.7/RESOURCE_ADMISSION_CLARIFICATION_2026-08-24.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/milestones/v0.7/RESOURCE_ADMISSION_CLARIFICATION_2026-08-24.md)
- [`../milestones/v0.7/ATTENTION_FOCUS_CONCENTRATION_2026-08-26.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/milestones/v0.7/ATTENTION_FOCUS_CONCENTRATION_2026-08-26.md)
- [`../milestones/v0.7/INCREMENT_F_WORKER_PROTOCOL_2026-08-26.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/milestones/v0.7/INCREMENT_F_WORKER_PROTOCOL_2026-08-26.md)

## Invariant

> **Persistent Cognition owns attention, admission, preemption, assignments, resource safety, launch authority, and recovery semantics. Workers merely execute bounded committed work and may disappear at any time.**
