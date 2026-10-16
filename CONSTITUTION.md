# Runtime Engineering Constitution

Engineering rules inherited from the pinned source revision. The source article numbers are retained so the accompanying specifications remain traceable. This document governs persistent state, memory, inference and execution; application mission policy are outside this engine repository.

Changes to these rules require explicit documentation and appropriate regression evidence. A source-code change or passing test does not silently redefine an invariant.

### Article 7 — Continuity belongs to the system

**Rule.** Memory, active working state, tasks, attention, interaction continuity, policy, execution state, and causal provenance belong to Persistent Cognition itself. They must not depend on an LLM context window, chat transcript, worker process, or named agent remaining alive.

**Why it matters.** A persistent cognitive system cannot be persistent if its state disappears when disposable compute disappears.

**Deep dive:** [`docs/architecture/COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md)

### Article 8 — Every LLM invocation is stateless; workers are disposable

**Rule.** Every LLM call starts fresh. No model invocation may inherit a hidden transcript or private context from an earlier invocation. Workers receive only bounded system-owned durable/task-local inputs, persist their result or checkpoint, and may then disappear. Agent-like names may describe temporary roles, but permanent agents are not owners of durable continuity or executive authority.

**Why it matters.** Stateless inference makes continuity inspectable, restart-safe, model-replaceable, and independent of process lifetime.

**Deep dive:** [`docs/architecture/COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md)

### Article 9 — Admitted durable memory is lossless, append-only canonical evidence

**Rule.** Once information is admitted as canonical durable memory or authoritative internal history, ordinary retention, indexing, summarization, aggregation, or storage-pressure policy must not replace, rewrite, or delete the exact canonical evidence. Canonical historical events are append-only: corrections, contradictions, and supersession are represented by new provenance-bearing records that refer to earlier evidence rather than mutating history. Derived structures are navigation aids, not substitute memories.

**Why it matters.** Persistent Cognition's accuracy objective depends on being able to return to exact source evidence and reconstruct what the system actually knew at a point in time rather than trusting successively lossy or retrospectively rewritten history.

**Deep dive:** [`docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md`](docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md)

### Article 10 — Derived memory is replaceable; canonical evidence is not

**Rule.** Indexes, embeddings, entity links, association edges, salience/activation metadata, summaries used as aids, and other derived structures must be explicitly non-authoritative and rebuildable wherever their derivation permits it. Non-deterministically derived assertions must retain provenance, method/version, and non-authoritative status.

**Why it matters.** Persistent Cognition must be able to improve its retrieval machinery without rewriting its history.

**Deep dive:** [`docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md`](docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md)

### Article 11 — Ordinary cognition and recall are bounded

**Rule.** No ordinary mechanism may solve memory scale by growing a model context window, hidden worker transcript, or ordinary-case inference count in proportion to total corpus size. Workers externalize bounded structured state and request additional exact evidence just in time.

**Why it matters.** External persistent memory only solves context growth if each inference remains bounded as history grows.

**Deep dives:** [`docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md`](docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md), [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 12 — Basic persistent-memory access precedes model routing

**Rule.** Every percept receives a small system-owned, bounded, provenance-bearing activation of potentially relevant persistent memory before a fresh model decides what further capability work is needed. A model must not first be asked whether unseen memory matters.

**Why it matters.** Asking a stateless model whether unseen evidence is relevant is circular: the evidence needed to answer may itself be unseen.

**Deep dive:** [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 13 — WorkingState is bounded canonical activation, not a hidden memory store

**Rule.** Active working state consists of bounded system-owned state that points back to canonical evidence. It must not become an ever-growing transcript, free-form summary, profile blob, inferred truth store, or substitute long-term memory.

**Why it matters.** WorkingState should preserve current cognitive focus without reintroducing the context-window problem through another name.

**Deep dive:** [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 14 — Conversations, sessions, devices, and interfaces are provenance, not cognitive boundaries

**Rule.** Stored conversation IDs and interface/session/device identifiers may constrain provenance, ordering, UI, debugging, or an explicitly scoped request, but they are not default semantic walls around memory or continuity.

**Why it matters.** Persistent Cognition is intended to maintain one persistent history and interaction continuity rather than fragment cognition into chat containers.

**Deep dive:** [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 15 — System control is deterministic wherever deterministic control is possible

**Rule.** Every operation that can be accomplished correctly through practical deterministic code must use that code. This includes canonical evidence lookup, structured field access, exact copying, validation, and evidence display as well as stable identity, total ordering, priority derivation, dependency handling, capability identity, execution order, resource policy, retention/deletion authority, permissions, retry semantics, and other control-plane decisions. An LLM is permitted only for the semantic work that deterministic code cannot reasonably perform, such as interpreting natural-language intent, selecting among semantic alternatives, or producing requested free-form language. A model must not be invoked merely to read, select text ranges from, or redisplay evidence when an admitted structured binding or faithful source display already satisfies the operation. Given the same authoritative durable state, authoritative observations, and policy versions, Persistent Cognition must reconstruct the same system decision.

**Why it matters.** Models may interpret semantics, but durable system authority must remain replayable, auditable, and independent of races or model preference.

**Deep dive:** [`docs/architecture/SYSTEM_DETERMINISM.md`](docs/architecture/SYSTEM_DETERMINISM.md)

### Article 16 — Race conditions never decide durable authority

**Rule.** Persistent Cognition must not let independently racing workers determine which durable task, resource, capability, or side effect wins. Selection is committed by deterministic system policy before workers act.

**Why it matters.** Operating-system scheduling may be nondeterministic; Persistent Cognition's durable executive decisions must not be.

**Deep dive:** [`docs/architecture/SYSTEM_DETERMINISM.md`](docs/architecture/SYSTEM_DETERMINISM.md)

### Article 17 — Attention priority and resource admission are separate

**Rule.** Attention determines which durable work deserves execution and its deterministic order. Resource admission determines which compatible subset can safely run concurrently under current authoritative capacity, reservations, and headroom. Persistent Cognition should exploit safe parallelism rather than serialize work unnecessarily.

**Why it matters.** Priority is a semantic/executive property; physical concurrency is a hardware-safety property. Collapsing them wastes resources or creates unsafe oversubscription.

**Deep dive:** [`docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md`](docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md)

### Article 18 — Resource safety is fail-closed and preserves headroom

**Rule.** Work may start only after the authoritative resource policy says its reservation fits safely. Persistent Cognition reserves headroom for the operating system and required or operator-authorized processes. Local LLM inference defaults conservatively to one concurrent slot until evidence justifies another value. Transient pressure may delay and trigger bounded re-observation; it must not silently weaken the committed safety thresholds.

**Why it matters.** A scheduler that can crash or severely degrade the host is not a valid attention mechanism.

**Deep dive:** [`docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md`](docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md)

### Article 19 — Preemption requires both priority and real contention

**Rule.** Higher-priority work does not preempt lower-priority work merely because it is higher priority. Preemption is considered only when relevant occupied capacity prevents safe admission and the victim's declared interruption policy permits yielding. The minimum deterministically selected work necessary to resolve the contention should be disturbed.

**Why it matters.** Persistent Cognition should focus resources when necessary without throwing away useful safe concurrency or violating execution safety.

**Deep dive:** [`docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md`](docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md)

### Article 20 — Disposable-worker execution is durable, guarded, and recovery-safe

**Rule.** A committed assignment is entitlement, not process-start permission. Worker launch must pass guarded claim-time admission; work must have durable identity, leases/ownership where needed, append-only checkpoints, explicit terminal results, and side-effect/idempotency semantics that fail closed when an irreversible effect cannot be proven safe to retry.

**Why it matters.** Process destruction is expected behavior, so recovery and duplicate-effect prevention must be properties of the protocol rather than worker memory.

**Deep dive:** [`docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md`](docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md)

### Article 21 — Models select semantic requirements; Persistent Cognition owns execution policy

**Rule.** Models may select among bounded application-owned semantic alternatives or capabilities. They do not author capability IDs, dependencies, execution order, resource policy, permissions, durable identifiers, or scheduling authority. Capability results should be structured evidence/state whenever possible.

**Why it matters.** Semantic interpretation is useful; model-authored control planes are difficult to validate, replay, secure, and audit.

**Deep dives:** [`docs/architecture/COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md), [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 22 — Model-generated natural language is control/state of last resort

**Rule.** Machine control and durable state prefer enums, booleans, bounded integers, application-owned IDs, and mechanically verified extractive selections before free-form generated language. Natural language is appropriate when language is genuinely the product or when no smaller mechanically verifiable representation can express the required semantics.

**Why it matters.** Closed representations reduce ambiguity, hallucinated control data, brittle parsers, and nondeterministic protocol behavior.

**Deep dives:** [`docs/architecture/COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md), [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 23 — Relevance, activation, evidence sufficiency, and truth are distinct

**Rule.** Attention and retrieval do not promote content to truth. Persistent Cognition must preserve distinctions among canonical evidence, user statements/beliefs, system interpretation, derived hypotheses, corrections/supersession, counterevidence, confidence, and unknown. Unsupported facts may remain unknown.

**Why it matters.** High-recall memory activation is useful only if it does not silently lower epistemic standards.

**Deep dives:** [`docs/architecture/COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md), [`docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md`](docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md)

### Article 24 — Material influence must leave durable causal provenance

**Rule.** Anything that materially influences Persistent Cognition's attention, reasoning, decisions, commitments, actions, or meaningful interaction continuity must leave enough durable provenance to explain that behavior later. High-volume external raw input may remain ephemeral before admission, but the causal record of what actually influenced the system must survive.

**Why it matters.** A persistent system must be able to explain why it acted as it did even if raw sensor/input buffers are later gone.

**Deep dives:** [`docs/architecture/COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md), [`docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md`](docs/architecture/LOSSLESS_PROGRESSIVE_MEMORY.md)

### Article 25 — Internal memory and external knowledge remain distinct evidence domains

**Rule.** Retrieval from Persistent Cognition's own persistent memory is distinct from external knowledge retrieval such as web/API/tool calls. Their provenance, authority, freshness, and failure semantics must remain explicit; one source must not silently masquerade as the other.

**Why it matters.** Remembering what the system/user previously experienced is epistemically different from learning something from the outside world now.

**Deep dive:** [`docs/architecture/COGNITIVE_ARCHITECTURE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/architecture/COGNITIVE_ARCHITECTURE.md)

### Article 26 — Natural-language continuity must not become vocabulary patchwork

**Rule.** Application policy must not accumulate phrase-specific English rules, regex vocabularies, or hand-written semantic parsers to decide whether persistent context exists, which prior conversation matters, or whether the model should remember. Lexical/entity/temporal/associative mechanisms remain legitimate inside the memory subsystem as evidence-retrieval machinery.

**Why it matters.** Surface-form patches overfit fixtures, regress each other, and shift semantic authority into brittle application code.

**Deep dive:** [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 27 — Local-first, operator-controlled, model-agnostic, replaceable

**Rule.** Durable state, replay, audit and the control plane must not depend on a hosted model service, a particular model, or heavyweight orchestration infrastructure. Keep components modular and inspectable, with model-dependent cognition degrading as a capability when no model is available. PostgreSQL and Ollama are the current reference implementations. This extraction does not claim a second storage backend is implemented.

**Why it matters.** Modest hardware and local data control are engineering requirements. Components can improve without making persistent state dependent on a vendor or one transient model.

### Article 29 — Development follows a falsifiable one-mechanism experimental discipline

**Rule.** Freeze a measurable baseline, change one mechanism, rerun the same experiment, and keep the mechanism only if the evidence justifies it. Preserve negative results. Do not add infrastructure, retrieval machinery, optimization solvers, models, or architectural layers because they are fashionable or theoretically attractive; add them when a frozen failure or measured limitation justifies the complexity.

**Why it matters.** Persistent Cognition is trying to discover which mechanisms are necessary. Changing several mechanisms at once destroys causal evidence and makes complexity accumulate without proof.

**Deep dive:** [`docs/engineering/EMPIRICAL_CONSTRAINT_GOVERNANCE.md`](docs/engineering/EMPIRICAL_CONSTRAINT_GOVERNANCE.md)

### Article 30 — Behavioral numbers must be classified and evidenced

**Rule.** Runtime-affecting numeric bounds must be structural invariants, external contracts, explicit collision bounds, empirical tunables, safety tunables, or environment-calibrated values. Behavioral magic numbers may not bypass classification. Environment-sensitive values require evidence from the intended environment; CI simulation alone cannot establish native safety or model behavior.

**Why it matters.** A value is not correct because it happened to make the current tests pass.

**Deep dive:** [`docs/engineering/EMPIRICAL_CONSTRAINT_GOVERNANCE.md`](docs/engineering/EMPIRICAL_CONSTRAINT_GOVERNANCE.md)

### Article 31 — Verification requires deterministic regression evidence and native acceptance where reality matters

**Rule.** Deterministic CI and development/deployment-machine acceptance have different jobs and neither substitutes for the other. Core invariants require restart/replay, cross-process, cross-session, provenance, bounded-context, hidden-transcript-absence, failure/recovery, and deterministic-order coverage. Resource- or local-model-sensitive claims must also be exercised on representative real hardware/runtime before they are treated as verified.

**Why it matters.** Synthetic determinism catches regressions; real-machine acceptance proves that assumptions about processes, memory pressure, local models, databases, and recovery survive contact with the actual deployment environment.

**Deep dive:** [`docs/engineering/TESTING_AND_ACCEPTANCE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/engineering/TESTING_AND_ACCEPTANCE.md)

### Article 32 — Constitutional changes must be explicit

**Rule.** Code, a milestone implementation, or a passing test cannot silently repeal a constitutional rule. Changing an article requires an explicit Constitution amendment, corresponding updates to affected constitutional deep dives, and acceptance evidence appropriate to the changed invariant. Superseded wording remains visible in history when it explains an earlier accepted result.

**Why it matters.** The Constitution is useful for audits only if architectural drift cannot redefine the rules implicitly.

**Deep dive:** [`docs/engineering/CONSTITUTIONAL_GOVERNANCE.md`](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/engineering/CONSTITUTIONAL_GOVERNANCE.md)

### Article 33 — Retrieval control is bounded, deterministic, and evidence-preserving

**Rule.** Persistent Cognition owns the retrieval sequence, source scope, history cutoff,
merging, evidence budgets, and stop conditions in ordinary replayable software.
The initial replacement policy runs a fixed finite sequence of retrieval routes;
models do not certify memory sufficiency, author free-form deficits as control
state, or become a second executive. Completion of retrieval does not establish
that a question is answerable. The final responder receives admissible canonical
evidence, separately labeled derived memory context, and authoritative completed
capability results, and must preserve uncertainty and legitimate unknowns.

**Why it matters.** A semantic coverage judge can suppress useful evidence or
repeat reasoning without improving recall. Deterministic bounded retrieval makes
that mechanism independently testable while preserving statelessness and the
separation of memory, action authority, and response realization.

**Deep dive:** [`docs/architecture/FIXED_RETRIEVAL.md`](docs/architecture/FIXED_RETRIEVAL.md)

### Article 34 — Durable queued work must have an explicit anti-starvation policy

**Rule.** Lower-priority durable work may wait behind more important work, but the scheduler must provide structured, deterministic service guarantees or an equivalent explicit anti-starvation mechanism. Exact guarantee thresholds are governed tunables and never override resource safety, dependencies, or interruption safety.

**Why it matters.** A durable intention that can remain runnable forever without any governed path to service is not meaningfully durable executable work.

**Deep dive:** [`docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md`](docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md)

### Article 35 — Insufficient authority or evidence fails closed

**Rule.** Persistent Cognition must not guess past invalid, stale, contradictory, missing, or ambiguous control authority. Invalid model-control output, unusable resource state, unresolved dependency authority, unsupported factual evidence, and ambiguous irreversible side effects must produce an explicit failure, abstention, wait, or reconciliation state as appropriate rather than fabricated success or weakened policy.

**Why it matters.** Determinism, provenance, resource safety, and epistemic accuracy all fail if the system silently invents authority when the evidence or control contract is insufficient.

**Deep dives:** [`docs/architecture/SYSTEM_DETERMINISM.md`](docs/architecture/SYSTEM_DETERMINISM.md), [`docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md`](docs/architecture/ATTENTION_AND_EXECUTION_GOVERNANCE.md), [`docs/architecture/INTERACTION_CONTINUITY.md`](docs/architecture/INTERACTION_CONTINUITY.md)

### Article 36 — Meaningful state transitions require independent artifact durability

**Rule.** Canonical events and meaningful cognitive or operational boundaries that matter for explanation, replay, recovery, or reconstruction must have an immutable, inspectable durable artifact outside the primary operational database. A disposable worker must durably publish its stage result before that stage is treated as terminal. Completed interactions must have a final-disposition manifest over their artifact chain; interrupted interactions retain their partial chain as recoverable state. PostgreSQL or any future primary database may be the indexed operational representation, but it must not be the only surviving copy from which Persistent Cognition's canonical history and recoverable cognitive progress can be reconstructed.

**Why it matters.** Stateless cognition is only genuinely restart-safe and user-auditable if the exact artifacts that crossed worker boundaries survive process failure and database loss. Independent artifacts also let Persistent Cognition diagnose what a worker actually knew, resume without repeating completed cognition, and rebuild canonical history after storage corruption.

**Deep dive:** [`docs/architecture/IMMUTABLE_ARTIFACT_JOURNAL.md`](docs/architecture/IMMUTABLE_ARTIFACT_JOURNAL.md)

### Article 37 — LLM workers are narrow semantic specialists

**Rule.** One guarded LLM worker process may own only one coherent semantic
responsibility. Independent decisions such as percept triage, evidence policy,
work selection, evidence interpretation, and response realization require separate
specialist stages with typed inputs and outputs. Retries or bounded reassessment may
repeat the same role, but a worker must not accumulate unrelated duties, hidden
intermediate cognition, or cross-role context merely to reduce process count.
Specialist boundaries must be explicit, independently auditable, and guarded so a
stage cannot invoke another specialist's model contract.

**Why it matters.** Narrow workers reduce model-context and transient-resource
pressure, make failures attributable to one decision, allow components and models
to be replaced independently, and prevent a convenient worker from quietly becoming
a general-purpose persistent agent.

**Deep dive:** [`docs/architecture/SPECIALIST_WORKER_MODULARITY.md`](docs/architecture/SPECIALIST_WORKER_MODULARITY.md)

