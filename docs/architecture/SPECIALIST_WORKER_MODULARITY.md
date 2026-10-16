# Specialist Worker Modularity

Current implementation update (2026-09-28): the Composer and MEMORY_REQUIREMENTS
control stages are retired. The authoritative current stage sequence, registry
contracts, and migration rules are in [Fixed retrieval](FIXED_RETRIEVAL.md).
Earlier Composer-specific
descriptions below record the previous design and do not override this update.


**Constitutional authority:** implements Article 37 of
[`../../CONSTITUTION.md`](../../CONSTITUTION.md) and is subordinate to the
Constitution.

**Applies to:** every LLM-backed worker, stage contract, model invocation, retry,
handoff artifact, and future non-user-percept pipeline.

## Principle

An LLM worker is disposable semantic compute for one coherent decision. It is not a
container into which adjacent reasoning tasks are placed for convenience.

Persistent Cognition therefore separates roles when they differ in any of these ways:

- output schema or decision authority;
- required evidence classes or provenance boundary;
- failure, retry, or stopping semantics;
- resource/model requirements;
- downstream consumer;
- audit question answered by the result.

Sharing transport, validation, artifact-writing, or deterministic helper code does
not merge semantic roles. Shared infrastructure should remain ordinary reusable
software beneath narrow stage contracts.

## Required worker contract

Every LLM-backed stage must declare:

1. one named semantic responsibility;
2. its bounded authoritative inputs and separately quarantined evidence;
3. one application-owned output contract, or mutually exclusive realization modes
   that produce the same stage outcome;
4. which model call kinds the stage may invoke;
5. its retry/reassessment boundary;
6. its durable result and causal provenance;
7. its resource estimate and admission requirements.

A guarded worker must fail closed if it attempts to invoke a model role assigned to
another stage. Repeated calls are permitted only for validation retries or bounded
reassessment of the same semantic question. They do not authorize a second role.

## Current user-prompt specialists

The implemented fixed retrieval user-prompt path uses these process boundaries:

| Stage | Responsibility | LLM use |
|---|---|---|
| RESOLVE_REFERENCES | Inspect bounded working-state availability and persist the capability-registration snapshot | None |
| EVIDENCE_POLICY | Commit historical source roles and response surface policy from the current prompt only | One narrow source-policy role with bounded validation retries; explicit prior-assistant references use deterministic recognition |
| PRECOGNITIVE | Open the policy-filtered historical aperture and select required non-memory capability indices | One narrow capability-selection role with bounded validation retries |
| EXECUTE_WORK | Execute the committed application-owned plan | None in the stage itself; invoked capabilities own their contracts |
| RETRIEVE_MEMORY | Execute fixed bounded routes and merge source-admissible evidence with explicit budgets and diversity | None |
| RESPOND | Produce exact-source or natural output under the committed policy | One realization mode per path, with bounded validation retries |
| PERSIST_RESULT | Persist the canonical response and activate working state | None |

The `EVIDENCE_POLICY` result is an independent immutable stage artifact. Its worker sees only the current user prompt, before `PRECOGNITIVE` exposes historical evidence. The aperture, fixed retrieval, and response realization inherit that exact source policy and may not reclassify it. `PRECOGNITIVE` separately commits the capability-selection disposition and execution plan; capability selection cannot change evidence-policy authority. Every explicit user prompt requires a response.

The six-stage non-user situation path (`MEMORY`, `TRIAGE`, `EXECUTE`, `RETRIEVE`, `RESPOND`, `PERSIST`) is separate. Its optional semantic triage does not replace either user-prompt specialist.

The retrieval result is an immutable stage artifact containing evidence and route
receipts. It has no requirements, deficit, coverage decision, or sufficiency flag.
The retired split is preserved as an experiment in
[COMPOSER_REQUIREMENTS_001.md](https://github.com/wtvr-guy/prometheist/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/experiments/COMPOSER_REQUIREMENTS_001.md).

## Percept triage for non-user inputs

Scheduled tasks, triggered events, anomaly flags, sensor observations, and other
non-user percepts need a dedicated **Percept Triage Specialist** when their next
semantic action cannot be established deterministically. Its job is limited to
classifying the perceived situation into an application-owned action/response plan
and naming required evidence domains. It does not retrieve evidence, judge whether
retrieved evidence is sufficient, execute work, or generate user-facing language.

Explicit user prompts do not delegate `response_required` to this specialist; their
intake contract sets it to true. Non-user response behavior remains governed by each
percept class's deterministic intake policy wherever possible.

## When to split a worker

Split a worker when it makes two independently testable semantic decisions, needs
two unrelated schemas/prompts, consumes evidence unnecessary for one of its duties,
or produces intermediate state that another component should be able to inspect,
retry, replace, or reuse independently.

Do not split deterministic formatting into an LLM worker, create specialists before
a concrete semantic need exists, or treat every helper function as a worker. The
boundary is semantic authority and resource lifetime, not source-file size.

## Enforcement and audit

Current enforcement consists of:

- one fresh guarded process per durable percept stage;
- an application-owned stage-to-specialist registry;
- a fail-closed allowlist of LLM call kinds for every stage;
- independent evidence-policy and stage-result artifacts;
- regression tests covering stage-role completeness, cross-role rejection, and
  exact policy inheritance without reclassification.

A constitutional audit must identify every production LLM invocation site and map
it to one specialist contract. An unmapped call, a stage allowed to invoke another
stage's role, or a single worker performing independent semantic decisions is a
failure of Article 37.
