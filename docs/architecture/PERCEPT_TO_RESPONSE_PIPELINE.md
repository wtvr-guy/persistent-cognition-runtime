# Percept-to-response pipeline

Current authority: Constitution Articles 12, 15, 21, 33, 35–37.

The current pipeline uses **fixed deterministic retrieval**, replacing the former Composer/Adaptive Recall control loop. See [Fixed retrieval](FIXED_RETRIEVAL.md) for the complete contract, registry map, budgets, stop rules, and migration behavior.

## Explicit user prompts: seven stages

| Stage | Responsibility | Model |
|---|---|---|
| RESOLVE_REFERENCES | Inspect bounded working-state availability and persist the capability-registration snapshot | No |
| EVIDENCE_POLICY | Commit historical source roles and response surface policy using only the current prompt | Narrow source-policy specialist, with deterministic recognition of explicit prior-assistant references |
| PRECOGNITIVE | Open a bounded, policy-filtered historical aperture and select registered non-memory capabilities | Narrow capability-selection specialist |
| EXECUTE_WORK | Run the committed registered-capability plan | Capability-dependent |
| RETRIEVE_MEMORY | Canonical roots/neighbors, then fixed BROAD → ASSOCIATIVE → RELATIONAL → FOCUSED routes | No |
| RESPOND | Realize an evidence-grounded answer or exact-source output under the committed policy | Narrow response specialist |
| PERSIST_RESULT | Persist the canonical response and activate working state | No |

Every explicit user prompt requires a response. The current-only `EVIDENCE_POLICY` worker runs before `PRECOGNITIVE` exposes historical evidence. Its immutable result controls the aperture, fixed retrieval, and response realization; capability selection cannot reclassify source policy. Source roles, exclusive cutoffs, byte bounds, exact-source extraction, and quarantined evidence transport remain enforced. Tool/action results bypass memory retrieval and reach the responder directly.

## Non-user situations: six stages

| Stage | Responsibility | Model |
|---|---|---|
| MEMORY | Reconstruct bounded working state and admit relevant memory context | No |
| TRIAGE | Classify the situation and select any registered work | Narrow specialist only when semantic triage is required |
| EXECUTE | Run registered capabilities under deterministic policy | Capability-dependent |
| RETRIEVE | Canonical roots/neighbors, then fixed BROAD → ASSOCIATIVE → RELATIONAL → FOCUSED routes | No |
| RESPOND | Realize an evidence-grounded answer, or explicit unknown | Narrow specialist when a natural response is required |
| PERSIST | Canonical response, artifacts, cursors, and working-state updates | No |

Each stage is a disposable guarded worker with independently durable stage artifacts before terminal database completion. Recovery uses the recorded stage result; it does not replay an already completed model call. Protocol-version mismatches fail closed.

Non-user situations preserve deterministic-first triage and execution. Optional semantic situation triage belongs to this path, not the explicit-user pipeline. Natural responses use the same fixed retrieval function, without reserving a model for the retrieval stage. Source policies control whether observations may enter durable memory.

The continuous percept service (`persistent_cognition.percept_service.PerceptService`, also exposed as `pcr-percept serve`) can run the same governed pipeline for scheduled events, anomaly/error alerts, external observations, and asynchronous tool/action outcomes. It uses bounded polling with backoff, durable restart-safe cursors, one PostgreSQL advisory ownership lock per scheduler namespace, and safe application-owned executor dispatch through trusted executors registered from process configuration. Observed data never supplies executor code, imports, or eval/exec behavior.

Installed source policy binds `trusted_executor` to an explicit deterministic task. Embedded applications supply a trusted worker argument sequence to re-register callbacks in every fresh process; the stock CLI worker intentionally has no application bindings. Structured work and observed success/failure receipts persist even without a response. External handlers must deduplicate by execution ID across crash retries. Tick failures are logged and durably recorded; database session loss stops execution until ownership is reacquired on restart.

There is no self-model or identity stage, schema, or response-policy field. Generic semantic facts, expectations, situations, and consolidation remain part of the pipeline.

The prior Composer, memory requirements, and coverage judge are historical experiments. Their prompts and decision fields are not active cognitive control. Original evidence remains inspectable in prior Git revisions and journal artifacts.
