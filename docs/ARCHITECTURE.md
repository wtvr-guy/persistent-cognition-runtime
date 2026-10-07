# Architecture and code map

This extraction includes the v2 worker architecture and its later deterministic retrieval update.

| Stage | Responsibility | Model call |
| --- | --- | --- |
| MEMORY | Reconstruct bounded working state and admit relevant memory context | No |
| TRIAGE | Classify the situation and select any registered work | Fresh specialist only when semantic triage is required |
| EXECUTE | Execute registered capabilities with deterministic authority | Capability-dependent |
| RETRIEVE | Follow canonical roots/neighbors and fixed bounded retrieval routes | No |
| RESPOND | Realize an answer from admitted memory and authoritative work results | Fresh specialist when a natural response is required |
| PERSIST | Record response, artifacts, and working-state updates | No |

Each stage has durable input/output artifacts and a guarded worker contract. Fixed retrieval replaced the former Composer loop upstream; restoring it would roll back a later engine change.

For factual recall, the current-only intent classifier selects
`EXTRACTIVE_VALUES`. With ordinary output, Python displays the admitted canonical
source records as clearly attributed quotations. It makes no response-stage model
call and does not select spans or infer which statement is true. Multiple or
conflicting sources remain visible. A quotation is evidence, not a declaration
that every requested fact was found. This preserves full wording and relationships
when the source is prose, without an English phrase parser or a generated answer.

A trusted caller with a known structured contract can supply `SourceValueBinding`
objects to `generate_final_response(..., source_bindings=...)`. Each binding names
an admitted canonical event, work result plan position, or permitted current
message and a typed dictionary/list field path. Code reads that field directly
and renders the value. Missing sources, missing fields, non-scalar fields, and
inadmissible roles fail closed without invoking a model. Raw single-value output
and multi-value output with a validated caller-supplied separator are deterministic
as well. The intent model cannot author these bindings.
See [source_value_response.py](../src/persistent_cognition/source_value_response.py).

`SYNTHESIS` retains the model for explanations, comparisons, summaries and other
requested free-form language. Explicit raw extraction from unstructured prose
still requires semantic selection when no structured binding exists; it uses the
existing validated exact-source contract. The runtime does not parse arbitrary
English into an inferred fact database. Semantic intent can still be wrong, and
retrieval can still miss evidence. Those limitations must remain visible in native
acceptance rather than being mistaken for failures of deterministic copying.

Code-generated responses persist `RESPONSE_RENDER` artifacts with renderer version,
committed policy, exact canonical sources, field bindings, evidence references and
output. Native receipts replay these inputs and reject any response-stage model
invocation for a deterministic response. Model responses retain invocation and
validation artifacts. The full natural-language pipeline still uses models for
current-only intent and, when needed, semantic work selection.

| Concern | Implementation |
| --- | --- |
| JIT memory and retrieval | [jit_memory.py](../src/persistent_cognition/jit_memory.py), [fixed_retrieval.py](../src/persistent_cognition/fixed_retrieval.py), [postgres_memory_kernel.py](../src/persistent_cognition/postgres_memory_kernel.py) |
| Canonical evidence | [event_store.py](../src/persistent_cognition/event_store.py), [canonical_neighborhood.py](../src/persistent_cognition/canonical_neighborhood.py), [schema.sql](../schema.sql) |
| Derived and learned memory | [semantic_memory.py](../src/persistent_cognition/semantic_memory.py), [association_projection.py](../src/persistent_cognition/association_projection.py), [canonical_neighborhood.py](../src/persistent_cognition/canonical_neighborhood.py) |
| Stateless workers and percept service | [percept_response_runtime.py](../src/persistent_cognition/percept_response_runtime.py), [percept_response_worker.py](../src/persistent_cognition/percept_response_worker.py), [percept_service.py](../src/persistent_cognition/percept_service.py), [llm.py](../src/persistent_cognition/llm.py) |
| Contracts | [contract_registry.py](../src/persistent_cognition/contract_registry.py), [prompt_registry.py](../src/persistent_cognition/prompt_registry.py) |
| Attention and execution | [attention.py](../src/persistent_cognition/attention.py), [worker_runtime.py](../src/persistent_cognition/worker_runtime.py), [advisory_lock.py](../src/persistent_cognition/advisory_lock.py), [trusted_executors.py](../src/persistent_cognition/trusted_executors.py) |
| Recovery and journals | [artifact_journal.py](../src/persistent_cognition/artifact_journal.py), [artifact_recovery.py](../src/persistent_cognition/artifact_recovery.py), [percept_journal.py](../src/persistent_cognition/percept_journal.py) |
| Model configuration and admission | [runtime_settings.py](../src/persistent_cognition/runtime_settings.py), [model_admission.py](../src/persistent_cognition/model_admission.py), [model_residency.py](../src/persistent_cognition/model_residency.py) |

Learned memory remains because source-linked representations and their counterevidence are part of the memory engine. Generic source contracts accept explicit percepts; no sensor or node service is started.

The always-on governed percept service can continuously process scheduled events, anomaly/error alerts, external observations, and asynchronous tool/action outcomes with bounded polling/backoff, durable cursors, a scheduler namespace ownership lock, and trusted application-owned executor dispatch. Programmatic callers use `PerceptService.run(conn=None, *, max_iterations=None)` and `request_stop()`; CLI operators use `pcr-percept serve [--scheduler-key KEY]`. See [the complete pipeline](architecture/PERCEPT_TO_RESPONSE_PIPELINE.md), [fixed retrieval](architecture/FIXED_RETRIEVAL.md), and [worker modularity](architecture/SPECIALIST_WORKER_MODULARITY.md). Bounded model context does not establish perfect recall or constant total computation/storage cost.

The namespace propagates through ticks, consolidation/reflex cursors, situation dispatch, task progress, and worker claims. Heartbeat polling only observes that namespace's claims. Source candidates remain global evidence, so distinct namespaces are independent consumers. Interactive/one-shot CLI chat, CLI recovery, headless jobs, and ticks acquire the same lock as the service before mutating their namespace.

Application-owned source policies may bind a deterministic reaction to a trusted executor. An injected trusted `worker_command` re-registers callbacks in each fresh worker; parent-only registrations are insufficient. A callback receives observed content as data and a stable execution ID for external deduplication. Missing bindings fail closed. Failed callbacks produce observed failed receipts, while storage failures remain recoverable stage errors. Governed action outcomes return through normal intake without causing unbounded reaction chains. Non-response completions retain structured work.

Outcome feedback owns an action-specific situation rather than replacing an entity's pending source reaction. Stage failures are quarantined per task, not retried ahead of healthy siblings. Explicit application recovery preserves immutable artifacts, receipts, action IDs and assignment history; ambiguous external effects require reconciliation. Scheduler epochs release failed reservations only when no live claim owns them. Model admission is derived from the actual triage/response stages; deterministic non-response consolidation is model-free.

No operational identity, persona, imprinting, self-model stage/schema/response-policy field, or self-reflection store remains. Generic semantic evidence and counterevidence are retained; removing identity does not remove perception, expectations, situations, consolidation, or reflexes. The supervisor logs and durably records recoverable tick errors, releases ownership after rollback, and exits on connection loss. Restarting reacquires ownership before doing work. See [embedding setup](SETUP.md#continuous-percept-service-and-embedding).
