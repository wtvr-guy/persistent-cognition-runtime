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
