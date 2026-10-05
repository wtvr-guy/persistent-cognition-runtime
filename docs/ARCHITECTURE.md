# Architecture and code map

This extraction includes the v2 worker architecture and its later deterministic retrieval update.

| Stage | Responsibility | Model call |
| --- | --- | --- |
| RESOLVE_REFERENCES | Reconstruct bounded working state | No |
| EVIDENCE_POLICY | Classify the current request's admissible historical sources | Fresh specialist |
| PRECOGNITIVE | Activate relevant context and select external work | Only when required by the catalog |
| EXECUTE_WORK | Execute registered capabilities with deterministic authority | Capability-dependent |
| RETRIEVE_MEMORY | Follow canonical roots/neighbors and fixed bounded retrieval routes | No |
| RESPOND | Realize an answer from admitted memory and authoritative work results | Fresh specialist |
| PERSIST_RESULT | Record response and working-state updates | No |

Each stage has durable input/output artifacts and a guarded worker contract. Fixed retrieval replaced the former Composer loop upstream; restoring it would roll back a later engine change.

| Concern | Implementation |
| --- | --- |
| JIT memory and retrieval | [jit_memory.py](../src/prometheist/jit_memory.py), [fixed_retrieval.py](../src/prometheist/fixed_retrieval.py), [postgres_memory_kernel.py](../src/prometheist/postgres_memory_kernel.py) |
| Canonical evidence | [event_store.py](../src/prometheist/event_store.py), [canonical_neighborhood.py](../src/prometheist/canonical_neighborhood.py), [schema.sql](../schema.sql) |
| Derived and learned memory | [semantic_memory.py](../src/prometheist/semantic_memory.py), [self_memory.py](../src/prometheist/self_memory.py), [self_memory_navigation.py](../src/prometheist/self_memory_navigation.py) |
| Stateless workers | [percept_response_runtime.py](../src/prometheist/percept_response_runtime.py), [percept_response_worker.py](../src/prometheist/percept_response_worker.py), [llm.py](../src/prometheist/llm.py) |
| Contracts | [contract_registry.py](../src/prometheist/contract_registry.py), [prompt_registry.py](../src/prometheist/prompt_registry.py) |
| Attention and execution | [attention.py](../src/prometheist/attention.py), [worker_runtime.py](../src/prometheist/worker_runtime.py) |
| Recovery and journals | [artifact_journal.py](../src/prometheist/artifact_journal.py), [artifact_recovery.py](../src/prometheist/artifact_recovery.py), [percept_journal.py](../src/prometheist/percept_journal.py) |
| Model configuration and admission | [runtime_settings.py](../src/prometheist/runtime_settings.py), [model_admission.py](../src/prometheist/model_admission.py), [model_residency.py](../src/prometheist/model_residency.py) |

Learned memory remains because source-linked representations and their counterevidence are part of the memory engine. This does not install a private imprint profile or claim a human identity. Generic source contracts accept explicit percepts; no sensor or node service is started.

See [the complete pipeline](architecture/PERCEPT_TO_RESPONSE_PIPELINE.md), [fixed retrieval](architecture/FIXED_RETRIEVAL.md), and [worker modularity](architecture/SPECIALIST_WORKER_MODULARITY.md). Bounded model context does not establish perfect recall or constant total computation/storage cost.
