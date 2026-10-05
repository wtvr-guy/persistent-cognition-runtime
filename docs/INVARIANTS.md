# Engineering invariants

These are requirements for the proposed standalone runtime.

1. Canonical events are append-only; interpretations and corrections retain their evidence.
2. Every meaningful result is attributable to explicit inputs, sources, schema versions, and policy versions.
3. Model invocations have no hidden continuity or independent durable authority.
4. Each semantic specialist owns one coherent task with typed input and output contracts.
5. Deterministic policy owns identifiers, scheduling eligibility, dependencies, and resource admission.
6. Retrieval, active working state, and model inputs have governed bounds.
7. Derived memory is rebuildable and cannot silently become canonical evidence.
8. Equal timestamps do not merge events or imply conflicting accounts.
9. Worker failure cannot erase completed durable work.
10. Uncertain external effects require reconciliation before retry.
11. Meaningful stage results are durably published before terminal completion.
12. Interrupted interactions retain recoverable partial artifact chains.
13. Invalid control output, missing evidence, or unusable resource state leads to explicit failure, wait, abstention, or reconciliation.
14. Runtime-affecting numeric bounds have a documented classification and evidence.
15. Native hardware acceptance complements deterministic regression tests.
16. Architectural changes are explicit, versioned, and supported by experiments.

The operational database must not be the only recoverable copy of meaningful history and cognitive progress. Artifact retention, integrity, backup, and replay must be tested together.
