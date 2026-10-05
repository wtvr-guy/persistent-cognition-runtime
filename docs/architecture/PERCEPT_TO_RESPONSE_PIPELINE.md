# Percept-to-response pipeline

Current authority: Constitution Articles 12, 15, 21, 33, 35–37.

The current pipeline uses **fixed deterministic retrieval**, replacing the former Composer/Adaptive Recall control loop. See [Fixed retrieval](FIXED_RETRIEVAL.md) for the complete contract, registry map, budgets, stop rules, and migration behavior.

| User stage | Responsibility | Model |
|---|---|---|
| RESOLVE_REFERENCES | Reconstruct bounded working state | No |
| EVIDENCE_POLICY | Current-prompt source and response-surface classification | Narrow specialist |
| PRECOGNITIVE | Open aperture, activate self-context, select external capabilities | Only if catalog requires selection |
| EXECUTE_WORK | Run registered capabilities under deterministic policy | Capability-dependent |
| RETRIEVE_MEMORY | Canonical roots/neighbors, then fixed BROAD → ASSOCIATIVE → RELATIONAL → FOCUSED routes | No |
| RESPOND | Realize an evidence-grounded answer, or explicit unknown | Narrow specialist |
| PERSIST_RESULT | Canonical response and working-state updates | No |

Every explicit user prompt requires a response. Evidence policy precedes exposing historical text to response realization. Source roles, exclusive cutoffs, byte bounds, exact-source extraction, and quarantined evidence transport remain enforced. Tool/action results bypass memory retrieval and reach the responder directly.

Each stage is a disposable guarded worker with independently durable stage artifacts before terminal database completion. Recovery uses the recorded stage result; it does not replay an already completed model call. Protocol-version mismatches fail closed.

Non-user situations preserve deterministic-first triage and execution. Natural responses use the same fixed retrieval function, without reserving a model for the retrieval stage. Self-proposal and self-evidence review remain separately guarded semantic roles. Source policies control whether observations may seed identity learning.

The prior Composer, memory requirements, and coverage judge are historical experiments. Their prompts and decision fields are not active cognitive control. Original evidence remains inspectable in prior Git revisions and journal artifacts.
