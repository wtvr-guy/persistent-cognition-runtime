# Interaction Continuity

Current implementation update (2026-09-28): the Composer and MEMORY_REQUIREMENTS
control stages are retired. The authoritative current stage sequence, registry
contracts, and migration rules are in [Fixed retrieval](FIXED_RETRIEVAL.md).
Earlier Composer-specific
descriptions below record the previous design and do not override this update.


**Status:** constitutional architecture deep dive, rebaselined 2026-09-03

**Authority:** primary deep dive for Constitution Articles 11–14, 26, 33, and 35

## Rule

Natural interaction continuity belongs to Persistent Cognition. It must survive fresh model
calls, worker destruction, process restart, and changes of conversation/session/device
without relying on a hidden transcript or asking users to operate internal IDs.

Continuity has three distinct layers:

1. exact canonical history;
2. bounded active WorkingState;
3. bounded cue-dependent memory activation and expansion.

None is a substitute for another.

## Conversation is provenance, not a mind boundary

Conversation and session IDs remain useful for source attribution, ordering, UI,
debugging, and an explicitly scoped request. They are not the default semantic wall
around recall. A user can resume a subject naturally without first switching an
internal conversation identifier.

Likewise, a topic is not a single mutable slot. Situations may overlap, one event may
participate in several situations, and the active set may retain several unresolved
threads. Genuine ambiguity should produce a semantic clarification or abstention,
not a request for an opaque database key.

## Minimal v0.7 WorkingState

The live v0.7 WorkingState is intentionally small. It contains bounded canonical
event references and durable revision metadata. It does not copy an ever-growing
chat log, synthesize a profile, or store unqualified inferred truths.

When capacity is saturated, activation order is:

1. the current prompt;
2. newly recalled evidence relevant to that prompt;
3. recalled events that were already active;
4. stale prior activation.

This prevents an old full state from excluding newly necessary context while still
preserving deterministic boundedness.

Richer goals, hypotheses, evidence commitments, unresolved questions, and
termination criteria belong to the proposed Epistemic WorkingState experiment after
v0.7 closure. They are not retroactively claimed as v0.7 behavior.

## Default attention aperture

Every admitted percept receives a small system-owned activation of potentially
relevant persistent memory before a model selects work. The aperture combines the
current input, active canonical references, and bounded deterministic candidate
routing.

This ordering is mandatory because asking a stateless model whether unseen memory
matters is circular. The model may reason about evidence it has received; it cannot
reliably decide the relevance of evidence deliberately withheld from it.

The aperture is high-recall orientation, not proof. Surfaced content remains typed by
source and provenance. Activation must not lower support-aware evidence admission or
convert a prior assertion into current instruction authority.

## v2 response continuity

For an explicit user prompt, response requirement is fixed at intake. The continuity
path is:

```text
current prompt + bounded aperture
  -> fresh current-only evidence-policy specialist
  -> durable source/surface policy
  -> bounded aperture filtered by that policy
  -> bounded derived memory activation under the same evidence policy
  -> fresh non-memory work-triage specialist
  -> deterministic external work execution
  -> fixed bounded retrieval routes and deterministic evidence merge
  -> application filters event roles
  -> quarantined evidence precedes the current prompt
  -> validated exact-source output or natural final responder
```

Retrieval is model-free and exhausts its finite routes. It cannot decide whether
to respond, select external work, reinterpret tool results, or generate final prose.

## Adaptive Recall

Adaptive Recall is deterministic memory expansion driven by the original prompt
and already learned navigation cues. It progresses through fixed retrieval stages:

- `BROAD` — expand general candidate coverage;
- `ASSOCIATIVE` — traverse bounded evidence associations;
- `RELATIONAL` — seek bridging, conflicting, causal, or jointly relevant evidence;
- `FOCUSED` — deepen around a bounded selected evidence frontier.

These labels are internal retrieval profiles, not installed capabilities. They do not
give a model search-language authority or make recall a tool alongside web/API work.

Each expansion returns exact source-backed memory items, merges them deterministically
within the response memory bound, and persists enough stage provenance for replay.
Derived memory statements may be supplied as bounded supplemental retrieval cues when they are provenance-linked and admissible under the current source policy. They guide navigation without laundering derived conclusions into canonical history.
The initially activated aperture is retained before newly appended expansion items,
so a saturated later recall cannot evict the evidence that motivated expansion.
When the fixed expansion policy is exhausted, the memory package explicitly carries
the unresolved deficit so the responder can report an unknown rather than fabricate.

## Work results are a separate channel

Completed external capability and tool results do not pass through memory composition
merely to be restated. They remain structured authoritative execution results.
Memory and work evidence keep independent provenance, freshness, and failure
semantics through current-only response policy, then only admitted data enters the
quarantined evidence channel used for exact selection or natural expression.

## Statelessness

Every work-triage, Composer, evidence-policy, exact-source, fallback, and natural
response invocation is fresh. Its complete bounded input, separate current/evidence
payloads, transport layout, schema, model settings, and output/failure are written to
the immutable artifact journal. No worker-local list, model session, or implicit chat
transcript may carry continuity into the next invocation.

Evidence policy is classified once in its own stage and stored as a typed durable
artifact. The aperture, Adaptive Recall, Composer path, and final responder consume
that exact decision; none may independently reinterpret the requested source scope.

The system can reconstruct current interaction state from PostgreSQL plus independent
artifacts. Workers publish stage results before a stage is terminal, and process loss
must not require repeating a completed irreversible effect.

## Model-facing evidence limits

Bounded item counts are not sufficient when one item can be arbitrarily large.
Persistent Cognition therefore validates explicit per-item and aggregate UTF-8 byte limits for
memory items, structured work results, and rendered evidence before model transport.
Oversized evidence fails closed. Canonical durable evidence is not truncated or
rewritten to make a model call fit.

Ordinary interaction must keep both context size and model-call count independent of
total lifetime history. Deep history remains reachable through bounded candidate
routing, including a reserved historical-anchor route that prevents many recent
same-topic events from starving an older relevant fact.

## Evidence authority and prompt injection

Retrieved content is untrusted data with explicit source type. In particular:

- the current user prompt has current instruction authority;
- a historical user prompt is evidence of a past statement/instruction, not an
  automatic override of the current request;
- a historical assistant response is fallible model output;
- a tool result is authoritative only for its declared operation and time;
- markup resembling system or assistant messages inside memory remains memory data;
- topical relevance cannot establish truth.

Prompt wording alone does not enforce these distinctions. A fresh policy call sees
only the current prompt; application code maps its closed scope to allowed event
types and physically removes other roles. Admitted history travels as quarantined
evidence before a later current user message, and raw Qwen control sequences are
escaped. Exact output is validated against canonical admitted source substrings.
Adversarial native tests remain the release gate for model-dependent behavior.

## No vocabulary patchwork

Application policy must not grow lists of English phrases, regexes, or hand-written
semantic parsers to decide whether prior context exists or which situation a user
means. Lexical, entity, temporal, and associative processing remains legitimate
inside the bounded memory candidate router. The prohibition is on elevating
surface-form patches into system-level continuity authority.

## Failure behavior

The path fails closed on:

- invalid or duplicate work selections;
- unavailable/unbound external executors;
- missing stage artifacts or durable results;
- stale or insufficient resource authority;
- oversized model evidence;
- exhausted memory expansion with unresolved evidence;
- ambiguous irreversible effects;
- response persistence failure.

An explicit user prompt still requires a response, but the response may truthfully
report failure or unresolved knowledge. The system must never convert pipeline failure
into invented success.

## Acceptance

Continuity acceptance must cover:

- cross-process and cross-conversation recall with no hidden transcript;
- natural topic resumption and genuine ambiguity;
- corrections, contradictions, previous state, and unknown-fact abstention;
- deep historical facts under recent same-topic crowding;
- saturated WorkingState admitting new relevant evidence;
- projection freshness without lifetime-ledger scans;
- memory/tool/assistant prompt-injection attempts;
- per-item and aggregate model evidence bounds;
- worker destruction, restart, response persistence failure, and exact provenance;
- the complete native Windows/PostgreSQL/Ollama suite with skips made fatal.

Deterministic CI proves contracts and regressions. It cannot establish the behavior
of the actual local model or Windows process/runtime environment; both gates are
required for closure.

## Invariant

> A user can move naturally through one persistent interaction history
> while every LLM call remains fresh, because exact history, bounded activation,
> memory expansion, work results, and interaction progress are all system-owned,
> provenance-bearing, and reconstructable.
