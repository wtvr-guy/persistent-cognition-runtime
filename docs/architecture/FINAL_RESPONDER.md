# Final Responder Contract

**Status:** current architecture decision, revised 2026-09-12 after specialist split
**Applies to:** user-facing response generation after deterministic response policy, required work, and memory sufficiency have completed or exhausted

The pipeline separates current-only policy, application-owned evidence admission,
exact-source output, and natural-language expression into narrow specialist
contracts. The natural final responder is a fresh, disposable LLM invocation. It
owns neither Prometheist's continuity nor any control-plane decision.

This document refines the final-response section of `PERCEPT_TO_RESPONSE_PIPELINE.md` and must be read consistently with the Constitution, `SYSTEM_DETERMINISM.md`, and the evidence-authority rules of the percept-to-response pipeline.

## 1. Expression is not control

The final responder receives an already-committed requirement to respond. It may express the completed result, but it must not decide whether Prometheist should respond, retrieve additional memory, schedule work, execute side effects, change permissions, alter resource policy, or mutate durable authority.

Deterministic system control therefore remains at temperature `0.0` wherever an LLM is used for a bounded semantic control decision. A non-zero final-response temperature does not relax Article 15 because user-facing phrasing is not durable control authority.

## 2. Mandatory core personality and evidence contract

Every user-facing final responder receives Prometheist's core interactive personality prompt. This is not an optional style preset. It establishes, among other things:

- Prometheist is the persistent system; the LLM invocation is a disposable semantic worker;
- the current user prompt is direct current evidence;
- a historical `USER_PROMPT` is direct evidence of what the user previously said, asked, named, preferred, corrected, or instructed;
- model-authored historical responses are fallible and cannot negate conflicting user-authored evidence;
- supplied memory is provenance-bearing evidence rather than unquestionable truth;
- personal or history-specific facts must not be invented;
- unresolved memory deficits must remain unresolved rather than being filled by plausible fabrication.

These accuracy and evidence-authority rules are mandatory even when a deployment supplies a custom personality.

### 2.1 Current-only policy and physical source admission

Before historical retrieval, a fresh evidence-policy specialist receives only the
current user prompt. It selects from closed application-owned enums:

- historical evidence scope: user-authored, model output, external tool, system
  record, derived internal, mixed conversation, or general/current;
- response surface: natural language, exact source substring, or exact source
  composition.

The model does not filter memory. Application code persists the exact typed policy,
maps the selected scope to event types, and physically restricts retrieval and
removes inadmissible items. A question
about what the user previously said therefore cannot be answered from an assistant
assertion—and assistant assertions are excluded from retrieval by default rather than
competing for evidence budget.

Policy inference is deliberately current-only. Persisted prompt injection never sees
or influences the call that decides which persisted source roles are admissible.
Explicit references to prior assistant actions (e.g. "what you just ruled out" or
"what did you tell me") deterministically select `MIXED_CONVERSATION` scope.
Retrieval, composition, and response realization inherit this committed policy; the
final responder never classifies it again.

### 2.2 Quarantined evidence transport

Admitted memory and work results are sent before the current prompt in a separate
quarantined evidence channel. For chat transport this is a tool-role message followed
by the current user message. For raw Qwen Instruct transport it is a bounded
tool-response block followed by a later user block; ChatML/tool control sequences in
both evidence and current data are escaped.

Historical instruction-shaped text remains data. It cannot synthesize a system role,
replace the current task, alter the output schema, expand permissions, or modify the
application-owned capability catalog.

### 2.3 Exact output is source-extractive

When the current request requires an exact stored value or exact multi-field format,
Prometheist does not ask the expressive responder to respell it. A deterministic-
temperature selector chooses an indexed exact substring from the already-admitted
sources. Application code verifies the index and substring membership and returns
the canonical source bytes. Multi-field output joins validated values only with a
formatting-only separator copied from the current request.

An explicit unsupported-history fallback is selected in a separate current-only
call and accepted only if it is a verbatim substring of the current prompt. This
prevents excluded history from manufacturing its own fallback or output contract.

## 3. Configurable personality is additive

`PROMETHEIST_PERSONALITY_PROMPT` is an optional expressive extension. When configured, it is appended beneath the mandatory core prompt as a user-configured personality layer; it does not replace the core prompt.

This permits deployments to steer tone, voice, verbosity, humor, formality, or other expressive traits while preserving the same evidence and accuracy contract.

The exact resolved system prompt is persisted in the independent LLM invocation artifact so later inspection can establish which personality instructions materially influenced a response.

## 4. Response temperature

Prometheist separates control temperature from response temperature:

- control/decision kinds remain deterministic at `0.0`;
- user-facing response kinds use the configured response temperature;
- the current default response temperature is `0.65`;
- `PROMETHEIST_RESPONSE_TEMPERATURE` may override the response value only within the validated implementation range.

The current response temperature is an environment-calibrated LLM behavior parameter, not a constitutional invariant. It belongs under the `LLM-NATIVE-001` evidence obligation and may change when native evaluation shows a better accuracy/personality tradeoff.

A higher response temperature is never permission to alter evidence, invent remembered facts, ignore authoritative tool results, or contradict the current user prompt. Personality and variation operate only inside the set of responses supported by the available evidence.

## 5. Structured response envelope remains mandatory

Expressive sampling does not remove the constrained response envelope. Natural
responses use the application-owned structured answer contract, with thinking
disabled and bounded generation. Exact-source modes bypass free-form expression
after validated selection and mechanical composition.

The intended separation is therefore:

```text
deterministic/system-owned response requirement
        +
current-only source/surface policy
        +
application-filtered quarantined evidence
        +
validated exact-source output OR evidence for natural expression
        +
mandatory identity/evidence rules
        +
optional configured personality
        +
response-only sampling temperature
        ↓
validated user-facing language
```

## 6. Invocation provenance

Every response-stage LLM invocation artifact must preserve enough information to
reconstruct the exact invocation contract, including:

- architectural stage and semantic call kind;
- model and backend;
- exact system prompt, including resolved personality instructions;
- exact current user prompt;
- exact quarantined evidence payload and transport layout for evidence-bound calls;
- canonical event references for admitted memory evidence;
- structured-output schema;
- generation token cap;
- effective temperature;
- normalized output or failure information.

This makes personality and temperature observable causal inputs rather than hidden runtime state.

Native natural-response acceptance uses these references as its structural oracle:
the automated test proves that the fresh responder received the required canonical
evidence, while the printed prose is reviewed by a human for semantic accuracy and
expression. Exact-string assertions are reserved for genuine exact-output or closed
security/control contracts, not added artificially to ordinary conversation.

## 7. Acceptance obligation

Changes to the core personality prompt, response temperature, evidence-admission
policy, transport layout, or response-generation policy require regression evidence
that accuracy is preserved. Native local-model acceptance is required where model
behavior matters; deterministic CI alone cannot establish that a policy classifier
or source selector behaves correctly on the configured model.

The target is not deterministic prose. The target is **accurate, evidence-grounded Prometheist behavior with a recognizable configurable personality, while deterministic authority remains outside the responder.**
