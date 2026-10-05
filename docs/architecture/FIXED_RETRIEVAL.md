# Fixed retrieval

Status: current production control path; retrieval quality remains experimental.
Policy: `fixed-retrieval/v1`. Introduced 2026-09-28.

The active path is:

1. Resolve references and classify the current prompt's source/surface policy.
2. Open the bounded attention aperture; activate derived self-context.
3. Select and execute any registered non-memory work. Empty catalogs need no model call.
4. Retrieve canonical self-model support/opposition roots where applicable and one bounded canonical neighborhood. Cues match learned statements, tags, and their admissible linked source text, so paraphrasing during learning does not erase the original navigation cues.
5. Run BROAD, ASSOCIATIVE, RELATIONAL, FOCUSED in that fixed order. A route without its required focus roots is explicitly recorded as skipped. An empty result from one route does not stop later routes.
6. Merge bounded route results deterministically, deduplicate event IDs, enforce source types and the exclusive history cutoff, alternate newer/older evidence within routes, and interleave routes and known evidence roles. Record budget exclusions without truncating or deleting canonical events.
7. Give the final responder admitted memory, separately labeled derived self-context, and authoritative capability results. Persist the response and final disposition.

The retrieval stage performs no LLM calls. It stops after exhausting the configured finite routes. It uses the original prompt and existing derived navigation hints, never a generated `memory_deficit`. Packet size and UTF-8 byte bounds remain the existing governed tunables. The initial implementation deliberately spends the bounded route budget even when an earlier route appears useful. Selective retrieval is a later experiment.

`ResponseMemoryPackage` records retrieval policy, route receipts, recall count, and stop reason. It has no sufficiency flag, requirements list, coverage decision, or Composer rounds. Retrieval completion establishes only that the configured searches ran. The responder must still acknowledge unknowns; a nonempty packet is not proof that an answer exists. Historical requests with no admissible evidence retain the explicit abstention path. Completed tool results bypass retrieval and remain available even when memory is empty.

`PerceptLLM` contains transport only. Semantic contracts live in `PerceptSpecialists`, and guarded worker processes remain restricted to one registered role. There is no new retrieval LLM.

## Registries

- `prompt_registry.py`: immutable prompt definitions and content hashes.
- `contract_registry.py`: active stage capabilities, allowed model roles, schema references, schema/prompt hashes, and links to the external capability registry.
- `capability_registry.py`: executable/discoverable capability contracts; internal memory stages are not selectable external work.
- `benchmarks/constraint_registry.json`: classified numeric bounds and calibration obligations.
- `schema.sql`: idempotent database bootstrap for canonical events and derived state.

`uv run python scripts/audit_registries.py` resolves the entire live contract catalog without a DB or model. CI runs this and the numeric constraint audit. LLM artifacts retain actual prompts/schemas and registered contract identities. Dynamic personality text and actual schemas remain visible in the invocation envelope.

The former Composer contracts remain in upstream Git history. They are absent from this engine’s active stage and prompt registries.

## Self-model epistemic policy

The proposal worker suggests evidence-linked interpretations. The review worker identifies possible opposition in a separately retrieved packet. Review output contains `opposition_indices` and `rationale`, never an establish/contest/reject verdict.

Ordinary code resolves source indices, rejects invalid links, deduplicates canonical roots, and applies the existing support, context breadth, and opposition policy. Known opposition prevents establishment. AVOWED, OBSERVED, INFERRED, ASPIRATIONAL, NORMATIVE, and SOCIAL_ATTRIBUTION remain distinct perspectives. An established derived representation is not canonical truth. The numeric support/breadth thresholds remain provisional and require longitudinal validation.

## Protocol transition

New interactions use `interaction-fixed-retrieval-v3`; new situation tasks use `v0.8-situation-v3`. Old incomplete workers cannot be replayed under the new contracts. Finish them on their original revision, or inspect their preserved artifacts and start a new interaction. Do not reinterpret or erase historical packages. Completed canonical evidence and independently durable artifacts are preserved.

## Research discipline

Keep derived-memory learning fixed when isolating a retrieval change. Preserve positive and negative outcomes, exact source delivery, model-call counts, and latency.

Known-answer retrieval fixtures are development targets: inspect the expected
canonical roots and tune general mechanisms against them. Once a fixture has
guided a change, report it as calibration evidence. Use independent histories,
opposite preferences, paraphrases, distractors, contradictions, and temporal
cutoffs to challenge the same rule, then evaluate on withheld questions. Never
route by fixture IDs or copy an expected answer into production behavior.

Measure required-source recall, irrelevant-source inclusion, forbidden-source
violations, deterministic replay, latency, and bounded resource use separately
from final-answer quality. Full source delivery can establish a retrieval result;
it cannot by itself establish correct interpretation or final-answer quality.
