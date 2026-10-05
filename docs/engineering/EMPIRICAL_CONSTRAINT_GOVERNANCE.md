# Empirical Constraint Governance

**Status:** constitutional engineering deep dive.  
**Constitutional authority:** primary engineering authority for Articles 29 and 30 of [`../../CONSTITUTION.md`](../../CONSTITUTION.md). This document is subordinate to the Constitution where wording conflicts.

## Constitutional experimental method

Persistent Cognition treats architectural mechanisms as falsifiable claims.

> **Freeze a measurable baseline, change exactly one mechanism, rerun the same experiment, and keep the mechanism only if the evidence justifies it.**

Negative results are retained. A failure mode discovered during development becomes part of the permanent experimental record and, where practical, a frozen regression scenario before recalibration or replacement.

Retaining a negative result means retaining enough executable evidence to reproduce
it. Rejected production mechanisms must not remain on the production path, but their
exact experimental implementations, frozen inputs, environment/revision metadata,
raw machine-readable outputs, decision rule, and verdict remain in a clearly marked
non-production experiment location. Prose summaries and aggregate tables do not
substitute for candidate source code or raw results. If older exploratory code was
not preserved, the record must say that exact reproduction is impossible and label
any later reconstruction as a new version rather than silently equating it with the
lost implementation.

New infrastructure, retrieval mechanisms, models, orchestration layers, optimization solvers, heuristic policy, or other complexity must earn its place by repairing a frozen measured failure or materially improving a declared objective without violating harder invariants. “It is standard,” “it might help,” or “the current tests pass” is not sufficient evidence.

Semantic fixtures must have platform-independent identities. Text fixture digests are
computed after canonical LF newline normalization, and Git attributes force the
committed fixture bytes to LF. Raw native result artifacts remain byte-preserved
because their exact transport form is evidence. A verifier may accept a documented
legacy fixture digest that differs only by LF/CRLF representation, but it must prove
that equivalence and report that the legacy exception was used; it may not weaken a
fixture check to parsed-JSON equality or silently ignore an arbitrary hash mismatch.

## Rule

Persistent Cognition must not treat an arbitrary numeric bound as correct merely because it passes the tests that happened to exist when it was introduced.

Any numeric value that changes runtime behavior must be one of the following:

1. **Structural invariant** — follows from the representation or algorithm itself (for example, a sequence number is non-negative, a percentage is bounded by 0 and 100, or a selected index must exist in its application-owned catalog).
2. **External contract** — imposed by a protocol, dependency, operating system, model API, or other authority outside Persistent Cognition.
3. **Identifier collision bound** — chosen from an explicit collision-risk calculation rather than a performance benchmark.
4. **Empirical tunable** — quality/cost trade-off whose value must be selected by a reproducible benchmark.
5. **Safety tunable** — operational trade-off whose value must be derived from failure-injection/load evidence under an explicit safety objective.
6. **Environment-calibrated value** — depends materially on the host/model/runtime and therefore must be calibrated on the intended deployment environment rather than inferred from CI.

The machine-readable source of record is `benchmarks/constraint_registry.json`. A static audit prevents newly introduced behavioral magic numbers from bypassing classification.

## What “optimal” means

There is rarely one scalar objective that can honestly combine correctness, abstention, latency, memory use, throughput, and safety. Inventing arbitrary utility weights would merely move the arbitrary number one level upward.

Persistent Cognition therefore uses a **constraint + Pareto + lexicographic** decision procedure:

1. Reject candidates that violate a hard correctness, provenance, determinism, safety, or recovery invariant.
2. Compare the remaining candidates on every scenario family independently, not only on a pooled average.
3. Maximize the worst-performing scenario family first.
4. Then maximize aggregate correctness/recall/abstention metrics in a declared order appropriate to the subsystem.
5. Only when quality is indistinguishable under the benchmark evidence, prefer the candidate with lower resource cost, latency, context size, or complexity.
6. If the selected candidate lies on the edge of the explored search space, expand that edge and rerun. A boundary winner is not accepted as evidence of an optimum.
7. Preserve all negative results and the complete sweep output so a later change can be compared against the same evidence.

This avoids both “vibes” and arbitrary weighted scoring.

## Benchmark design requirements

Every empirical benchmark specification must identify:

- the exact constraint family being calibrated;
- the current value(s) and the search procedure;
- the null hypothesis/baseline;
- scenario families and relevant interactions;
- oracle truth that is kept separate from system inputs;
- hard invariants that no candidate may violate;
- primary and secondary metrics;
- a deterministic decision rule;
- exploratory versus holdout cases;
- scale/adversarial cases;
- required environment and hardware, when applicable;
- confounds and known coverage limits;
- the revision, model/runtime versions, and raw result artifact used for the decision.

A benchmark must include all materially distinct scenarios we currently know how to construct. When a new failure mode is discovered, it becomes a permanent scenario family before the parameter is recalibrated.

## Search-space rule

The initial search grid is not treated as authoritative. Search is iterative:

- finite integer constraints begin with values that span below and above the current setting and expand geometrically until quality has saturated or a hard feasibility boundary is reached;
- continuous thresholds/decays begin with a coarse sweep over their mathematical domain, then refine the non-dominated region;
- interacting parameters are swept jointly when changing one can reverse the ranking of another;
- any optimum at a tested boundary forces expansion/refinement before acceptance.

Thus the benchmark does not silently encode the answer through a narrow hand-picked range.

## Generalization controls

For deterministic benchmarks, scenario generators use fixed published seeds and separate exploratory and holdout partitions. The optimizer may inspect exploratory results; the final decision must also survive the holdout set.

For stochastic local-model or host measurements, sample size is determined from observed variance and the effect size needed to distinguish competing candidates rather than from a fixed arbitrary repetition count. Raw samples are retained. Repeated runs across cold/warm model state and representative host pressure are required.

## Constraint families

### Memory scoring and evidence admission

Must cover exact facts, paraphrases, sparse cues, distractors, confusables, polarity, causal-source questions, temporal/deictic references, cross-conversation recall, previous-state/supersession, contradictory facts, unknown-fact abstention, long histories, and high-overlap irrelevant evidence.

### Associative traversal and retrieval breadth

Must vary graph fan-out, graph depth, cycles, guarded relationships, disconnected distractors, multiple plausible anchors, candidate starvation, and scale. Quality is evaluated together with candidate/association work performed and latency.

### Working-state/context bounds

Must cover short and long conversations, interleaved topics, multiple simultaneously active facts, corrections, resumed tasks, accumulated capability results, and distraction pressure. Truncation must never silently remove required evidence in a passing scenario.

### v2 work selection and Adaptive Recall bounds

Must include prompts needing zero, one, several independent, and several dependent
non-memory work items; memory-sufficient, progressively expanded, and exhausted-memory
cases; and explicit-user versus non-user response policy. Measure completion
correctness, unnecessary work/model calls, Composer convergence, retrieval work,
direct work-result delivery, and failures to preserve the deterministic response
requirement. Memory retrieval stages must never appear as selectable capabilities.

### Scheduler service guarantees

Must exercise bursty and steady arrivals across every service class, dependencies, preemptible/checkpoint-only/atomic tasks, overload, long-running work, and recovery. Measure per-class queue-delay distributions, starvation, interactive latency, background progress, throughput, and preemption cost.

### Resource admission and process estimates

Must be calibrated on deployment-class hosts. Measure real CPU/RAM peaks and system responsiveness for cold/warm local-model inference, PostgreSQL, worker startup, concurrent non-LLM work, and background OS pressure. Headroom must be derived from observed demand/error distributions plus an explicit safety objective. CI may verify formulas but cannot substitute for native calibration.

### Worker leases/retries/timeouts

Must include normal, slow, hung, killed, restarted, DB-delayed, and temporarily resource-starved workers. Optimize false abandonment and needless delay subject to no duplicate irreversible effects and bounded recovery.

### Local-model retry/token/timeout bounds

Must use the actual supported local model(s), prompt/schema sizes, cold/warm state, malformed first outputs, and host contention. Measure schema success, truncation, retry recovery, latency distribution, and wasted inference.

## v0.7 evidence status

The v0.7 audit is fail-fast in CI and the 2026-09-03 closure candidate discovers
**166** production/operational numeric constraints. All 166 are explicitly registered
with exact expected values; a new constraint, stale registration, changed value,
invalid classification, missing benchmark, or missing required evidence makes the
static gate fail. This count is evidence for this candidate only and must be updated
if the audited source changes.

One previously audited constraint was eliminated rather than justified: derived association IDs no longer truncate SHA-256 to a 20-character prefix. Association projection version 3 uses the full digest, removing that collision-risk knob from the production design.

The current deterministic evidence is intentionally conservative:

- `MEM-SCORE-001`: **INSUFFICIENT_DISCRIMINATION**. The current scoring policy passes the frozen corpora, but thousands of materially different policies are quality-equivalent under present evidence.
- `MEM-BREADTH-001`: **INSUFFICIENT_DISCRIMINATION**. Packet limit 5 passes 31/31 frozen questions with complete required-evidence recall and abstention, but every tested limit from 3 through 24 is quality-equivalent; other breadth layers remain unresolved.
- `MEM-GRAPH-001`: **INSUFFICIENT_DISCRIMINATION**. The current 2-hop/0.85 baseline passes, but 120 explored hop/decay policies are quality-equivalent, including a one-hop alternative.
- `CAP-DISCOVERY-001`: **INSUFFICIENT_DISCRIMINATION**. All five materially different coefficient vectors tested preserve every current routing oracle.
- `SCHED-SERVICE-001`: **INSUFFICIENT_DISCRIMINATION**. Quarter-, half-, current-, double-, and quadruple-scaled wait vectors preserve the only currently measurable service-order invariant; exact values cannot be identified without workload-arrival data and queue-delay objectives.
- lexical/stemming heuristics remain **PROVISIONAL** because the current scoring benchmark does not independently vary them.
- `PIPELINE-NATIVE-001`, `RES-NATIVE-001`, `WORKER-NATIVE-001`,
  `OLLAMA-RUNTIME-001`, and `LLM-NATIVE-001` remain **NATIVE_REQUIRED** where
  local-model or target-host behavior is part of the quantity being measured.

The deterministic record is `benchmarks/results/DETERMINISTIC-CONSTRAINTS_2026-08-27_initial.json`. The scoring record is `benchmarks/results/MEM-SCORE-001_2026-08-27_initial.json`.

Native calibration is deliberately separate from release acceptance. On the intended Windows development host, collect pilot evidence with:

```powershell
.\scripts\run_native_constraint_calibration.ps1
```

That writes a timestamped `benchmarks/results/NATIVE-CONSTRAINTS_*.json` file containing host-resource samples, the frozen worker/runtime block duration/result, and controlled Ollama token-cap/latency observations. A pilot does not automatically verify a value; it establishes actual native data from which the required cold/warm/pressure/fault scenarios can be expanded if the evidence remains non-discriminating.

## Relationship to constitutional constants

Constitutional rules intentionally do not freeze most numeric implementation bounds.

Examples:

- “LLM context remains bounded” is constitutional; the exact token cap is classified here.
- “resource safety preserves headroom” is constitutional; exact RAM/CPU reserves are safety/environment tunables.
- “memory reassessment is fresh and bounded” is constitutional; the exact Adaptive
  Recall stage/item and Composer-call bounds are empirical or structural as recorded
  in the registry.
- “service guarantees prevent starvation” is a constitutional execution requirement; exact wait-cycle thresholds are empirical.

This separation prevents a provisional calibration from becoming permanent architecture by accident.

## Release rule

A release is not considered empirically calibrated merely because all functional tests pass. Before release, every runtime constraint discovered by the audit must be classified, every empirical/safety/environment tunable must have a benchmark specification, and every release-required calibration must have a result artifact tied to the release revision.

When current evidence cannot establish an optimum, the registry must say so explicitly. The accepted value is then a documented provisional bound, not falsely presented as optimal.

Testing and environment-evidence requirements are further defined in [`TESTING_AND_ACCEPTANCE.md`](https://github.com/wtvr-guy/persistent_cognition/blob/106bb22be4ad60f2455ece8bc8c4e2806225d0fe/docs/engineering/TESTING_AND_ACCEPTANCE.md).
