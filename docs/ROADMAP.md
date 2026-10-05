# Evaluation and roadmap

## Phase 1 — Freeze the extraction boundary

Inventory the source implementation at a pinned commit. Identify reusable event, memory, worker, artifact, scheduler, and capability contracts. Record implementation gaps separately from intended invariants. Preserve attribution and licensing.

## Phase 2 — Extract a runnable vertical slice

Demonstrate percept ingestion, canonical persistence, bounded retrieval, one stateless semantic task, durable result publication, and restart recovery. Publish reproducible setup and a small synthetic dataset.

## Phase 3 — Validate failure and recovery

Test interrupted workers, controller restart, database reconstruction from independent artifacts, stale projections, invalid model outputs, dependency failures, simultaneous events, and uncertain external effects.

## Phase 4 — Evaluate useful cognition

Freeze cases for corrections, cross-session recall, zero-overlap paraphrases, distributed evidence, contradictory reports, legitimate unknowns, and unfinished work resumed after restart. Separate retrieval failure, evidence admission failure, semantic decision failure, and response generation failure.

## Phase 5 — Measure modest-hardware operation

Record CPU/RAM use, model load time, context size, latency, history size, retrieval cost, disk growth, and recovery time. Compare matched tasks and hardware against a simple transcript-based agent and a retrieval-based baseline.

## Experimental rule

Freeze a baseline, change one mechanism, rerun the same experiment, and retain the change only when evidence justifies its complexity. Preserve negative results.

Report exact code revisions, model/backend versions, dataset versions, hardware, budgets, and raw outcomes. Synthetic benchmark success does not establish general accuracy. Bounded model context does not establish bounded total storage or constant runtime cost.

No standalone benchmark results are claimed by this scaffold.
