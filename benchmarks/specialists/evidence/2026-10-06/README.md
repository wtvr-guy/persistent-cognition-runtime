# Initial live specialist experiment — 2026-10-06

The benchmark branch is executable, but **this is not a completed comparison with
Qwen and is not a deployment qualification**. The final experiments below use
small fictional seed cases and provisional labels, through the real PCR worker
methods. No fine-tuning has been performed.

## Completed live inference

- Code revision: `eab8671e81c0539101491875c735b6c54e269acb` (clean when each final run started).
- ModernBERT-base-NLI: 149,607,171 parameters; checkpoint
  `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`; 21 worker cases per precision,
  repeated twice, including one deterministic bypass per repeat.
- Hammer 2.1 0.5B: 493,789,952 parameters; 11 worker cases per precision,
  one pass, including one deterministic bypass.
- Each model was actually loaded in FP32, BF16 and FP16. Dtypes, complete weight/
  config hashes, adapter source hashes and package versions are in the manifests.
- CPU inference, four torch threads, deterministic algorithms, eager attention.
  These measurements are from the execution host, **not the Windows laptop**.

All 150 semantic case executions produced worker-valid results. Six additional
NLI and three Hammer executions bypassed inference deterministically and do not
count toward model accuracy. Valid schema output still includes wrong decisions.

## Decision correctness on unique seed cases

The following **case counts**, not independent repeated samples, were identical
across the three precisions. Train/dev/test remain separate; labels are unreviewed.

| Worker | Train correct | Dev correct | Test correct |
|---|---:|---:|---:|
| V2_RESPONSE_POLICY | 5/6 | 1/2 | 4/5 |
| SELF_SCHEMA_REVIEW | 2/2 | 1/1 | 2/4 |
| PRECOGNITIVE_USER_PROMPT_WORK | 1/4 | 1/1 | 3/5 |

NLI's errors included the scope of remembered user facts, exact composition,
context-specific counterevidence, and multiple contradictions. It cannot be
considered a trained PCR specialist merely because it produces bounded logits.

BF16 changed the surface-mode decision for `policy-exact-compose` relative to
FP32 in both repeats (one unique case). FP16 agreed with FP32 on all final
NLI decisions. BF16 selected the expected composition mode, but the scope was
still wrong, so its full-case accuracy did not improve. Identical aggregate
accuracy therefore did **not** mean identical decisions.

Hammer selected an empty capability list on **every semantic case**, in all three
precisions. It got the negative cases right and all positive cases wrong under
this initial virtual `select_work` adapter. This is evidence of a zero-shot
contract-adapter mismatch to investigate before training; it is not a claim that
Hammer cannot learn tool selection. There were no precision decision differences
in this Hammer seed run.

## Median full worker latency (seconds)

Includes the actual worker validation/retry path, excludes model loading and
bypasses. Model loading and memory telemetry are recorded separately in manifests
and results. CPU FP16 was particularly slow for Hammer.

| Profile | Policy | Self review | Work selection |
|---|---:|---:|---:|
| modernbert-nli-fp32 | 0.456 | 0.054 | — |
| modernbert-nli-bf16 | 0.234 | 0.036 | — |
| modernbert-nli-fp16 | 1.768 | 0.137 | — |
| hammer-fp32 | — | — | 4.417 |
| hammer-bf16 | — | — | 2.021 |
| hammer-fp16 | — | — | 21.827 |

Process-lifetime high-water RSS was used on this container because its PID
namespace differs from its mounted procfs. That metric includes model loading
and framework overhead. Do not interpret it as weights-only memory or directly
compare it with Ollama's `/api/ps` memory field.

## Missing comparisons

- Qwen baseline: no Ollama service or installed `qwen3:4b-instruct` here. The saved
  preflight reports `UNAVAILABLE`; no baseline score was manufactured.
- FunctionGemma FP32/BF16/FP16: download returned a gated-repository 401. The local
  Hugging Face identity must accept the license and have access before testing.
- ModernBERT-base: no PCR-trained classification heads exist; its three profiles
  are explicitly blocked. A random head would not test the proposed specialist.
- Semantic triage currently has baseline cases but no trained encoder head.

Run the Qwen baseline and matched specialist profiles on the Windows laptop using
[the setup guide](../../../../docs/SPECIALIST_WORKER_PRECISION_BENCHMARK.md).
Use a larger independently reviewed corpus before fine-tuning or deployment.

## Preserved evidence

`all-runs.zip` contains the complete manifests, frozen case corpora, raw results,
logits/native outputs, retries, errors and summaries for the final runs, plus all
completed/partial development runs and preparation outcomes. `index.json` lists
file hashes and explicitly identifies the two final experiments. Extract the
archive to recover the original run-directory layouts for reviewed export.

Final runs:

- `20261006T190442Z-951c15de` — Hammer, all three precisions.
- `20261006T190938Z-5ad1e5e3` — NLI, all three precisions, two repeats.

Development runs include an incomplete cache, PID telemetry failures, a run with
unreliable PID-based BF16 memory readings, earlier-source comparisons, and an
interrupted Hammer run while its native list parser was corrected. Their records
are preserved for traceability and **must not be used as comparative model
scores**. The interrupted run can have a `READY` profile with no final summary.
The archive contains fictional seed evidence only, not personal production memory.

`inference-environment.txt` freezes the installed package versions. Inference
checkpoints are addressed by commit and file hash; weights are not vendored.
Only independently approved **train** labels may enter `export-reviewed`; teacher
answers and seed expected labels are not automatically promoted to ground truth.

## Verification

- 111 unit tests passed, including 14 benchmark/adapter tests.
- Ruff passed for all new Python files.
- Contract registry audit passed.
- Constraint audit passed with zero unregistered/stale/mismatched/invalid entries.
- Pinned extraction verification passed; upstream source hashes were preserved.
- The PostgreSQL-dependent full suite and Windows/Ollama acceptance were not run.
