# Specialist worker precision experiment

This branch adds an **experimental harness**, not automatic production model
routing. The starting point is PCR main `582fdc245deeec976604df50aa5d56383a0f70cc`.
The default conversational responder and production contracts are unchanged.

## What is compared

| Profile | Worker methods | Status |
|---|---|---|
| Installed `qwen3:4b-instruct` in Ollama | Policy, work selection, triage, self review | Baseline; records the actual installed digest |
| FunctionGemma 270M FP32/BF16/FP16 | `UserPromptLLM.decide_disposition` | Native function-call adapter; requires gated weights |
| ModernBERT-base-NLI FP32/BF16/FP16 | `_response_policy`, `SituationLLM.review_self_schema` | Experimental zero-shot/pairwise NLI adapters |
| ModernBERT-base FP32/BF16/FP16 | Policy and triage | **Blocked** until PCR classification heads are trained |
| Hammer 2.1 0.5B FP32/BF16/FP16 | `UserPromptLLM.decide_disposition` | Optional research comparator; noncommercial license |

ModernBERT-base is an encoder, not an already trained PCR classifier. Loading a
random classification head and scoring it would not test the proposed specialist.
The NLI checkpoint offers an executable initial experiment, with different,
explicit adapters; it is not a replacement for training PCR-specific heads.
The triage baseline is captured now so its data can support that later work.

The 36 seed cases are fictional contract examples, **not production traffic**.
Their labels are provisional and marked `seed-contract-example/unreviewed`.
Scores are useful for finding bugs and designing the next experiment, not for
estimating deployed reliability. Add real cases and independently reviewed labels
before evaluating a production routing change. Self-proposal and exact-source
selection/composition are outside this first experiment.

## How the Python works

`invoke_worker(case, worker)` invokes the existing worker method. A recording
transport replaces only `_structured_with_evidence`. Stage guards, original
prompts, evidence quarantine, token caps, retries, Pydantic validation, capability
bounds, and source-policy ceilings still run. Self-review also checks related
evidence indices against the supplied packet before scoring. It does not run
PostgreSQL, schedule work, execute capabilities, or mutate durable identity.

For example, PCR's work selector still gets 48 tokens initially and 96 on retry.
Selecting index 99 from a three-item catalog is rejected by the real validator.
The failed answer, retry and validated answer are all saved. An empty admitted
catalog bypasses the model and is counted separately from model accuracy.

Tiny-model inference runs in a persistent subprocess with its own Python 3.12
environment. PCR retains its Python 3.14 requirements. One specialist is loaded
at a time. The subprocess receives frozen worker requests as JSON lines and
returns both native output/logits and the adapted worker result.

FunctionGemma receives the official native tool template with a non-executable
`select_work(capability_indices)` function. A closed parser accepts only that
function and an integer list; it never evaluates generated code or invents a
missing answer. Native prompts, tool declarations, output and token counts are
preserved. This is a zero-shot contract adapter that will itself need tuning.

The NLI policy adapter ranks fixed hypotheses for scope and surface independently.
These hypotheses are preserved with the logits. Self review compares each related
evidence item to the candidate, then Python selects contradiction indices by
argmax. Its rationale is deterministic diagnostic text and is excluded from
decision accuracy. **Pairwise NLI does not jointly reason over the support roots**;
context-sensitive failures are an expected limitation to measure, not hide.

## Run on the Windows development machine

Run from the repository root in PowerShell. The inference environment is separate
because the core project targets Python 3.14 and ML-library compatibility differs.

```powershell
git fetch origin
git switch experiment/specialist-worker-precision
uv sync --group dev
uv venv --python 3.12 .venv-inference
uv pip install --python .venv-inference/Scripts/python.exe -r benchmarks/specialists/requirements-inference.txt
```

For CPU-only PyTorch, install `torch` from its official CPU wheel index first:

```powershell
uv pip install --python .venv-inference/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
```

Do not alter the baseline model to make it fit this experiment. Start Ollama with
the **existing** `qwen3:4b-instruct` model you have been using. Preflight saves the
installed model digest and fails if that exact tag is unavailable:

```powershell
uv run python -m prometheist.specialist_benchmark run --profile qwen3-baseline --preflight
uv run python -m prometheist.specialist_benchmark run --profile qwen3-baseline --repeats 3
```

Download/freeze the NLI checkpoint explicitly. Preparation resolves a full commit
SHA; inference subsequently loads only cached files, never a moving `main` ref:

```powershell
.venv-inference/Scripts/python.exe scripts/prepare_specialist_models.py --model tasksource/ModernBERT-base-nli --output benchmarks/generated/specialists/frozen-nli.json
uv run python -m prometheist.specialist_benchmark run --matrix benchmarks/generated/specialists/frozen-nli.json --profile qwen3-baseline --profile modernbert-nli-fp32 --profile modernbert-nli-bf16 --profile modernbert-nli-fp16 --inference-python .venv-inference/Scripts/python.exe --repeats 3
```

For FunctionGemma, first accept the publisher's model license and authenticate
locally with Hugging Face (`hf auth login` in the inference environment). Then:

```powershell
.venv-inference/Scripts/python.exe scripts/prepare_specialist_models.py --model google/functiongemma-270m-it --output benchmarks/generated/specialists/frozen-functiongemma.json
uv run python -m prometheist.specialist_benchmark run --matrix benchmarks/generated/specialists/frozen-functiongemma.json --profile qwen3-baseline --profile functiongemma-fp32 --profile functiongemma-bf16 --profile functiongemma-fp16 --inference-python .venv-inference/Scripts/python.exe --repeats 3
```

The earlier candidate list also included Hammer 2.1 0.5B. Its three profiles are
optional research comparators, not production defaults. Prepare the checkpoint
with `--model MadeAgents/Hammer2.1-0.5b` and select `hammer-fp32`, `hammer-bf16`
and `hammer-fp16`. Its publisher's native tool template and a closed one-function
parser are used. The model has a noncommercial license; a later production
decision must account for that. The 1B xLAM candidate is outside the **under-1B**
precision experiment, as are the 1.5B Hammer, 1.7B Qwen and 3B Ministral candidates.

On Linux use `.venv-inference/bin/python`. Change CPU thread count or device only
by making a new matrix file; those settings are part of the experiment identity.
Use `--corpus path/to/cases.jsonl` for an expanded corpus. Keep all paraphrases,
catalog permutations and cases from the same underlying event in one `group_id`
and one split. Identical inputs crossing splits are rejected as well.

The default matrix intentionally contains blocked base-encoder profiles. Select
the executable profiles above. Missing checkpoints/runtimes are `UNAVAILABLE`
and exit with code 2; per-case inference or validation failures remain recorded
errors in a completed run and count against accuracy. No dtype fallback occurs.

## Precision and reproducibility

FP32, BF16 and FP16 are casts of **one immutable released checkpoint**. If its
released weights were BF16, casting them to FP32 cannot recover lost training
precision. This measures inference arithmetic and deployment precision, not
three separately trained models. No FP16/BF16 accuracy equivalence is assumed.

Every HF profile verifies actual parameter dtypes, records full checkpoint/config
hashes and package versions, uses greedy/argmax decisions and deterministic torch
algorithms, disables TF32, and does not silently truncate inputs. Hardware/kernel
differences can still change floating-point results; repeated runs and paired
decision comparisons measure that. Model-load time includes file verification.

Metrics are separated by worker **and train/dev/test split**: decision accuracy,
field confusions, selection false positives/negatives on valid outputs, first-call
and final worker validity, retries, median/p95 latency, and paired FP32-versus-half
precision decision changes. Deterministic bypasses are excluded from semantic
accuracy. Repeated copies are not independent examples or a confidence interval.

HF inference records process RSS (including framework/weights) and CUDA peak
allocated bytes when applicable. In containers where PID-based telemetry is
unavailable, it records process-lifetime high-water RSS explicitly. Ollama memory
is represented by its runtime `/api/ps` snapshots, **not a comparable measured
process peak**. This first harness does not measure TTFT or instrument the Ollama
service's peak RSS. Compare latency on the intended laptop before choosing a
precision; CPU FP16/BF16 support or speed must not be assumed.

## Preserve the data and prepare training

Each uniquely named run has four files: `manifest.json`, `cases.jsonl`,
`results.jsonl`, and `summary.json`. Results are flushed/fsynced after each case,
so an interruption retains completed examples. The manifest records source SHA,
dirty-worktree status, host, contracts, models and corpus hashes. Each result has
its own content hash. Requests include the real rendered worker prompt, evidence,
schema and token budget; logits/native outputs, errors and retries are preserved.

Use `--output` to choose a durable local experiment directory. Generated data is
ignored by git because real memory and worker evidence can contain personal data.
Back up the entire run directory together; moving only a summary loses the
training source. The public branch may contain explicitly selected fictional
seed-run evidence, not personal production journals.

Existing PCR journals can be copied into a candidate-label collection:

```powershell
uv run python -m prometheist.specialist_benchmark capture-journal --source .prometheist --output benchmarks/generated/teacher-invocations.jsonl
```

Use your actual `PROMETHEIST_ARTIFACT_ROOT` if different. Captures preserve the
original invocation envelope and mark **all teacher answers unreviewed**. An
accepted Qwen result is not ground truth. Captured envelopes alone may lack full
worker arguments (catalog, source policy, review evidence grouping); reconstruct
those from the same stage's artifacts into `WorkerCase.inputs` before claiming
an end-to-end worker replay. Do not guess missing evidence or provenance.

A review file is JSONL with one explicit approval per frozen case hash:

```json
{"case_sha256":"HASH_FROM_RESULTS","decision":"APPROVE","reviewer":"REVIEWER_ID","label":{"capability_indices":[0]}}
```

The label is the semantic output schema, not the later worker disposition.
For policy, include the scope/surface fields; for self review include the reviewed
opposition indices and a rationale. Reviews can correct a teacher answer; they do
not merely approve its schema. Record the actual reviewer, never a placeholder.

```powershell
uv run python -m prometheist.specialist_benchmark export-reviewed --run PATH_TO_RUN --reviews PATH_TO_REVIEWS --output benchmarks/generated/reviewed-training.jsonl
```

Export rechecks frozen corpus/result hashes, passes each label through the real
worker validators, deduplicates approved cases, excludes deterministic bypasses,
and rejects **all dev/test examples**. Its format is contract-neutral: it retains
the worker request, typed inputs, independently approved label and review/run
provenance. A later training job must map that to FunctionGemma's native tool
template or encoder labels. This branch does not train checkpoints or claim that
the resulting corpus is sufficient for a model trained from scratch.

The next gate is a larger independently reviewed corpus, then contract-specific
head/fine-tune training and a frozen held-out comparison with Qwen on the laptop.
Only after that comparison should the production registry route to a specialist.

## Verification

```powershell
uv run pytest -q tests/unit/test_specialist_benchmark.py
uv run ruff check src/prometheist/specialist_benchmark.py scripts/specialist_inference.py scripts/prepare_specialist_models.py tests/unit/test_specialist_benchmark.py
```

The tests use scripted inference only to check real validator/retry behavior,
policy limits, bounds, deterministic bypass accounting, native syntax parsing,
split leakage and reviewed-export integrity. They are not model-quality results.
