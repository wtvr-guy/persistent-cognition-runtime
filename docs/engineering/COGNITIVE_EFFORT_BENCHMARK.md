# Cognitive effort and energy benchmark (COG-EFFORT-001)

**Status:** v1.1 experimental, local-host measurement; no production cognition is changed.

This benchmark asks how much wall time, CPU work, memory pressure, local-model
work, and *if an external electrical meter is available*, actual energy are
needed for different **measured workloads**. Passing a microbenchmark is not
equivalent to full PCR acceptance. Do not use these synthetic cases alone to
select production strategies or claim system-wide energy reductions.

## Workloads and their limits

| Phase | Execution | Oracle | Interpretation |
|---|---|---|---|
| idle_baseline | Sleep with resource sampler running, before and after | Always succeeds | Measurement overhead, background drift; not zero-watt power |
| deterministic_classifier | Actual PCR model_admission.classify_task, in a bounded batch | 16 frozen code/general cases | Cheap rule-based task classification; batch timing, divide by iterations |
| lexical_retrieval | PCR synthetic_benchmark.run_benchmark on jordan_vale_v1.json | Frozen 15/18, 100% unknown abstention | Real deterministic in-memory retrieval, not PostgreSQL JIT |
| associative_retrieval | PCR associative_benchmark.run_associative_benchmark | Frozen 18/18, 100% unknown abstention | Real deterministic associative recall, not the production persistence pipeline |
| ollama_unload (optional) | POST /api/generate with keep_alive=0 | Verified not resident through /api/ps | **Explicitly unloads target model**; overhead measured separately |
| ollama_cold (optional) | First bounded Ollama classification after verified unload | Same C02 input as paired warm | Includes model load; request duration only |
| ollama_warm_paired (optional) | Identical C02 request, immediately after cold | Same expected answer | Paired cold/warm comparison, model resident |
| ollama_prewarm (optional) | Empty generation and residency check | Verified resident | Cost of warming up when cold runs are disabled |
| ollama_warm_classifier (optional) | Ollama structured classifier over same 16 cases, shuffled with fixed seed | Exactly the same fixture oracle | Model capability and cost; **direct local Ollama**, not PCR source policy |
| native_ollama_acceptance (opt-in) | Existing full pytest -m ollama suite | Existing acceptance assertions | End-to-end real-model gates; requires a disposable TEST_DATABASE_URL |

The direct local Ollama workload uses a closed JSON field, temperature zero,
2048-token context and a 40-token completion cap. Inputs are synthetic, and no
models are downloaded or installed. It does not make a remote LLM call.

The initial comparison deliberately includes an easy classification problem.
It can demonstrate the expense of unnecessarily using a language model; it
cannot establish that a rule-based classifier solves difficult semantic tasks.

## Requirements

Use the benchmark branch and *its exact commit* on the Windows laptop. The
project requires Python 3.14+, uv and its pinned dependencies. PostgreSQL
is **not** required for the default workloads. Ollama is required only with
-WithOllama. For cold runs, stop other users of the target model; the opt-in
cold workload explicitly unloads it and changes its residency.

PowerShell in the repository root:

~~~powershell
git fetch origin bench/cognitive-effort-energy-v1
git switch bench/cognitive-effort-energy-v1
uv sync --frozen
uv run --locked pytest -q tests/test_cognitive_effort_benchmark.py
.\scripts\run_cognitive_effort.ps1
~~~

The default run includes baseline, deterministic classification and both
retrieval mechanisms. It touches neither your production PostgreSQL database
nor your PCR canonical event ledger. Each run uses a fresh, ignored output
directory and does not rewrite any production settings.

When that passes, run the local-model stages:

~~~powershell
.\scripts\run_cognitive_effort.ps1 -WithOllama -ColdRepeats 2 -WarmRepeats 2
~~~

The cold option unloads only the configured target model, never other named
models. Without -ColdRepeats, the runner prewarms the model and measures that
cost separately. It verifies model residency before measured warm cases, and
refuses to label a request as cold when the model was not verified unloaded.
A busy multi-user Ollama server will compromise comparisons even when this
check succeeds. Close competing applications and use wall power if possible.

Optional native PCR acceptance, after configuring a **disposable** test database
with its schema:

~~~powershell
$env:TEST_DATABASE_URL = "postgresql://USER:PASS@localhost:5432/pcr_test"
.\scripts\run_cognitive_effort.ps1 -WithOllama -WithNative
~~~

**Caution:** native tests may delete/replace data in TEST_DATABASE_URL. The
runner rejects a missing URL, a DB name without "test" or "benchmark", and
the same host/database as DATABASE_URL. These are basic safeguards, not
permission to use any important database. Run this only against a freshly
created, disposable database. An explicit native test does not rewrite
PCR's current source policy. It produces an exact stdout/stderr artifact.

## Evidence and audit files

Each run is written under:

    benchmarks/generated/cognitive_effort/YYYYMMDDTHHMMSSZ-<random>/

Files include:

- manifest.json: commit, branch, dirty state, environment, host, frozen fixture
  hashes, run options, model name and stage limitations. Credentials are not
  intentionally recorded; optional test URL is stored only as a digest and DB name.
- events.jsonl: fsynced SHA-256-linked run/stage/residency/artifact events.
- cases/<trial-id>.json: exact synthetic input, expected and actual result,
  oracle details, monotonic duration, UTC time boundaries, hardware pressure
  samples, model inference counters, and complete Python exception traceback.
- native_output.json: only when optional native acceptance runs. Its complete
  stdout/stderr may contain sensitive local diagnostics, so **do not publish it
  automatically**.
- summary.json and summary.csv: derived, reproducible reports; not authoritative
  evidence. They can be regenerated from immutable trial artifacts.

Verify an existing run:

~~~powershell
uv run --locked python -m benchmarks.cognitive_effort.runner verify --run-dir "benchmarks/generated/cognitive_effort/RUN_DIRECTORY"
~~~

A modified or missing artifact, broken event chain, incomplete trial, or
interrupted run is rejected. An execution failure still has an inspectable
case artifact. Inspect events.jsonl for trial_start, trial_end, and run_error;
the case JSON contains the Python traceback, model response and measured
resource context. All fixture failures are recorded as **FAIL**, not silently
corrected, relabeled or retried.

The suite **does not guarantee OS-level forensic integrity**: a privileged
actor who rewrites both the evidence and chain could forge a new chain. For
stronger custody, copy the final chain-head digest to another trusted location.

The generated directory is gitignored. If a run fails, **keep that entire
run directory**, not just a clipped console log.

## Electrical energy (optional; external meter required)

psutil measures CPU time, usage, process RSS and host memory, but not
electrical power. Ollama's token and duration counters also **do not measure
joules**. The runner therefore leaves energy fields null unless a meter
exports watts with wall-clock timestamps.

Use a power meter for the *entire laptop or power domain* capable of exporting
CSV; ideal wall-power data should be sampled at roughly 1 Hz or better.
Synchronize the meter clock to the laptop's UTC clock and record power
continuously from before the idle_start phase through after idle_end. CSV:

~~~csv
timestamp_utc,power_watts
2026-10-12T18:00:00.000Z,22.1
2026-10-12T18:00:01.000Z,22.3
2026-10-12T18:00:02.000Z,25.7
~~~

The times above illustrate format only and are **not** sample measurements.
Once the benchmark and meter recording have finished, export the meter file
to disk and run:

~~~powershell
uv run --locked python -m benchmarks.cognitive_effort.runner report --run-dir "benchmarks/generated/cognitive_effort/RUN_DIRECTORY" --power-csv "C:\path\meter.csv" --meter-scope wall
~~~

Each trial's joules are calculated using trapezoidal integration of power
in watts over **exactly that trial's timestamped start/end interval**.
The report refuses uncovered windows and sample gaps over five seconds
(configurable using --max-gap-seconds). In that case electrical energy
is UNAVAILABLE, not guessed. The CSV's SHA-256 fingerprint is retained
in summary.json; preserve the original meter file too.

Optionally include an independently measured idle wattage value:

~~~powershell
uv run --locked python -m benchmarks.cognitive_effort.runner report --run-dir "benchmarks/generated/cognitive_effort/RUN_DIRECTORY" --power-csv "C:\path\meter.csv" --meter-scope wall --idle-watts 22.2
~~~

This estimates incremental joules by subtracting idle watts times trial
duration; negative incremental values are preserved as a diagnostic, not
silently rounded to zero. The two idle_baseline trials are available to
estimate baseline watts, detect drift and assess sampling overhead. Idle
must be measured using the **same meter and comparable system conditions**.

If your meter reports CPU package power rather than wall power, use
--meter-scope cpu_package and do not interpret it as laptop-wide energy.
Battery level percentages, TDP, processor utilization, RSS and Ollama
tokens must never be reported as measured electric power.

On battery-powered systems, wall power from the laptop adapter may not
accurately represent actual system consumption while a battery is charging
or discharging. Prefer stable plugged-in, fully charged conditions,
and note battery/thermal/background activity in experimental notes.

## Interpretation and follow-up

1. Compare only equivalent cases, especially the paired C02 cold/warm trial.
   For deterministic classifier batches, divide duration by iteration count
   when comparing one classification against one model request.
2. Compare accuracy first, before any efficiency claim. A failed classifier is
   not a cheaper valid solution. The 16 cases are deliberately limited.
3. Compare medians across repeated trials; record idle drift, thermal state,
   battery state, model residency, and competing applications.
4. Separate model loading, prewarming, and execution. A warm resident model
   costs RAM even if no request is running; don't double-count resident pages
   as incremental per-request RAM.
5. Treat process RSS as an approximation which can double-count mapped pages.
   Ollama CPU is found by process name and might miss runner processes.
6. Report total elapsed times and inference counters, but never infer joules
   from token counts or CPU percentage.
7. If native tests fail, inspect native_output.json and the PCR-specific
   artifacts under native_artifacts. Native test stdout and full case
   traceback are the immediate debugging path.

The current pilot does not justify production strategy changes. The next
benchmark revision should add equivalent *production worker* tasks under
the same resource probe and exact oracle, optionally including trained
tiny classifiers against existing specialist decisions, before introducing
a cost-aware cognitive governor.


## v1.1: production policy, shared monitoring, automatic evidence ZIP

The deterministic baseline runs without Ollama or PostgreSQL and adds a
production_precheck phase using PCR's real
response_policy.explicit_prior_assistant_reference function. Its five frozen
cases are copied from existing native response-policy acceptance tests.
This tests only the deterministic forced-mixed-scope fast path. It is **not**
a substitute for the full semantic source-policy specialist.

The Windows -WithOllama invocation now runs --with-production by default
(unless -SkipProduction is specified). That calls PCR's actual
UserPromptLLM._response_policy, including validation and retry semantics.
The same five frozen cases have explicitly specified evidence_scope,
answer_kind and surface_mode expectations. The public worker interface
returns a validated policy, not the underlying model transport response;
therefore raw token statistics are unavailable for this phase, not invented.

Unlike v1, the run uses **one background sampler** instead of creating and
joining a monitoring thread on every case. Python process_time_ns and
perf_counter_ns are recorded around each workload. Host-wide and named-Ollama
CPU/RSS remain approximate sampled observations, and sub-millisecond cases
often have no in-window sample. Those intervals are reported as uncovered,
not zero. A new host_samples.json artifact retains the full series and the
sampler's own scan wall-time overhead.

The summary now includes actual versus expected classifications, actual
structured model output and token counts when available, and trial artifact
locations. This makes it easier to diagnose the recurring C03, C10, C12 and
C15 wrong-enum failures identified from the first run.

### Automatic ZIP packaging

After every complete run, even one containing model classification failures,
the runner automatically creates:

    benchmarks/generated/cognitive_effort/RUN_ID/evidence.zip

The ZIP includes the full cases subdirectory, the audit chain, run manifest,
host observation series, summary JSON/CSV, and any native output. An interrupted
run also attempts to produce a diagnostic ZIP of available evidence. Member
hashes are recorded in BUNDLE_MANIFEST.json, independently verifiable with:

~~~powershell
uv run --locked python -m benchmarks.cognitive_effort.runner verify-zip --zip "benchmarks/generated/cognitive_effort/RUN_ID/evidence.zip"
~~~

Upload **only evidence.zip** to supply the complete run. No manual compression
or separate cases upload is necessary. Note that raw native stdout, model
inputs and case artifacts can include sensitive operational data.

### Windows rerun

~~~powershell
git fetch origin bench/cognitive-effort-energy-v1
git switch bench/cognitive-effort-energy-v1
uv sync --frozen
uv run --locked pytest -q tests/test_cognitive_effort_benchmark.py
.\scripts\run_cognitive_effort.ps1
.\scripts\run_cognitive_effort.ps1 -WithOllama -ColdRepeats 2 -WarmRepeats 2
~~~

Use -SkipProduction to omit actual PCR source-policy worker comparisons.
This remains an exploratory suite: the five native policy oracle cases
are not a sufficient held-out corpus for production strategy selection.
