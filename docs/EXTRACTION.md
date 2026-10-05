# Engine extraction — 2026-10-05

## Source

Copied from Prometheist commit `106bb22be4ad60f2455ece8bc8c4e2806225d0fe`, the sequential model-residency fix based on main `d56517bf8eaf548b7d6e3b39e3d79549ad46c721`.

The original v0.7 closure tag was inspected as historical context. This extraction uses the later engine because the requested scope includes subsequent improvements: v2 workers, fixed deterministic retrieval, canonical neighbors, learned-memory source links, semantic facts, compact journals, model parameter/routing contracts, and sequential model unloading with peak-memory admission.

The source's later fixed retrieval update replaces Composer control. Restoring the old Composer loop would discard an engine improvement rather than preserve v2 compatibility.

## Boundaries and adaptations

The source application is unchanged. This repository has its own history and the same exact LICENSE file.

- Retained the `prometheist` import namespace, durable IDs, prompts, protocols, and environment names for compatibility.
- Renamed the installed distribution to `persistent-cognition-runtime` and exposed `pcr` and `pcr-percept` commands.
- Extracted model settings/catalog/probe/locking into headless modules. The headless chat job retains model admission and verified sequential residency.
- Removed CLI app/profile commands, automatic sensor startup, GUI/file/firewall job actions, and deployment-only SQL identity binding.
- Removed Android/node/tunnel services, sensor discovery, OS-security controls, web assets, GUI servers, imprint setup, and application mission documents.
- Retained source-grounded learned-memory structures because they supply canonical retrieval evidence; person-fidelity runners and subjective scoring are excluded.
- Preserved frozen synthetic retrieval fixtures and their assertions. Test-only fixture loaders are separated from production code.
- Removed only schema/constraint registrations belonging to excluded modules; all remaining numerical bounds are still audited.
- Kept current technical specs and engineering rules, with absent historical references linked to the pinned upstream source.

`SOURCE_BASELINE.json` records **220 byte-identical source files and 28 adapted files**, with source and extracted digests. Adaptations and exclusions are explicit. Runtime behavior is tested with the retained regressions; source parity alone is not a functional acceptance result.

## Verification

Local extraction checks:

- Locked Python 3.14.7 environment installed successfully.
- Ruff passed.
- Registry resolution passed.
- Constraint audit: 317 discovered, 317 registered; zero uncovered, stale, mismatched, or invalid entries.
- Pure engine tests: 97 passed.
- 589 tests collected.
- Deterministic constraint calibration completed; provisional/insufficient-discrimination results remain provisional.
- Source and wheel distributions built; both command help paths loaded successfully.

Full PostgreSQL CI is being run on the extraction branch; its result will be recorded before completion. This workspace cannot start a normal unprivileged PostgreSQL server, so no local database test pass is claimed. Real Ollama and native Windows acceptance remain separate from hosted deterministic CI; no model-quality improvement is claimed by this extraction.
