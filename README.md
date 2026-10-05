# Persistent Cognition Runtime

**Persistent memory. Disposable reasoning. Durable execution.**

Persistent Cognition Runtime explores a local-first cognitive architecture in which continuity belongs to durable system state. Language models perform bounded semantic tasks through fresh specialist invocations; the runtime owns memory, scheduling, control authority, and recovery.

The practical goal is useful, continuous AI on modest hardware without requiring an ever-growing conversation context or a permanently loaded large model.

## Status

This repository is an architecture and research scaffold derived from the engineering work in [Prometheist](https://github.com/wtvr-guy/prometheist). It does not yet contain an extracted, independently runnable implementation. The source project contains the evolving implementation; its tests and experimental results must be assessed at their recorded revision.

This repository focuses on the underlying engineering architecture. Its documents describe design requirements and proposed evaluation, not a claim that every requirement has already been implemented or independently verified.

## Core design

- **Canonical history:** append-only events preserve original evidence and causal provenance.
- **Rebuildable memory:** indexes, associations, and generalized knowledge are derived projections.
- **Just-in-time context:** narrow memory requests produce bounded evidence packets.
- **Stateless specialists:** each LLM worker has one semantic responsibility and explicit typed inputs and outputs.
- **Deterministic control:** the runtime owns identifiers, scheduling eligibility, resource admission, and execution authority.
- **Durable work:** tasks, checkpoints, capability results, and dispositions survive worker failure.
- **Resource-aware execution:** admit work against measured CPU, RAM, and model requirements.
- **Independent artifacts:** meaningful boundaries remain inspectable and reconstructable outside the operational database.

A fixed context budget does not guarantee constant retrieval cost, perfect recall, or intelligence equal to a larger model. Those properties require separate measurements.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Invariants](docs/INVARIANTS.md)
- [Evaluation and roadmap](docs/ROADMAP.md)
- [Prior work and research approach](docs/PRIOR_ART.md)
- [Contribution guidance](CONTRIBUTING.md)

## License

[PolyForm Noncommercial License 1.0.0](LICENSE), identical to the source project's license and project-specific notices. Commercial use requires a separate license from the licensor.
