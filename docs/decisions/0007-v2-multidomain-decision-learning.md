# ADR 0007 — Separate task semantics in a shared decision framework

Status: accepted research direction from the user's v2 review and follow-up;
implementation pending. Date: 2026-10-18 local time.

## Context

Ordivo v1 learns exact synthetic algorithm winners. The user's latest
clarification positions Ordivo as an MCDM-focused decision intelligence
framework, applicable across domains and using general decision models underneath.
Expedia and varied real-world data remain part of the research scope.
Relevance grades, stated choices, actual behavior and solver outputs have
different meanings; fitting one dataset or pooling their labels would not
establish the intended generality.

## Decision

Keep v1 immutable as a baseline. Introduce a common context/candidate inference
contract with typed source-specific supervision outside model inputs, source-
appropriate grouping, and task-specific evaluators. Use Expedia and ESCI first,
then MSLR and Swissmetro; optional explicit-ranking data broadens supervision.
Implement specialist baselines before neural adaptation. Evaluate one shared
model with and without MCDM pretraining, both within domains and with entire
domains held out. Exact MCDM is both an explicit-weight baseline and a serving
analysis component where its numerical domain is applicable. Add deterministic
feasibility, Pareto, sensitivity and traceable explanation boundaries. The neural
layer estimates preferences/context; it cannot override mandatory constraints.
Study GRPO and calibration-focused RLCD-inspired training after supervised
baselines, without claiming unverified compatibility or proprietary reproduction.

## Consequences

The synthetic renderer/validator cannot be reused unchanged for behavioral
targets. Raw ranking scores do not automatically become probabilities or
universal utilities. Splits differ by query, respondent, official fold and
domain. Additional source adapters are expected; a separate model per source
is insufficient as the final result. Data, testing and training remain remote.
Existing W&B history and v1 release identifiers are preserved.
General-purpose describes domain coverage of the intended component; it does
not replace the framework's specialized multi-criteria decision methodology.

See the [audit](../../reports/V2_RESEARCH_AUDIT.md),
[design](../superpowers/specs/2026-10-18-ordivo-v2-design.md), and
[implementation plan](../superpowers/plans/2026-10-18-ordivo-v2-baseline.md).
