# ADR 0008: deterministic verification before controlled RL comparisons

Date: 2026-10-18. Status: accepted implementation direction from the user's
`Ordivo_v2_model_training.md` (original user file remains unchanged).

## Problem and alternatives

V1 predicts a synthetic winner; it does not resolve human preferences, enforce
arbitrary constraints or verify model explanations. V2 needs these distinct
boundaries before an RL reward can claim mathematical correctness.

A: Generate complete reasoning and trust its self-reported checks. Rejected:
self-report is not numerical evidence.
B: Replace the exact engine with neural scoring. Useful as a behavioral baseline,
but cannot provide exact formula guarantees or authoritative unknown weights.
C: Reuse reference-tested solvers behind a typed decision boundary; keep learned
utilities and uncertainty explicit. Selected: small, independently testable and
compatible with future backbones.

## Rulings and implementation scope

- Use SFT followed by independent calibration/RLCD and GRPO comparison branches,
  then final held-out calibration. The source plan gives conflicting stage orders;
  this preserves attribution. Cost if wrong: some sequential curricula may later
  need a dedicated experiment rather than being covered by the first comparisons.
- Implement `decision/specification.py`, `weights.py`, `verification.py` with
  three public functions instead of one file per small analysis. Domain code has
  no provider/model/telemetry dependencies. Cost if wrong: split modules when
  additional analysis policies make them difficult to maintain.
- Reuse existing exact solvers. Verification is independent of model claims,
  not an independent second implementation of every formula. Reference-library
  tests remain the numerical cross-check. Cost if wrong: a common solver defect
  can affect both labels and verification; reference tests are essential.
- Objective weights use all supplied alternatives; constant criteria supply no
  objective information. Entropy uses raw nonnegative columns; CRITIC uses
  benefit-oriented minmax. These declared conventions avoid silent data changes.
  Cost if wrong: changing the population or convention requires an explicit new
  configuration/version and can change recommendations.
- Run all further checks directly on RunPod per the user's steering. An initial
  GitHub-hosted check demonstrated the existing GPU entrypoint guard; that
  workflow was removed rather than weakened. Local work remains source editing
  and control-plane operations only.

## Completion boundary

This slice adds provenance-aware weight resolution, feasibility, Pareto,
recomputed scores/rankings, proposed-claim checks and explicit weight scenarios.
It does not implement preference learning, v2 SFT/RLCD/GRPO/DPO, real-data
adapters, production authentication, or arbitrary scenario boundary search.
See the five research documents and remote validation report for next steps
and actual evidence. Existing v1 artifacts and runtime guards are preserved.
