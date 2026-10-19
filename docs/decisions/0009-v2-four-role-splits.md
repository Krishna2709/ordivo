# ADR 0009 — Disjoint model selection and calibration for v2

Status: accepted implementation correction, 2026-10-18. This ADR supersedes the
three-way split and validation-fitted temperature wording in the earlier v2
design. It does not revise the user's original plan or any v1 synthetic dataset.

## Context

The design originally assigned Expedia groups 80/10/10 across train,
validation and test, then used validation both to select a model and fit
temperature. That reuses evidence across two adaptive roles. The experiment
matrix and RL research design already require disjoint model-selection,
calibration and final-test groups. Independent methodology review identified
the conflict before real-data ingestion.

## Decision

Use four group-disjoint roles: train, model-selection validation, calibration,
and final test. Expedia's `expedia-2013-split-v2-calibration` policy maps SHA-256
buckets 0–79/80–84/85–89/90–99 to those roles. Its historical `split-v1` hash
salt stays stable; the policy ID changes because bucket meaning changes. ESCI
keeps its official test queries untouched and derives selection/calibration
from official train. Swissmetro splits by respondent. MSLR keeps a declared
official fold. No source group or respondent where respondent disjointness is
the declared split unit can appear in multiple roles. Fit transforms and tune
models, thresholds and calibration only on their assigned development roles;
the final-test labels remain untouched. Other entity overlap, such as a hotel
appearing in separate Expedia searches, is diagnosed rather than silently
claimed absent. Domain-holdout evaluation excludes the target domain from all
of those steps. Save policy ID, group assignments and overlap checks in the
manifest. Fit calibration only after selecting the model, and assess it once
on the untouched final population.

## Consequences

Validation and calibration samples are smaller than under 80/10/10, so pilot
estimates may be noisy. Report counts and intervals rather than claiming
precision from a tiny slice. A source without enough eligible groups may not
support a calibrated-probability claim. A remote fixture or data-manifest test
checks implementation; actual source counts and leakage checks still require
remote source receipts. The [release criteria](../V2_RELEASE_CRITERIA.md) and
[experiment matrix](../EXPERIMENT_MATRIX.md) describe the claim gates.
