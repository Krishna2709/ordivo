# Ordivo v2 release criteria

**Status: research candidate; no v2 public model or results release authorized by this document.** This checklist is an evidence gate, not a claim that the gates have passed. The [experiment matrix](EXPERIMENT_MATRIX.md) names planned work; the [v1 research audit](../reports/V2_RESEARCH_AUDIT.md) is historical control evidence only.

## Release decision

A public v2 adapter requires a reviewed, immutable release dossier with a source commit, model/base revisions, adapter hash, environment lock, data and split manifests, evaluation artifacts, cost receipts, model/data cards, and an independently replayed inference sample. The reviewer records `pass`, `fail`, or `pending` for each gate with an artifact path and date. Missing evidence is `pending`, never a zero or a pass. A code-complete implementation or a fixture-only run cannot satisfy a real-data gate. The public announcement and Hub creation occur only after review of the dossier and explicit root authorization.

| Gate | Required evidence | Current state |
|---|---|---|
| Source and rights | Each training/evaluation source has origin, revision, file hashes, applicable terms, permitted model/artifact use, notices, and redistribution decision. Restricted raw rows stay outside public artifacts. | Swissmetro, ESCI and Expedia source files acquired/hashed; Expedia 399,344-group source-bound four-role build passed. Public trained-derivative rights remain source-specific and unresolved. |
| Group and leakage integrity | Complete candidate groups; source-specific split unit; frozen train/selection/calibration/final-test IDs; train-only transforms; no outcomes, exposure, provenance, IDs, or source order in predictor features. Group overlap and failed-query counts reported. | Four-role Swissmetro/ESCI source audit passed; Expedia four-role audit found 399,344 unique searches, zero duplicate groups/candidates, source mismatches or wrong-hash partitions. Full multi-source neural path pending. |
| Task semantics | Separate observed clicks/bookings, stated choices, graded relevance, explicit ranking, and synthetic solver labels. Per-task target/gain policies and eligible denominators frozen before final test. | Contract design exists; empirical validation pending |
| Comparators | Same groups and input evidence for random/heuristic, domain-appropriate classical, base d1, frozen v1, behavioral-only, and MCDM-pretrained comparisons where applicable. Exact solver is a mathematical reference, not a behavioral oracle. | Swissmetro and ESCI classical baselines measured; one preliminary two-domain neural adapter has selection-validation diagnostics and a bounded paired base comparison; matched multi-seed/control comparisons pending. |
| Held-out quality | Per-domain/task results with numerator/denominator, three matched seeds for final neural comparisons, paired group bootstrap intervals, within-domain and entire-held-out-domain evaluations, and failure coverage. A public shared-model generality claim requires at least three real domains including choice and relevance, with one shared component and actual held-out-domain evidence. A two-domain baseline remains preliminary. | Classical results plus a bounded paired neural probe: adapted Swissmetro accuracy 0.66 versus base 0.68 on 450/450 scenarios; adapted ESCI NDCG@10 0.752276 versus base 0.778290 on 27/50 selected groups (23 overflows). This is negative/mixed evidence; protected final, intervals and transfer pending. |
| Safety of decisions | Independent verification of numerical claims, infeasible proposal and served rates, unknown-constraint coverage, Pareto and weight-source handling; sensitivity to plausible weights. Generated prose alone is no verification. | V0 verifier exists; v2 end-to-end evaluation pending |
| Probability and uncertainty | Define event/population; fit any calibrator only on calibration groups; report NLL, Brier, ECE with bins and intervals, coverage, near-tie behavior and abstention where applicable. Ranking scores are not probabilities. | Pending |
| Robustness and operability | Candidate/criterion permutation, alias renaming, missingness, candidate count/context overflow, adapter reload, latency/memory and source-limited failure cases. No silent candidate truncation. | Validation-only pilot candidate-order probe found 52/180 Swissmetro and 7/12 admitted ESCI top-choice flips; 8/20 ESCI groups overflowed. Base comparison, full robustness and serving path pending. |
| Cost and provenance | GPU wall time/hours, hardware, tokens/throughput if measured, actual provider billing when available versus estimates, W&B/offline receipt, retained hashes and replay commands. | Pending |
| Public packaging | Public card and report agree with receipts, base and adapter loading instructions work, license and notice obligations satisfied, no credentials/restricted rows/source IDs in files or telemetry. | Preliminary adapter retained with private SHA readback; public packaging pending. |

## Minimum experiment design

Freeze source cohort, split manifest, feature allowlist, task gains, prompt/rendering revision, comparison budget, seeds, metric code, selection rule, and final-test policy before inspecting final results. Preserve a model-selection set distinct from calibration and final test. A failed run stays in the run ledger; do not quietly remove groups from the denominator. Source adapters must account for every expected query and candidate. For Expedia, single-booking conditional choice is a selected population; it does not estimate unconditional booking probability. For ESCI/MSLR, grades measure relevance, not purchase intent. For Swissmetro, stated choices do not prove revealed market behavior.

Report exact formula agreement on synthetic controls separately from preference prediction. VIKOR may lack a unique compromise winner. Unknown weights must remain unknown or be presented as explicit scenarios. A model's proposed alternative may be rejected by the deterministic verifier; report proposal and served outcomes separately. Even a passing aggregate must expose losses by task, group size, missingness, domain, and near-tie slice. ESCI and MSLR NDCG measures ordering within source-sampled judged candidate pools; it is not retrieval recall over all products or pages. Unjudged candidates must not become grade-zero negatives.

## Evidence record format

For every metric, retain `run_id`, executed commit, dataset and split manifest hashes, source rights record, task/event definition, model and base revisions, seed, metric name/version, value, numerator/denominator or eligible group count, exclusions, interval method, and artifact path. Store only sanitized aggregate values in the public HTML. The report builder accepts an allowlisted aggregate JSON object; it must never receive raw episodes, prediction rows, identifiers, or restricted excerpts. If a receipt cannot be independently located, the report says `pending`.

## Decision log

| Review item | Decision | Evidence |
|---|---|---|
| v2 release | Pending | No frozen multi-domain held-out model dossier |
| Expedia-derived model/artifact public rights | Pending | Access terms and acquired-source terms must be reviewed |
| RLCD/GRPO benefit | Not demonstrated | Bounded C1/R0 matched continuations both worsened selection-validation loss versus the shared start; no continuation held-out comparison. G0/G1 not executed. |
| v1 historical control | Available with limits | [Saved audit](../reports/V2_RESEARCH_AUDIT.md); synthetic winner imitation only |
