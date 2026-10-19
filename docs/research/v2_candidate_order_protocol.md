# Frozen adapter candidate-order diagnostic

This is a validation-only robustness check on the completed Swissmetro + ESCI pilot. It does not select a checkpoint or alter the already frozen 50-unit held-out probe. The adapter, prompt version, feature policy, model revision, maximum prompt length, and source manifests must match the pilot receipt exactly.

For each source, select at most 20 whole validation split units by the fixed seed-42 hash of source, partition, and unit ID. The selection does not use choice or relevance labels. Score every selected episode with at least two available candidates twice using the same frozen adapter: once in source candidate order and once after a deterministic, label-independent shuffle of *all* candidates. The shuffle must change the order and preserve each candidate's ID, availability and features. The renderer remaps IDs to prompt aliases separately for each order. Compare the resulting probability distributions after mapping them back to source candidate IDs.

Report selected units and episodes, paired scored episodes, original-only, permuted-only and both-order prompt overflow, fewer-than-two-available exclusions, and any failed queries. For paired scored episodes, report the fraction whose top candidate ID changes (canonical-ID tie break) and mean sum of absolute probability changes across eligible candidates. Save complete per-query probabilities and statuses only in private remote storage. A complete aggregate requires exact paired coverage and at least one scored pair per source. A changed top candidate or distribution is evidence of positional sensitivity for this frozen model and selected validation cohort; this diagnostic does not estimate accuracy, test-set robustness, or causal learning.

Operations runs `scripts/diagnose_v2_candidate_order.py` on the guarded remote GPU with the exact effective pilot config and adapter, `configs/v2_candidate_order_validation.yaml`, controller-verified source archive identity, and a new `/workspace` output directory. Review the source diff and remote fixture test before execution. Preserve the aggregate, selection manifest, per-query files and receipt with read-back hashes. Do not use the diagnostic to retune the current pilot.

The source-reviewed diagnostic ran remotely at integrated commit
`a19d32d275db0012cecb50aafff84f4a125fbf7a` (archive SHA-256
`bafe4d600f7da2cc924c395e806b11e9a4dc0a8c76827e4aaf42a9f9d7772021`).
Remote focused fixtures and direct CLI launch passed before execution (10 tests,
exit 0). The aggregate was read back byte-identically with SHA-256
`1caff28407551758975ab5ebcf67abd3708f44fa8f2fa4a29ead7f04beeabad7`.
Private per-query output is under
`/workspace/ordivo-v2-data/candidate-order/validation-20-a19d/`.

| Source | Selected whole validation units | Paired scored episodes | Both-order prompt overflow | Top-choice flip rate | Mean probability L1 change |
|---|---:|---:|---:|---:|---:|
| Swissmetro | 20 | 180 / 180 | 0 | 0.288889 | 0.396628 |
| ESCI | 20 | 12 / 20 | 8 | 0.583333 | 0.318871 |

There were no failed queries or one-order-only overflows. These sizable order
effects caution against treating the pilot's positional prompt as a stable
decision policy. ESCI's scored denominator is only 12 queries, and the probe
uses the already consulted model-selection validation split. It is a robustness
diagnostic, not an untouched held-out efficacy or accuracy result.
