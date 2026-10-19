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

## S0b three-domain validation follow-up

The separate three-domain S0b pilot used train-only candidate permutation,
a lower learning rate, and a longer prompt limit. Its completed frozen receipt
was used with the same diagnostic script and
`configs/v2_candidate_order_validation_s0b.yaml` and its own exact effective
pilot config and best adapter. The config adds Expedia while retaining seed 42
and the same 20 validation units for Swissmetro and ESCI. Their manifests and
feature policies remained byte-identical, and their new
`selected_units_sha256` values equaled the first diagnostic's
Swissmetro `92252ab1fc60fa892e6af59b9da25198b21a9f83c857564e20eaa4d73a65d57e`
and ESCI `85e4fb1b1759a23112657243c231dc313e93655c3070280ea77f33e942b9740c`.
The same seed and hash permutation then reorders the same candidate IDs.

Private per-query files were compared by episode ID. Swissmetro uses the same
180 scored episodes. For ESCI, report positional metrics on the **original 12
scored episodes** separately from S0b's complete newly admitted set; if any
of the old 12 cease to be scored, disclose the shared-pair intersection and
each new exclusion reason rather than silently dropping them. Also report
S0b's selected 20 and all new overflow/failed counts. Expedia's 20 validation
searches are selected without labels, then only exactly-one-booking groups
are admitted under S0b's frozen `conditional_booking_choice` event. Zero- or
multiple-booking searches are counted together as `not_exactly_one_booking`
before rendering; report selected and admitted denominators separately. The
diagnostic still measures order sensitivity, not booking accuracy. Retain old
and new private per-query files,
and publish only aggregate paired comparisons. Because S0b changes domain
mixture, augmentation, learning rate, and prompt length together, any
change in positional sensitivity cannot be attributed to one factor. No test
split access or retuning is part of this follow-up.

The first remote attempt at integrated `1023208` failed during preparation:
the conditional-booking renderer rejected a zero-booking Expedia search.
No diagnostic aggregate was produced. The narrowly reviewed source fix
`484008d` reused the paired evaluator's event eligibility before rendering
and in coverage accounting, counting zero- and multiple-booking searches as
`not_exactly_one_booking`. The selection still hashes 20 whole validation
units without looking at labels. Remote focused fixtures (11) and the full
suite (255, no skips) passed at integrated `b621461315dc5fe276c563106375ab22ab6b06d7`;
direct script launch also passed. The corrected run exited 0 at that commit.
Its aggregate was read back byte-identically at SHA-256
`9be6417337ac574cc6fd38b95fe7f5c0b6e873848601db04eed3993110c344c4`
as `reports/runs/v2-swarm-20261019T030009Z/s0b-order-aggregate.json`
in the Operations worktree.

| Source | Selected whole validation units | Admitted pairs | Excluded | Top-choice flip rate | Mean probability L1 change |
|---|---:|---:|---:|---:|---:|
| Swissmetro | 20 | 180 / 180 episodes | 0 | 0.361111 | 0.340100 |
| ESCI | 20 | 20 / 20 groups | 0 | 0.500000 | 0.300219 |
| Expedia conditional booking | 20 | 15 / 20 searches | 5 not exactly one booking | 0.800000 | 0.223358 |

No query failed or overflowed. The Swissmetro and ESCI selected-unit hashes
matched the first diagnostic exactly. Operations privately joined both runs'
per-query files by episode ID, recording consumed file SHA-256s without
exporting IDs. All 180 old Swissmetro scored episodes remain admitted: on
these same episodes, the first pilot's flip rate / mean L1 were
0.288889 / 0.396628 and S0b's were 0.361111 / 0.340100. All 12 old admitted
ESCI groups remain admitted, with eight additional S0b groups. On the **same
old 12**, flip rate / mean L1 were 0.583333 / 0.318871 for the first pilot
and 0.750000 / 0.286416 for S0b; the full new 20-group rates in the table
have a different denominator. The sanitized private-match receipt has SHA-256
`25689eff2d9aecebeed627c0bfdfee04f54e15bafce5cd46828a182369968207`
at `reports/runs/v2-swarm-20261019T030009Z/s0b-order-old-match.json` in
the Operations worktree. These small, validation-only positional diagnostics
remain sensitive to candidate order; train-only shuffling was not shown
sufficient by the matched Swissmetro and ESCI flip rates.

The next experiment should compare base and S0b on the same validation
queries under both candidate orders, then isolate one training change at a
time in a controlled intervention. Keep design and checkpoint selection on
validation data and reserve a fresh untouched test cohort for final claims.
