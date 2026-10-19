# Ordivo v2 behavioral baseline evidence

This file records the evaluator and specialist baseline contract and the
Swissmetro and ESCI receipts below; Expedia remains pending. Each live run
receipt must name an exact code commit, source hashes, terms, split manifest,
dependency versions, candidate coverage, test outcome, and remote artifact paths.

## Fixed target and metric meanings

| Source/task | Target policy | Scored population |
|---|---|---|
| Expedia observed hotel searches | `expedia-click-booking-v1`: grades 0 neither, 1 clicked without booking, 2 booked; gains 0/1/5 | Entire eligible exposed hotel set per retained search. Click and booking endpoints retain independent binary gain and denominators. |
| ESCI source judgments | `esci-task1-official-gain-v1`: E=1, C=0.1, S=0.01, I=0 | Judged candidates within each source query. Unjudged products remain counted but do not consume metric ranks. This is relevance, not a click or purchase event. |
| Swissmetro stated mode choice | `swissmetro-choice-v1`: selected available mode | Available modes only; accuracy@1 and reciprocal selected rank. The runner fits a respondent-disjoint conditional logit specialist. |

Macro query NDCG@5 and @10 use `gain/log2(rank+1)` with rank starting at
one. Ideal DCG zero yields no NDCG contribution, with an exclusion count per
endpoint. Expedia booking MRR and hit@1 include only groups with at least one
booking. Multiple booked candidates use the first booked rank. Scores tie by
canonical candidate ID. The evaluator requires a prediction for every expected
namespaced episode and every available candidate, rejects extras and non-finite
scores, and never silently shrinks the denominator. It groups aggregates by
source, task type, and policy; it publishes no pooled probability metric.

The ESCI gain conversion follows [Amazon's official Task 1 ranking preparation](https://github.com/amazon-science/esci-data/blob/7916cdf6ab75a462e77f20ab40428a10923998d5/ranking/prepare_trec_eval_files.py).
It is a declared source task mapping, not a human utility scale. The Expedia
0/1/5 gains are a research policy, not a competition reproduction.

## Baselines and training isolation

The controls are a seeded, order-invariant random score; explicit Expedia
price and review heuristics; ESCI lexical title/description overlap; and
per-source LambdaRank. Swissmetro has an availability-aware conditional logit
specialist. LambdaRank fits numeric medians and categorical vocabularies on
train only, with unknown categories mapped to `-1` and missingness channels.
Input fields pass the data contract's inference allowlist. The fit API accepts
train and model-selection validation only. The remote runner reads a separate
calibration split for integrity counts but never uses its labels, and reserves
test for final scoring. Expedia's inference policy explicitly names
`prediction_event: combined_click_booking_relevance` for the declared 0/1/5
LambdaRank objective. Click and booking remain separate evaluated endpoints;
the ranking score is not a booking probability. A future exactly-one-booking
probability model must name `conditional_booking_choice` instead. No
probability temperature or calibration claim exists.

The runner calls `require_remote_gpu()`, writes bundles and complete per-query
scores to `/workspace`, and emits a sanitized aggregate file without raw query
IDs. Real source rows, fitted models, and complete per-query records stay in
the permitted remote storage scope. The pinned dependency list is a candidate
until versions and wheels are verified by a remote acceptance run. Remote
dependency receipt `esci_deps` verified `pyarrow==21.0.0` with Python 3.12.3
and NumPy 2.2.6; LightGBM/model tests remain separate gates.
The runner requires the operations controller's `ORDIVO_SOURCE_COMMIT` from a
SHA-verified git archive; that receipt must be reconciled against controller
provenance. Candidate-pool scope is checked per source: exposed hotels for
Expedia, the official reduced Task 1 candidate list for ESCI, and three survey
modes for Swissmetro. The Swissmetro runner rejects respondent overlap across
all four splits; its fixed-setting conditional logit fits only train data,
saves its feature state, and checks validation score parity after JSON reload.
Its time, cost and headway inputs are source fields, not invented weights.

## Evidence ledger

| Gate | Status |
|---|---|
| Source code and hand-authored tests | Included in the remote integrated suite below; later neural/ranking fixtures passed remotely at `854de08` (12) and `73d9085` (15). Candidate-order diagnostic fixtures remain pending. |
| Integrated remote suite | 214 tests, exit 0, no skips at integrated commit `9c532d98050e40ab3c9056e0bec1eeace123ccef`; controller archive SHA `4dd4793819f8d4507312894dbdbac9e510fc67d01afc6da7a6c292276f491838`. This includes evaluator fixtures and LightGBM save/reload parity fixture. |
| Remote LightGBM save/reload parity | Fixture check included in the 214-test suite; real ESCI bundle persisted and scored. |
| Real ESCI lexical and ranker results | Complete for frozen old manifest at commit `22707ed34fd125e96aade9123f9baaf34cd64a18`; source-bound rebuild has byte-identical train/validation/calibration/test split files. |
| Real Expedia baseline | Pending authenticated source access and terms record. |
| Swissmetro stated-choice baseline | Real 1,044-episode test run complete with full score coverage and model reload parity; figures and limits below. |

Do not turn a fixture pass or a completed test process into a real-data result.

## Swissmetro held-out stated-choice receipt

The private sanitized aggregate was read back from the operations receipt for
integrated commit `9c532d98050e40ab3c9056e0bec1eeace123ccef`. The grouped
manifest SHA-256 is `028e5f2fc87cda849ca72213d8ba846fb00ac0784b7595e942f0644c1167b53a`;
raw survey SHA-256 is `27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2`.
Respondent-disjoint split counts are train 8,532, model-selection validation
549, unused calibration 594, and test 1,044. All 1,044 test situations were
scored with zero failed queries and zero choice-metric exclusions. Conditional
logit saved/reloaded validation scores matched. Per-query scores remain under
the remote `/workspace/ordivo-v2-data/baselines/swissmetro-9c532d/` directory.

| Baseline | Held-out accuracy@1 | Eligible |
|---|---:|---:|
| Conditional logit | 0.624521 | 1,044 |
| Travel time ascending | 0.636973 | 1,044 |
| Cost ascending | 0.281609 | 1,044 |
| Seeded random score | 0.359195 | 1,044 |

The simple travel-time heuristic is numerically ahead of this fitted control
on this split. No uncertainty interval or causal interpretation has been
computed. The source terms field points to Biogeme's site but redistribution
rights have not been established; keep model/per-query artifacts private and
do not promote a public efficacy claim from this receipt.

## ESCI old-manifest relevance receipt

The official Amazon repository was acquired at commit
`7916cdf6ab75a462e77f20ab40428a10923998d5`. The source Task 1 gain script
at that revision defines the E/C/S/I conversion used above. The remote run
used integrated code commit `22707ed34fd125e96aade9123f9baaf34cd64a18`
and frozen manifest SHA-256
`012fa5f58aaa6b75b0d73c1a9f84f811713e87c61df1dc5137867d34c2cba540`.
Raw examples/products SHA-256 values are
`4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a`
and `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265`.
The sanitized aggregate readback matched remote SHA-256
`d4e953a11ba9102b09fb5dd515b059731db856f6b26d4e8bb97fe66643be2673`.
Split counts were train 30,372, validation 1,688, unused calibration 1,744,
and official test 14,496. All 14,496 test groups were scored, with zero failed
queries, zero unjudged candidates, and zero zero-ideal-gain NDCG exclusions.
The fitted LightGBM 4.7.0 bundle selected iteration 138 on validation.

| Baseline | Test macro NDCG@10 | Eligible |
|---|---:|---:|
| ESCI lexical-feature LambdaRank | 0.659554 | 14,496 |
| Lexical overlap control | 0.649744 | 14,496 |
| Seeded random score | 0.533827 | 14,496 |

The ranker had at least one tied score in 14,495 test queries, and the lexical
control in all 14,496. A later exact analysis at integrated commit
`f69643794a932b14dac46b2e866667f5a7814bc4` compared the canonical-ID
tie break to the mathematical expectation under uniform permutations within
equal-score blocks. The analyzer consumed the original baseline aggregate and
three score files; their byte hashes matched Operations' private retention
manifest. Its sanitized aggregate matched local/remote SHA-256
`ea9145726743573c091e38c9af21d1ccd9cd8fb7881c0fd6836e28b2e0ee538e`.

| Baseline | Canonical NDCG@10 | Uniform-tie expected NDCG@10 | Queries whose tie permutations can affect NDCG@10 |
|---|---:|---:|---:|
| LambdaRank | 0.659554 | 0.656403 | 13,360 / 14,496 |
| Lexical overlap | 0.649744 | 0.646555 | 13,730 / 14,496 |
| Seeded random | 0.533827 | 0.533827 | 0 / 14,496 |

The expectation is an analytic tie-handling sensitivity calculation on fixed
scores, not an alternate trained model or uncertainty interval. The baseline
run remains attributed to the old frozen manifest. The source-bound rebuild
manifest SHA-256 is `9fd4065fc0db4586453a412123eb9ef393b1750cd8196bb11cad794d020f07cc`;
an independent remote audit confirmed all four split-file byte hashes match
the old manifest, so the evaluated groups are unchanged. Per-query files
remain in private remote storage.
