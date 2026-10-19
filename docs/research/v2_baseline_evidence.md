# Ordivo v2 behavioral baseline evidence

This file records the evaluator and specialist baseline contract and the
Swissmetro receipt below; Expedia and ESCI results remain pending. Each live run
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

The ESCI gain conversion follows [Amazon's official Task 1 ranking preparation](https://github.com/amazon-science/esci-data/blob/main/ranking/prepare_trec_eval_files.py).
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
| Source code and hand-authored tests | Present on the evaluation branch; remote execution pending. |
| Integrated remote suite | 214 tests, exit 0, no skips at integrated commit `9c532d98050e40ab3c9056e0bec1eeace123ccef`; controller archive SHA `4dd4793819f8d4507312894dbdbac9e510fc67d01afc6da7a6c292276f491838`. This includes evaluator fixtures and LightGBM save/reload parity fixture. |
| Remote LightGBM save/reload parity | Fixture check included in the 214-test suite; real ESCI/Expedia model bundle pending. |
| Real ESCI lexical and ranker results | Pending source acquisition, manifested dataset, remote run. |
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
