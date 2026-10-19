# Ordivo v2 behavioral baseline evidence

This file records the evaluator and specialist baseline contract. It is not a
claim that real Expedia, ESCI, or Swissmetro benchmarks have run. The live run
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
| Independent remote metric tests | Pending exact commit and data-contract integration. |
| Remote LightGBM save/reload parity | Pending dependency verification. |
| Real ESCI lexical and ranker results | Pending source acquisition, manifested dataset, remote run. |
| Real Expedia baseline | Pending authenticated source access and terms record. |
| Swissmetro stated-choice baseline | Runner/config ready; respondent-disjoint real manifest, remote tests and result pending. |

Do not turn a fixture pass or a completed test process into a real-data result.
