# Ordivo v2 — general-purpose contextual decision learning

Status: research design based on the user's supplied v2 review. Implementation
and benchmark execution are pending. Engineering defaults below are proposed
by the implementer and must be frozen in configuration before scored runs.

## Research objective

The user's follow-up explicitly broadens v2 beyond Expedia: build toward one
general-purpose model across multiple decision domains. Test whether MCDM
pretraining improves contextual decisions relative to an identical backbone
trained only on real-data labels. Preserve Ordivo v1,
exact solvers, published checkpoint hashes, and historical evidence.

v2 receives observed context and explicit candidate attributes, with optional
actual constraints or stated preferences. It returns candidate scores and
ranking. An algorithm name is unnecessary for the learned model. Exact MCDM
remains a separate, explicitly weighted baseline. Bookings and clicks are
observations; neither defines optimal utility or a mathematically correct choice.

Document extraction, retrieval, invented preferences, enterprise procurement,
and production decision claims are outside this milestone. Generality is a
testable research objective, not a property established by dataset count.

## Multi-domain scope and success criteria

Use Expedia as the first adapter, not the product boundary. See the
[dataset research register](../../V2_DATASET_REGISTER.md) for primary sources.

| Domain/source | Supervision | Role |
|---|---|---|
| Expedia hotel searches | Observed clicks/bookings | Real numerical choice sets |
| Amazon ESCI retail search | Human relevance judgments | Text plus candidate attributes; multilingual ranking |
| MSLR-WEB10K web search | Graded relevance | Independent engineered-feature ranking benchmark |
| Swissmetro transport | Stated choices with availability | Cost/time tradeoffs and respondent-grouped evaluation |
| SUSHI preferences | Explicit preference orders | Secondary preference-control dataset; not the flagship evidence |
| Existing exact MCDM data | Synthetic algorithm labels | Pretraining and mathematical diagnostic control |

First build Expedia and ESCI adapters to exercise two materially different
schemas. MSLR and Swissmetro extend the initial release benchmark; SUSHI is
optional. This is the intended suite, not a claim of acquired or validated data.
An initial multi-domain result needs at least three real domains, including
observed or stated choices and relevance judgments, through the same inference
contract. The final domain set is frozen before model comparison.

Compare a single shared model trained on the mixture against per-domain models.
Condition on the requested task and feature semantics, not a dataset ID as a
prediction shortcut. Use a shared backbone and shared heads by task type, not
one head per domain. Start with uniform domain sampling, uniform queries within
domain, and query-normalized losses; cap repeat exposure to small datasets and
log actual draws. Source row counts must not silently determine training weights.

Evaluate within-domain held-out data and leave-one-domain-out transfer. In each
transfer fold exclude that entire domain from training, preprocessing fitting,
hyperparameter/prompt selection, and calibration. Freeze generic templates and
target-domain adapter semantics using the dictionary, without target-label tuning.
Report zero-shot transfer and separately budgeted few-shot adaptation curves.
Base zero-shot, shared training without MCDM, and target-supervised models are
distinct controls. A target-supervised result is not an unseen-domain result.

Publish per-domain/task metrics and sample counts first. Any macro summary must
give equal declared domain weight and compare matched task metrics; never pool
choice accuracy with relevance NDCG or claim that raw scores share a utility scale.
Measure negative transfer, candidate-order stability, missing-feature behavior,
larger candidate sets, constraint violations, latency, memory, and cost.

## Architecture choices

| Approach | Benefit | Cost or limitation | Decision |
|---|---|---|---|
| Extend v1 teacher episodes with optional labels | Small initial diff | Solver validation and winner filtering conflate tasks | Reject |
| Separate behavioral episodes and grouped ranking | Clear provenance, independent metrics, reusable models | New adapter/evaluator boundary | Select |
| Immediately replace d1 with an encoder ranker | Direct alternative scores, smaller backbone | Changes task and model simultaneously | Later comparison |

The data flow is: remote source data → versioned grouped adapter →
frozen split/feature manifests → model-specific features → per-query scores →
shared ranking evaluator → sanitized experiment artifacts.

New modules live under existing `liquid_mcdm` boundaries; retain current imports
and commands. `data/behavioral.py` owns records/validation; `data/expedia.py`
owns its source conversion; `data/esci.py` is the second adapter;
`data/grouped.py` owns source-specific splits; `evaluation/ranking.py`
owns metric semantics; `models/lambdarank.py` owns the baseline; a separate
remote entry script owns execution and telemetry. No behavioral record enters
the synthetic solver validator or `render_example`.

## Record contract

The following is a fabricated schema fixture, not an Expedia record or an
expected model prediction. A real episode contains its complete source group.

```json
{
  "schema_version": "ordivo-behavioral-v1",
  "episode_id": "expedia-2013:fixture-search",
  "source": "expedia_personalized_sort_2013",
  "task_type": "observed_choice",
  "group_id": "fixture-search",
  "context": {"srch_length_of_stay": 3},
  "criteria": [{"name": "price_usd", "direction": "min", "weight": null}],
  "constraints": [],
  "preferences": {},
  "alternatives": [
    {"id": "fixture-hotel-a", "features": {"price_usd": 145.0, "prop_review_score": 4.0}},
    {"id": "fixture-hotel-b", "features": {"price_usd": 175.0, "prop_review_score": null}}
  ],
  "supervision": {
    "kind": "observed_choice",
    "policy_id": "expedia-click-booking-v1",
    "outcomes_by_id": {
      "fixture-hotel-a": {"click_bool": 1, "booking_bool": 1},
      "fixture-hotel-b": {"click_bool": 0, "booking_bool": 0}
    }
  },
  "exposure": {"random_bool": 1, "position_by_id": {"fixture-hotel-a": 2, "fixture-hotel-b": 1}},
  "provenance": {"raw_file_sha256": "fixture-only", "adapter_version": "expedia-v1"}
}
```

The common inference contract has `task_type`, context, a variable-length feature
schema, optional actual criteria/constraints/preferences, and alternatives with
numeric, categorical, text and missing-value fields. Feature definitions record
names, types and source units; a criterion direction can be absent for relevance
features with no cost/benefit meaning. Availability is an inference-time property
when the source supplies it. Keep unavailable candidates in the raw record and
mask them for choice scoring; a positive label on an unavailable item is an error.
Never equate an unjudged candidate with an observed negative.

Supervision is a discriminated union: `observed_choice` retains distinct outcome
events; `stated_choice` retains selected ID and respondent grouping;
`graded_relevance` retains the source grade/category for each judged item;
`explicit_ranking` retains ordered groups and partial-ranking coverage.
Each adapter owns a versioned target/gain policy. These are not interchangeable
labels. For example, ESCI E/S/C/I labels never become fake click/booking events.

An inference projection exposes task type and only approved context,
criteria/constraints/preferences, availability, and alternative features. Supervision, exposure,
provenance, source row order, and group identifiers never become predictors.
Candidate IDs are output keys; remap to aliases for d1 rather than embedding
property identity in the primary model. Missing values remain missing until a
train-fitted transformation. Zero-valued source fields require field-specific
semantics, not a universal zero-to-null or epsilon replacement.

Dataset criteria directions are analyst baseline choices, not measured human
preferences. Default weights are absent. A separate MCDM configuration supplies
explicit weights. Unsupported constraints/preferences remain empty.

Expedia training rows require aligned binary observations for every candidate; hold
unlabeled competition-test rows in an inference-only artifact. Duplicate
`(srch_id, prop_id)` keys, conflicting context, inconsistent randomization flags,
and malformed labels fail with diagnostic counts rather than silent deduplication.
Other adapters declare their own identity, grouping and label invariants.
Prediction output is keyed by episode and candidate, with a finite score for
each eligible candidate plus explicit unavailable status where applicable.
Reject extra, missing, or duplicate IDs and non-finite scores. Failed queries
remain failures in the expected-query ledger, not silently removed denominators.

## Access and ingestion

The user explicitly selected Expedia for use and requested a broad dataset
mixture. Proceed with authenticated remote retrieval when available. No Kaggle
credential variables or default credential file were found in this workspace
during this audit. Access configuration is still needed to retrieve the files.
Record applicable source terms and artifact permissions; user authorization to
use data does not create third-party redistribution rights. Do not upload source
rows or trained derivatives under an invented open license. MSLR has its own
agreement and task. ESCI source research and adapter work can proceed independently.
The public-source findings and links are in the [audit](../../../reports/V2_RESEARCH_AUDIT.md).

Once access is established, inspect data on a remote runtime. Preserve raw
checksums and source dictionary. Stream CSV into group-key partitions; finish
groups across chunk boundaries before constructing episodes. Profile query
size, missingness, label frequency, booking/click inconsistencies, and singleton,
zero-positive, and multi-positive groups. Do not sample individual rows.

Candidate fields for the initial dictionary include source search context,
displayed price, hotel rating, and location scores. Exclude IDs, displayed
position, randomization indicator, outcome columns, and transaction value from
the primary predictors. Historical/affinity/competitor features require a
documented availability check before inclusion. No inferred budget, city-center
distance, business-trip indicator, or preference weights. Price units remain
source-described until the country/site convention is established.

For Expedia, deterministic initial split: SHA-256 of
`expedia-2013|split-v1|42|<srch_id>`; interpret the
first 8 bytes as an unsigned big-endian integer modulo 100. Buckets 0–79 train,
80–89 validation, 90–99 test. Persist assignments; expect approximate ratios,
not exact group counts. Split before augmentation or label-dependent filtering.
Derived rows inherit the parent split. This prevents query overlap, not user
or hotel overlap. Record hotel overlap; add temporal and hotel-disjoint stress
tests later, with separately versioned manifests. This is not a universal split
policy: preserve ESCI's official query train/test partition and derive validation
only from its train queries; freeze a single published MSLR fold; group Swissmetro
and SUSHI by respondent. Namespaced manifests record split-policy ID, grouping
entity, raw checksums and overlap diagnostics. Cross-domain duplicate examples
or related source versions must be identified before constructing transfer folds.

Learn encodings, imputations, scaling, and any outcome-derived statistics on
train only. Unknown categories have an explicit sentinel. No label-derived
aggregate features in the first baseline. For reproducible subsets, order whole
groups by a separate seeded hash, independent of labels. All methods use the
same eligible query IDs and candidate sets in each comparison.

## Expedia benchmark contract

Primary LightGBM objective: LambdaRank with integer labels 0 (neither),
1 (clicked, not booked), 2 (booked), and explicit `label_gain=[0,1,5]`.
Preserve the two raw outcomes. This gain is our declared research policy;
do not present it as a verified competition leaderboard reproduction.

Report macro query NDCG@5 and @10 using these gains and log2 rank discount.
For zero-ideal-gain groups, exclude from NDCG and report their count/fraction;
do not silently accept a library default that scores them as perfect.
Use the same explicit query mask during validation selection and final scoring.
Also report booking-only and click-only binary-gain NDCG, booking MRR and
booking hit@1 on groups with at least one booking. Multi-positive groups use
first booked rank for MRR and any booked candidate for hit@1. Keep no-booking
groups in source artifacts and report coverage; no invented winner.
Compute zero-ideal-gain eligibility separately for each endpoint. A click-only
group can contribute to click NDCG while being absent from booking NDCG.
Any missing model prediction invalidates the scored run; report failure coverage
and resume it rather than publishing success on a favorable reduced subset.

Scores tie-break by canonical candidate ID, independent of original row order;
document this arbitrary tie policy and publish tied-score frequency. MCDM
tie groups and VIKOR compromise eligibility are retained alongside any ranked
score output. For VIKOR rank by ascending Q but do not label its top item a
unique compromise without satisfying the solver's conditions.

LightGBM uses contiguous query rows with `sum(group_sizes)==row_count`.
Initial engineering default: 31 leaves, learning rate 0.05, at most 500 trees,
50-round validation patience, minimum 20 child samples, four threads, CPU,
deterministic mode and column-wise histogram building. Candidate dependency:
LightGBM 4.7.0; pin only after remote compatibility validation. Select on our
validation NDCG@5 implementation, not an unexamined built-in score. Save the
best iteration. Test data are never used for early stopping or tuning.

Start with seeded random ordering and explicit price/rating heuristics, then
LightGBM. Add MCDM only after a common, documented numerical-domain projection
works for every method; report projection/exclusion coverage and run LightGBM
on the identical projected-feature subset as a fair control. Never silently
drop individual candidates or force unavailable numerical values to positive.

Ranking scores are not probabilities. Calibration metrics are optional and
require a defined event: for example conditional choice among exactly-one-booking
groups, with temperature fitted only on validation. Report that selected
population explicitly. Unconditional booking probability requires a separate
event/no-purchase model. Randomized impressions are a bias-analysis slice,
not proof of unbiased utility or known examination propensities.

## Task-specific baselines and shared neural representation

For ESCI, retain category labels and freeze a relevance-gain mapping in its
own policy; for MSLR, preserve grades and its official fold convention. Neither
uses Expedia's click/booking mapping. For Swissmetro use availability-aware
conditional choice likelihood and accuracy, including a multinomial-logit
baseline. Explicit rankings use observed order/pair agreement with a documented
tie/partial-ranking policy. Final policies are fixed by adapter work before
training; no source is scored using a placeholder target transformation.

LightGBM is a per-domain reference where features support it, not a universal
baseline made by padding unrelated numeric columns. ESCI first needs lexical
retrieval features or a declared frozen text representation and a lexical-only
control. These must be trained/tuned only on the appropriate development data.
MSLR has engineered features; it does not acquire semantic hotel criteria.

The shared neural representation encodes each feature's name/description, type,
unit, value, and missingness rather than assigning each dataset a different
fixed-width vector. Numeric values have an explicit numeric pathway as well
as text serialization. Apply transforms learned on source-domain training only;
retain unit metadata and avoid assuming unknown feature names have known meaning.
Candidate representations are shared across the set; task-specific losses act
on aligned scores or event probabilities. Serving receives a task descriptor,
not source labels, dataset identity, or source-specific trained heads. This
representation is a research hypothesis and requires cross-domain ablations.

## Neural research sequence

1. Measure zero-shot base d1 and frozen Ordivo v1 using one new behavioral prompt,
   identical features and complete groups. Neither prompt assumes an MCDM method.
   The frozen v1 model is an out-of-task transfer control.
2. Compare base d1 → behavioral LoRA against v1 → behavioral LoRA, matching
   prompt, supervision, seed, training groups, downstream updates, optimizer,
   trainable capacity, and validation selection. Freeze the adapter continuation
   policy before running: do not compare one trainable adapter with two stacked
   adapters. Use a cost-matched extra-training control before attributing gains
   specifically to MCDM labels; record the inherited synthetic training cost.
   Before that later experiment, select the exact extra-training control corpus
   and supervision, match processed tokens/optimizer updates, and publish its
   configuration. The initial LightGBM work does not depend on this neural-only
   design decision; no MCDM-specific causal claim precedes it.
3. First d1 endpoint is choice probability on eligible single-booking groups.
   Full ranking must be a named separate protocol. Sorting one-pass option
   scores is a ranking baseline, not a claim that the model was ranking-trained.
   Pairwise scoring adds quadratic comparisons/cycle risk; iterative removal
   changes the candidate set. Compare these only after establishing cost and
   consistency. Never truncate candidates to fit a context window silently.
4. Compare a numerical shared scorer/Deep Sets control with a 350M encoder
   plus typed numerical/text features and a shared candidate score head. Start with a
   frozen encoder; then investigate set attention and limited fine-tuning.
   Keep candidate positional embeddings absent and mask padding throughout.
   Prove score equivariance under candidate permutations. Serialized inputs
   alone do not guarantee that property. Encoder quality must beat numeric-only
   controls to justify the text backbone on this largely tabular dataset.

Use matched seeds 42, 43, 44 for final comparisons. A seed-42 smoke is diagnostic.
Report each seed and paired group-bootstrap intervals (queries for independent
query datasets, respondents for panel choices); do not bootstrap candidate rows
or treat repeated respondent choices as independent observations. Reserve untouched
final groups; a failed transfer hypothesis remains a publishable negative result.

Architecture references: [Deep Sets](https://papers.nips.cc/paper/6931-deep-sets),
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), and the pinned
[Liquid encoder card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/blob/b886781f7c6f10ca9b7096e21b83e30a073c2f39/README.md).
These motivate experiments, not prevalidated Ordivo implementations.

## Artifacts, execution, and completion

Run data processing, tests, baselines and neural workloads remotely. Local
activity is source/docs preparation and control-plane inspection only. CPU
baselines may run on the CPU of an approved remote runtime; a new GPU training
allocation is unnecessary for this milestone. Runtime guards must support
remote CPU work without removing the existing local-execution prohibition.

Track under the existing W&B project with experiment family
`ordivo-v2-behavioral`, task-specific metric namespaces, immutable manifests,
code revision, seeds, feature policy, gains, dependency lock, wall time, memory,
and provider billing evidence when available. Log sanitized aggregate metadata
by default. Real rows, source IDs, fitted models, and prediction tables require
artifact permissions; hashing an identifier is not redistribution permission.
Weave remains optional and must not duplicate restricted inputs into traces.

Completion of the first implementation slice requires remote adapter/metric
checks, a frozen access and split manifest, actual data-quality evidence, a
saved/reloaded LightGBM model, reproducible held-out results, and W&B receipt
or explicitly offline artifacts. Fixtures prove pipeline behavior only.
Until this exists, do not start behavioral d1 fine-tuning or announce a v2 model.
Completing one Expedia benchmark alone does not complete the multi-domain v2
objective; subsequent adapters and shared-model transfer evaluation are required.
