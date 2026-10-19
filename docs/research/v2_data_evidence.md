# Ordivo v2 data boundary and source evidence

This file describes implemented adapters. A source is **integrated** only after
its remote acquisition receipt, raw checksums, full-group profile, four split
files, and manifest have been inspected. Source code alone does not establish
that status. No source rows or model artifacts are licensed for public upload by
this document.

## Record and projection

`ordivo-behavioral-v1` episodes use namespaced IDs (`source:group_id`), task type,
context, criteria, constraints, preferences, alternatives, a discriminated
supervision object, separate exposure and provenance. The four supervision kinds
are observed choice, stated choice, graded relevance, and explicit ranking.
Observed clicks and bookings remain distinct binary events. Graded relevance
requires every candidate to be explicitly judged or listed in `unjudged_ids`;
unjudged is never a zero grade. Explicit ranking uses ordered tie groups and
explicit `unranked_ids`. Unavailable positive choices fail validation.

`inference_input(episode, feature_policy)` requires `context_features` and
`alternative_features` allowlists. Observed-choice policies also declare
`prediction_event`: `combined_click_booking_relevance` for the 0/1/5 ranking
objective, or `conditional_booking_choice` for a group with exactly one
booking. The latter is a categorical choice task; an unconditioned rank score
is not a calibrated booking probability. Its output contains task type, the
declared prediction event, approved
context, actual criteria/constraints/preferences, and alternatives with
`{id, features, available}`. IDs are score keys; neural prompts must replace
them with per-query aliases. Source/group/episode IDs, outcomes, judgments,
position, randomization, provenance and respondent IDs are excluded. Missing
values remain null. Weights remain null when they are unknown.

The `candidate_pool_scope` is source-exposed candidates for Expedia, the source
Task 1 reduced candidate list for ESCI, and three survey modes for Swissmetro.
Complete source groups do **not** imply all products, hotels, or travel options
that could have been retrieved or considered.

## Source contracts

| Source | Primary reference | Labels and fields | Terms evidence | Status |
|---|---|---|---|---|
| Expedia Personalized Sort | [official Kaggle data dictionary](https://www.kaggle.com/c/expedia-personalized-sort/data) | `srch_id` search, `prop_id` hotel, click and booking flags; position and `random_bool` are exposure | [competition rules](https://www.kaggle.com/c/expedia-personalized-sort/rules) require account review | Adapter implemented; official file listing succeeded but data.zip download returned HTTP 403, so no source acquired |
| Amazon ESCI | [official Amazon Science repository](https://github.com/amazon-science/esci-data) | Task 1 uses `small_version=1`, E/S/C/I judgments, `split`, product join on `(product_locale, product_id)` | [Apache-2.0](https://github.com/amazon-science/esci-data/blob/main/LICENSE) and [NOTICE](https://github.com/amazon-science/esci-data/blob/main/NOTICE) | Remote acquisition and full ingestion pending |
| Swissmetro | [official Biogeme source data](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat) and [example loader](https://biogeme.epfl.ch/sphinx/_modules/biogeme/data/swissmetro.html) | Stated `CHOICE`, respondent `ID`, `TRAIN_AV`/`SM_AV`/`CAR_AV`, times/costs/headways; `CHOICE=0` excluded as in the example | [Biogeme data page](https://biogeme.epfl.ch/) makes data available for research and education; source-specific redistribution terms still require confirmation | Adapter implemented; remote acquisition and full ingestion pending |

ESCI ranking's [official TREC preparation](https://github.com/amazon-science/esci-data/blob/main/ranking/prepare_trec_eval_files.py)
maps E/C/S/I to grades 4/3/2/1 and uses TREC gains 1/0.1/0.01/0. The
policy ID `esci-task1-official-gain-v1` preserves source letters in records;
the evaluator applies the gain map. The labels are relevance judgments, not
clicks or purchases.

The Expedia adapter allows only documented search context, displayed price,
star rating, review score, and two location scores as initial predictors. It
does not admit display position, randomization, source IDs, outcome fields,
gross booking amount, history/affinity or competitor fields. `price_usd` is
the source column name; currency convention should be checked in the acquired
dictionary before any cross-market monetary interpretation. A search can have
zero or multiple clicks/bookings. Test rows without labels are outside the
labeled benchmark.

The Swissmetro adapter preserves all three modes and availability masks.
`CHOICE=0` rows are omitted from supervised episodes, and the remote profile
must report their number. A selected unavailable mode is an error. The
respondent ID is retained only in supervision for respondent-disjoint splitting.
These are stated survey choices, not observed transport purchases.
The `cost_chf` feature preserves the source `TRAIN_CO`, `SM_CO` or `CAR_CO`
value. It is **raw survey cost**, not the traveler-effective amount for a GA
travel-card holder. Biogeme's [official preparation](https://biogeme.epfl.ch/sphinx/_modules/biogeme/data/swissmetro.html)
defines `TRAIN_COST = TRAIN_CO * (GA == 0)` and
`SM_COST = SM_CO * (GA == 0)` for its model, while car cost is unadjusted.
The `ga` context field is preserved so a specialist baseline can apply that
documented transformation using the same split; do not label raw rail cost as
actual paid fare. Car headway is null because that mode has no headway field.

## Immutable splits and remote materialization

All split policy IDs end in `v2-calibration`. The four output files are
`train.jsonl`, `validation.jsonl`, `calibration.jsonl`, and `test.jsonl`.
Validation is for model selection. Calibration is separate and cannot be reused
as test. Split assignment happens before target filtering or augmentation.

- Expedia: SHA-256 of `expedia-2013|split-v1|42|<srch_id>`, first eight bytes
  unsigned big-endian modulo 100. Buckets 0–79 train, 80–84 validation,
  85–89 calibration, 90–99 test. This retains the prior test membership.
- ESCI: official test queries remain test. Official train queries are assigned
  by a separate stable query/locale hash to train, validation or calibration.
- Swissmetro: hash respondent ID; all repeated scenarios for a respondent
  inherit one train/validation/calibration/test assignment.

Adapters spool arbitrary row order to a temporary SQLite table on disk, sort by
source group and row identity, and only hold one completed group in Python
memory. Duplicate row keys, conflicting context and missing products fail.
The writer stages split files in a temporary sibling and atomically renames the
directory only after full validation, hashes, and manifest creation succeed;
failed builds remove the stage. It fails on an existing output directory.
The manifest reconciles raw source rows = selected + omitted by reason, selected
rows = retained source rows, and all retained groups to four split counts.
`candidate_rows` separately counts mode alternatives when one Swissmetro survey
row expands to three mode candidates. It records source SHA-256 hashes, split
file hashes, missing feature counts, singleton and positive-label diagnostics.
It does not fit a transform; encoding/imputation must be fitted
on `train.jsonl` only by the modeling pipeline.

On the approved remote runtime, run `scripts/acquire_v2_sources.py esci --output
<remote-dir>` or `... swissmetro --output <remote-dir>`. The script resolves an
official Git revision, streams only requested files, and writes
`acquisition.json` with revision, size, SHA-256 and rights reference. Then run
`scripts/acquire_v2_sources.py build-esci --input <remote-dir>/repo --output
<dataset-dir>` or `build-swissmetro --input <remote-dir>/swissmetro.dat --output
<dataset-dir>`. For Expedia, the official Kaggle CLI accepts a scoped
`KAGGLE_API_TOKEN` as documented in its [current authentication guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md).
`scripts/acquire_v2_sources.py expedia --output <remote-dir>` requests only the
competition's `data.zip`, streams its `train.csv` member to disk, hashes both
archive and training file, then deletes the archive to limit retained copies.
It does not accept competition rules on behalf of the account holder. Then
`build-expedia --input <remote-dir>/train.csv --output <dataset-dir>` creates
the four-way materialization. Authentication or a rules-denied response must
be reported as a blocker; there is no mirror fallback. Raw source data and JSONL derived
records remain in the approved remote storage scope.

## Remaining evidence gates

The remote receipts must include the exact executed code commit, acquisition
revision, file hashes/sizes, source terms, raw and retained rows/groups,
duplicate/missingness/choice diagnostics, split counts and hashes, group
integrity checks, and test results. The Swissmetro data-specific rights check
is open. On the scoped official Kaggle CLI, `competitions files` listed
`data.zip` (434,573,097 bytes), but `DownloadDataFile` returned HTTP 403
Forbidden. No archive or training CSV was acquired, and automation accepted
no new rules. The 403 does not identify the exact account-side cause; a user
must review access in Kaggle if Expedia remains required. The first release comparison needs
actual held-out scored groups; passing adapter tests alone is insufficient.

For a third future real domain, the [SUSHI author page](https://www.kamishima.net/sushi/)
explicitly permits research use and prohibits redistribution. Its corrected
`sushi3-2016.zip` archive and ranking order files support a respondent-disjoint
explicit-ranking adapter; this adapter and its receipts are still future work.
The [official MSLR page](https://www.microsoft.com/en-us/research/project/mslr/)
requires acceptance of an online agreement before use, so an MSLR download
cannot be presumed authorized from a public URL alone.
