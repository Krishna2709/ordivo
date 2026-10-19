# Ordivo v2 research audit

Reviewed 2026-10-18 (America/Chicago), against source commit
`513bca619b91f9c725e988765212b64eb9931374` and release run
`20261019T005111Z`. This is a source and saved-artifact audit, not a new model
evaluation. No dataset or weights were downloaded and no training was run.

## Decision

Adopt the supplied review's direction: **contextual choice and ranking from
observed behavior**, with v1 retained as the synthetic teacher-imitation control.
The central question is whether MCDM pretraining improves a matched neural
model's held-out behavioral ranking compared with behavioral supervision alone.
This was already a research question in [the original brief](../PROJECT_BRIEF.md);
v2 prioritizes it over further synthetic-only accuracy tuning.

The user subsequently requested broad multi-domain coverage and explicitly
selected Expedia for use. The latest clarification makes Ordivo an **MCDM-focused
decision intelligence framework**, using a shared neural component across travel,
retail, web relevance and transport. Deterministic constraints, Pareto analysis,
sensitivity and traceable explanations are required differentiation; ranking
accuracy alone cannot validate that objective. These capabilities and RLCD/GRPO
training are planned, not present in v1.

The next deliverable is an independently versioned multi-source episode contract,
Expedia and ESCI adapters, and reproducible remote baselines. None is implemented
or evaluated yet. The [dataset register](../docs/V2_DATASET_REGISTER.md) records
the additional primary-source research and source-specific supervision.
See the [v2 design](../docs/superpowers/specs/2026-10-18-ordivo-v2-design.md)
and [implementation plan](../docs/superpowers/plans/2026-10-18-ordivo-v2-baseline.md).

## Verified v1 evidence

The following values come from the saved [base metrics](runs/20261019T005111Z/training/base_metrics.json),
[post-training metrics](runs/20261019T005111Z/training/post_metrics.json),
[training metrics](runs/20261019T005111Z/training/post_train_metrics.json),
and [prediction table](runs/20261019T005111Z/training/prediction_rows.json).

| Measure | Base | Ordivo v1 | Interpretation |
|---|---:|---:|---|
| Test winner agreement | 25/87, 28.74% | 51/87, 58.62% | Synthetic, disjoint scenario families |
| Training-subset agreement | 32.42% | 93.55% | 512 seen examples; not independent evaluation |
| Test ECE, 10 bins | 0.11830 | 0.23938 | Worse estimated confidence calibration |
| Test multiclass Brier | 0.78108 | 0.62310 | Better probability error by this proper scoring rule |
| Near-tie flag agreement | 20.83% | 41.67% | Only 24 eligible examples |
| Other examples | 31.75% | 65.08% | 63 eligible examples |

The 87-row table contains 87 unique episode IDs. Of the paired predictions,
29 changed from wrong to right, 3 from right to wrong, 22 remained correct,
and 33 remained incorrect. This supports the reported aggregate improvement;
it does not constitute replication or a multi-seed result.

The 34.93 percentage-point train/test gap combines fitting seen examples with
a deliberate distribution shift. It does not alone diagnose ordinary IID
overfitting. ECE and Brier move in different directions; do not summarize this
as every aspect of probability quality worsening. Ten-bin ECE on 87 episodes
needs uncertainty estimates before strong calibration conclusions.

## Data, splits, and denominators

- The saved [manifest](runs/20261019T005111Z/acceptance/data/manifest.json)
  contains 1,000 episodes: 800 train, 100 validation, 100 test. The family split
  precedes sampling; root seed is 42. There is no permutation augmentation.
- Training uses the first 512 eligible training episodes, not a random sample
  of all 800. The method counts are COPRAS 145, TOPSIS 144, VIKOR 79, WASPAS 144.
- The full test split has 13 nonunique cases, leaving 87 for winner evaluation.
  The [CPU baseline](runs/20261019T005111Z/acceptance/baseline.json) reports those
  exclusions. Training evaluation prefilters them, so its
  `excluded_nonunique=0` describes the already-filtered input, not all 100 rows.
- Test method counts are 25 COPRAS, 25 TOPSIS, 12 VIKOR, 25 WASPAS.
  Their adapted accuracies are 56%, 56%, 66.67%, and 60%, respectively.
- Test winner counts by alias: A 24, B 24, C 12, D 12, E 9, F 2, G 3, H 1.
  Base predictions choose A 60 times; adapted predictions choose A 40 times.
  Candidate counts vary, so compare alias frequencies conditional on presence
  and set size before inferring position bias. Training alias frequencies
  were not established from the artifacts inspected here.
- The uniform-probability baseline resolves argmax ties to the first option:
  its 27.59% accuracy is **not** a sampled uniform-random accuracy estimate.
- The `near_tie` scenario family (42 eligible cases) differs from the numerical
  near-tie flag (24 cases). The flag uses an absolute raw-score gap and is
  method-dependent; it is not a common measure of decision uncertainty.
- Sizes overlap across splits (3–8 alternatives, 2–6 criteria). There is no
  evidence of size extrapolation or real-domain generalization.

Sources: [generator](../src/liquid_mcdm/data/synthetic.py),
[metrics](../src/liquid_mcdm/evaluation/metrics.py),
[mathematical policies](../docs/mathematics.md).

## Training, input leakage, and architecture

The [renderer](../src/liquid_mcdm/training/rendering.py) whitelists method,
parameters, criteria, alternatives, and normalization. Targets, solver scores,
generation metadata, and provenance are excluded from the prompt. The
[tokenizer test](../tests/test_prompt_format.py) checks prompt masking and
target exclusion. This supports absence of direct target-field leakage in
the inspected synthetic path, not a general claim about future real features.

The [trainer](../scripts/train_d1_overfit.py) uses the train and test splits;
`eval_strategy='no'` and no validation dataset are supplied. Test examples are
also reused for compatibility and reload diagnostics. No label-gradient leak
was found, but this is not an untouched final test for iterative research.
Use a development split for diagnostics and reserve a new final test for v2.

The model produces one option distribution. Sorting those numbers is not an
evaluated full-ranking interface. `predict_standard` calls `render_example`,
which validates synthetic solver targets; it cannot ingest behavioral episodes
unchanged. A separate behavioral renderer and evaluator are necessary.

Solver permutation tests and renderer identity checks exist. No saved learned
model permutation, alias-renaming, or criterion-reordering evaluation was found.
The fixed A-to-H serialization makes these a priority for remote inference.

The [release receipt](runs/20261019T005111Z/training/run.json) records 400 steps,
BF16 A40 execution, LoRA rank 16, and a frozen vision tower. Reload comparison
preserved 16/16 decisions, with maximum probability difference 0.02147.
Repeats and independent reloads matched on those checked examples. This does
not prove exact probability parity with the in-memory training model.

Public HF metadata checked during this audit identifies
`blackOrder/Ordivo-v1` revision `997027add06b9c88c1ad04fce5378086b4c5190a`.
Its adapter LFS SHA-256 matches the saved publication receipt:
`691b01dac4b7a2dd13c5636c9187d446d378aa27eac699e925b9d773fa0c789c`.
This verifies remote metadata agreement, not a fresh payload download/reload.

## Tracking and documentation

Saved acceptance receipts record 130 passing remote tests. Saved tracking
material links [W&B run 4vblgw80](https://wandb.ai/black-order/liquid-mcdm-lab/runs/4vblgw80),
training telemetry, artifact uploads, and before/after predictions. Live W&B
contents and access were not re-audited in this pass. Weave is not implemented.

`configs/d1_3b_lora_v2.yaml` names the second **v1 experiment**, not the new
behavioral Ordivo v2. Preserve historical filenames and introduce a distinct
`ordivo-v2-behavioral` experiment family. The old internal dataset-card text
still says a license is unselected; the published Hub card declares CC BY 4.0.
Use the published card as the release reference and reconcile stale internal
copy separately without editing historical receipts.

## Primary-source checks and qualifications

Sources consulted on 2026-10-18 local time:

1. [Expedia data dictionary](https://www.kaggle.com/c/expedia-personalized-sort/data):
   grouped searches, click/booking fields, randomized-order flag, and an
   unlabeled test set fit the proposed benchmark. Access requires agreement;
   the page explicitly prohibits outside data. Displayed price is not always
   nightly. Customer-to-property distance is not distance to the city center.
   Do not manufacture the review's illustrative business-trip, budget,
   preference weights, or center-distance fields in real episodes.
2. [Historical competition rules](https://storage.googleapis.com/kaggle-media/competitions/kaggle/3504/media/Expedia%20Personalized%20Sort%20Official%20Competition%20Rules%20Final.pdf):
   outside-data and purpose/confidentiality restrictions mean these documents
   do not establish permission for our pretrained-model research or publication.
   Current applicable terms and permitted derived-model distribution remain
   unresolved. The user authorized using Expedia; local Kaggle authentication
   was not configured when checked. No agreement was accepted. This is a research-access finding,
   not a claim that all offline research is prohibited.
3. [Microsoft MSLR](https://www.microsoft.com/en-us/research/project/mslr/):
   an independent relevance-ranking candidate, also requiring an agreement.
   It is not a permission-free substitute or hotel-choice dataset.
4. [LightGBM ranker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
   and [parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html):
   query groups must describe contiguous rows, and relevance gains must be
   explicit. Pin and validate a version remotely; stable docs currently show
   4.7.0 with validation-API changes, so do not copy an older unpinned recipe.
5. [Liquid encoder card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/blob/b886781f7c6f10ca9b7096e21b83e30a073c2f39/README.md):
   the 350M identifier exists; API metadata still resolves to the previously
   researched revision `b886781f7c6f10ca9b7096e21b83e30a073c2f39`. It is a
   bidirectional masked-language encoder with a custom downstream head path,
   not a ready-made d1 ranking model. Runtime compatibility is still untested here.

## Priority and remaining uncertainty

1. Establish permissible real-data access and feature semantics; document actual
   group sizes, missingness, duplicate keys, label distributions, and available
   event-time information through remote inspection.
2. Build the behavioral contract, grouped split manifest, leakage allowlist,
   ranking evaluator, and LightGBM benchmark. Report actual results only after
   remote execution and read-back of artifacts.
3. Add source-specific real-data adapters and baselines across the dataset suite;
   freeze held-out-domain folds. Run a matched base-d1 / frozen-v1 inference comparison on identical groups,
   plus remote synthetic permutation diagnostics. Preserve full candidate sets.
4. Compare real-data-only fine-tuning with MCDM-pretrained fine-tuning on a
   shared multi-domain model; then investigate the 350M encoder against
   numeric/text controls. Separate in-domain results from unseen-domain transfer.

Actual Expedia schema distributions, live W&B completeness, behavioral model
quality, size limits, inference latency, and transfer benefit are unmeasured.
The review is a sound research direction, not evidence that these are solved.
