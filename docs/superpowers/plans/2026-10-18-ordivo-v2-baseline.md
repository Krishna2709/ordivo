# Ordivo v2 multi-domain benchmark implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Use remote execution for every workload.

**Goal:** Build reproducible real-data decision benchmarks through a common
multi-source contract, beginning with Expedia and ESCI, before neural fine-tuning.

**Architecture:** Preserve v1 modules and artifacts. Add typed behavioral/relevance
episodes, source-specific adapters/splits, task-specific target policies, and
one score-validation boundary. Establish per-domain baselines before evaluating
a shared neural model across held-out domains.

**Tech stack:** Existing Python package; remote pandas/Arrow and LightGBM candidate
4.7.0; standard-library schema/split contracts; W&B. Exact dependency versions
are frozen after remote compatibility checks, not installed locally.

**Spec:** [Ordivo v2 design](../specs/2026-10-18-ordivo-v2-design.md).
Also read the [audit](../../../reports/V2_RESEARCH_AUDIT.md) and
[dataset register](../../V2_DATASET_REGISTER.md).

## Global constraints

- All data processing, tests, baselines and neural workloads run remotely.
- No v1 solver, checkpoint, historical run receipt, or dataset semantics changes.
- Source data, test labels, and exposure metadata never leak into model features.
- Preserve full candidate sets and source-appropriate split groups.
- No behavioral d1 fine-tuning before the executed data/baseline milestone.
- No pooled accuracy across observed choices, stated choices, and relevance labels.
- This plan is not implemented; commands below are proposed remote checks.

## Review focus

- Groups spanning file chunks must not be split or partially sampled (Task 2).
- Panel respondents and official test partitions need different split policies (Tasks 2, 6).
- Missing predictions and zero-positive groups must not inflate scores (Task 3).
- Different sources can reuse local IDs without becoming the same episode (Task 1).
- Real rows must not enter public telemetry by default (Task 5).

## Task 1: Source-independent episode and prediction contracts

**Files:** create `src/liquid_mcdm/data/behavioral.py`,
`src/liquid_mcdm/evaluation/ranking.py`, `tests/test_behavioral_contract.py`.

**Interfaces:** define JSON-serializable `DecisionEpisode` and
`DecisionInput` TypedDicts following the design; a discriminated `supervision`
union contains observed choice, stated choice, graded relevance or explicit
ranking. `CandidateScores = dict[str, float]`.
`validate_episode(episode: DecisionEpisode) -> None` and
`inference_input(episode: DecisionEpisode, feature_policy: dict) -> DecisionInput`.
`validate_scores(episode: DecisionEpisode, scores: CandidateScores) -> None`
checks exactly the available, eligible candidate set.

- [ ] Write failing fixtures for Expedia and ESCI that have different supervision
  types but the same input/score interface. Assert no observations, grades,
  position, provenance, group/source IDs or forbidden keys enter inference input.
- [ ] Add duplicate candidate, mismatched outcome IDs, unavailable-positive,
  partial judgment, non-finite score, missing/extra score and source-ID-collision
  cases. Assert explicit failure or the source policy's explicit unjudged mask.
- [ ] On remote runtime run `pytest -q tests/test_behavioral_contract.py` to
  establish failure, implement the boundary, rerun to pass, then commit this slice.

## Task 2: Expedia/ESCI adapters and immutable grouped splits

**Files:** create `src/liquid_mcdm/data/expedia.py`, `data/esci.py`,
`data/grouped.py`, `configs/behavioral/expedia.yaml`,
`configs/behavioral/esci.yaml`, `tests/test_behavioral_ingestion.py`.

**Interfaces:** `expedia.iter_episodes(path: Path, policy: dict) -> Iterator[DecisionEpisode]`;
`esci.iter_episodes(examples: Path, products: Path, policy: dict) -> Iterator[DecisionEpisode]`;
`assign_split(source: str, group_id: str, policy: dict, official_split: str | None) -> str`;
`write_dataset(episodes: Iterable[DecisionEpisode], output: Path, policy: dict) -> dict`
returns a versioned manifest and fails if output already exists.

- [ ] Write tiny hand-authored CSV/Parquet fixtures; no restricted rows in git.
  Assert chunk-size 1 and shuffled input produce identical normalized groups,
  assignments and hashes. Duplicate source keys and conflicting context fail.
- [ ] Test Expedia namespaced hash split from the design, source-independent IDs,
  zero/multiple outcomes, and complete groups. Test ESCI locale-sensitive product
  joins, missing-product failure and unchanged official test-query membership.
- [ ] Implement streaming group completion, source allowlists, target policy IDs,
  raw/source hashes, counts, diagnostics, and train-only transformation state.
  No byte-level or numerical promise about real data until it has been inspected.
- [ ] Run `pytest -q tests/test_behavioral_ingestion.py` remotely before and after
  implementation. Commit when determinism and grouping checks pass.

## Task 3: Independent query metrics and coverage ledger

**Files:** extend `evaluation/ranking.py`; create
`src/liquid_mcdm/evaluation/target_policy.py`, `tests/test_ranking_metrics.py`.

**Interfaces:** `targets(episode: DecisionEpisode, policy: dict) -> dict`
returns aligned raw labels, gains, judgment/availability masks and event labels;
`evaluate_ranking(episodes: Iterable[DecisionEpisode], predictions: dict[str, CandidateScores], policy: dict) -> dict`
returns per-query metrics, explicit denominators, failures and macro metrics.
No missing prediction is accepted as a completed scored run.

- [ ] Hand-check a perfect rank with NDCG=1 and first-relevant MRR=1; reversed
  binary pair has NDCG@1=0 and MRR=0.5. Check k exceeds candidate count safely.
- [ ] Pin Expedia gains `[0,1,5]` to grades `[0,1,2]`. Test zero-ideal-gain
  exclusion separately for combined, click and booking NDCG, multi-booking
  hit@1 and first-booking MRR, canonical-ID score ties, and unknown IDs.
- [ ] Define ESCI gain policy from its source task before scored execution;
  test all four source categories without relabeling them as behavior.
  Add missing/duplicate episode predictions and partial-judgment tests.
- [ ] Implement an evaluator independent of LightGBM defaults. Remote
  `pytest -q tests/test_ranking_metrics.py` must pass hand calculations and a
  second implementation check on queries where metric conventions agree. Commit.

## Task 4: Specialist baselines and validation-only selection

**Files:** create `src/liquid_mcdm/models/lambdarank.py`,
`models/ranking_baselines.py`, `tests/test_lambdarank_pipeline.py`,
`configs/behavioral/expedia_lambdarank.yaml` and an isolated remote dependency lock.

**Interfaces:** `fit_ranker(train: list[DecisionEpisode], validation: list[DecisionEpisode], config: dict, output: Path) -> dict`
persists model, train-fitted feature state, best iteration and validation receipt;
`predict_ranker(model_dir: Path, episodes: Iterable[DecisionEpisode]) -> dict[str, CandidateScores]`.
`fit_ranker` has no test-data argument. Feature policy and task policy IDs are
stored in the saved bundle and checked on reload.

- [ ] Prove contiguous groups, exact group-size sum, label alignment, unknown
  category handling, and train-only transform fitting with remote fixtures.
- [ ] Implement seed-42 LightGBM defaults from the design and a custom validation
  NDCG callback consistent with Task 3's zero-gain mask. Early stopping sees
  validation only. Do not depend on deprecated/unverified APIs.
- [ ] Add seeded random and declared numerical heuristics for Expedia; add a
  lexical baseline for ESCI before a lexical-feature ranker. Never claim a
  numerical model can use text without specifying its feature transform.
- [ ] Remote checks must show model save/reload score parity, metric parity with
  Task 3, and failed predictions causing failure rather than reduced test sets.
  Run `pytest -q tests/test_lambdarank_pipeline.py`, then commit.

## Task 5: Remote runner, telemetry, and actual first data benchmark

**Files:** extend `src/liquid_mcdm/runtime.py` with a remote-CPU guard;
create `scripts/benchmark_behavioral.py`, `tracking/behavioral_run.py`,
`notebooks/05_behavioral_baselines.ipynb`, `tests/test_behavioral_remote.py`;
update runtime/tracking tests and the baseline runbook.

**Interfaces:** `require_remote_runtime() -> dict` validates supported remote
identity without demanding CUDA; retain `require_remote_gpu()` for neural work.
CLI: `python scripts/benchmark_behavioral.py --config CONFIG --data DATA --output OUTPUT`.
Outputs: manifest/config/lock references, data profile, model bundle, per-query
scores in permitted storage, metric report, timings, telemetry receipt.

- [ ] Test local execution refusal and supported remote CPU identities; ensure
  the existing GPU guard remains required for model-weight work. No local
  package installs or data processing to validate these changes.
- [ ] Test telemetry using intentionally forbidden feature/ID fields; assert
  default W&B artifacts contain only allowlisted aggregate metadata. Public
  raw rows, fitted models and Weave traces require source-specific permissions.
- [ ] Configure authenticated Expedia access remotely and inspect real files.
  If credentials are unavailable, record that specific dependency and continue
  ESCI ingestion on an available remote runtime. Do not manufacture results or
  allocate a new paid service implicitly from this research plan.
- [ ] Freeze feature/gain/split manifests after source profiling. Run the actual
  seed-42 Expedia baseline and ESCI lexical baseline. Preserve all groups in
  each declared evaluation population, including zero-event coverage counts.
- [ ] Read back artifacts and W&B/offline receipts. Add a report clearly
  distinguishing fixtures, real-data results, unresolved fields and failures.
  This completes the first baseline milestone, not the whole v2 objective.

## Task 6: Expand domains and freeze neural comparisons

**Files:** add source-specific adapters, configs and tests for MSLR and
Swissmetro; optional SUSHI adapter; `docs/research/v2_transfer_protocol.md`;
`configs/behavioral/domain_suite.yaml`. Exact adapter paths follow Task 2.

- [ ] Preserve one official MSLR fold and respondent-disjoint Swissmetro
  partitions. Test missing/invalid choice, availability masks, repeated
  respondents and source-specific gains. Record terms/provenance per source.
- [ ] Run MSLR LambdaRank and Swissmetro multinomial-logit baselines remotely;
  implement group/subject bootstrap uncertainty and report coverage/failures.
- [ ] Freeze domain sampling/repetition caps, typed feature representation,
  task heads, complete-group model size limits and leave-one-domain-out folds.
  Add tests that held-out-domain records cannot enter training, development,
  transform fitting, prompt selection or calibration.
- [ ] Specify matched d1 and v1 continuation policies, seeds 42/43/44,
  downstream budgets, model-selection rules and the precise compute-matched
  extra-training control. Do not start that run with an unspecified control.
- [ ] Write the next neural implementation plan only after actual baseline
  evidence exists. It covers shared multi-domain learning, MCDM ablation,
  unseen-domain transfer and encoder/numeric/text comparisons, not just a
  larger Expedia fine-tune.

## Evidence required before reporting progress as complete

Source code plus fixtures are not dataset integration. A completed integration
has real source hashes, profile, intact groups, a split manifest and remote
validation evidence. A completed baseline has a persisted model, reload check,
scored held-out queries and telemetry receipt. A general-purpose research result
requires a shared model and held-out-domain evidence; dataset breadth alone
does not satisfy it. This planning turn completed none of those executions.
