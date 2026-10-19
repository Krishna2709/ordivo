# V2 Permutation-Consistent Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the d1 alias/row-order failure and establish a matched, permutation-consistent numerical ranking baseline before any optional text encoder or renewed RL work.

**Architecture:** Freeze the existing d1 controls and probe them through a validation-only rendering seam. Build a separate label-free, fixed-schema numerical projection feeding shared candidate scorers; keep feasibility, preference authority and exact verification in the existing decision layer. Compare only aligned source objectives and complete groups.

**Tech Stack:** Existing Python, NumPy/PyTorch, pytest, grouped behavioral manifests, remote RunPod execution. Optional later Liquid LFM2.5-Encoder-350M, pinned and audited.

**Spec:** [ADR 0010](../../decisions/0010-v2-permutation-consistent-ranking.md); preserve [ADR 0009](../../decisions/0009-v2-four-role-splits.md), the [candidate-order protocol](../../research/v2_candidate_order_protocol.md), and source-specific [held-out protocol](../../research/v2_neural_heldout_protocol.md).

## Global constraints

- Current v2 results and base d1/S0/S0b checkpoints/receipts are immutable controls. This plan has **not** run any diagnostic, model or GPU job.
- Local workspace is source/document editing only; all tests, source builds, model loads and workloads run on the approved remote runtime. Keep private rows and per-query predictions there.
- Four disjoint roles: train, model-selection validation, calibration, final test. Existing exploratory test exposure and classical-test consumption require an audit before any protected claim.
- No candidate top-K, evidence truncation, test tuning, synthetic label reuse, undeclared feature, fabricated preference weight, or changed target under a shared metric name.
- Predeclare dtype/device score tolerances, seed list (proposed 42/43/44), run limits and cohort manifests before execution. Preserve hashes of code/config/data/checkpoints and read back outputs.

## Review focus

1. A rotated alias accidentally moves the answer slot: Task 1 tests fixed option/readout order and exact ID-to-slot mapping in all four cells.
2. An unavailable/padded candidate enters set context or loss: Task 3 tests equal scores with padding and excludes unavailable candidates.
3. All candidates are infeasible: Task 4 tests the explicit `no_feasible_alternative` path, without softmax.
4. A benchmark silently changes population or event: Task 4 tests complete ID coverage, policy, split and feature-manifest equality.
5. A prior test artifact is called untouched after partial subtraction: Task 4 tests exposure provenance and blocks a protected-test label if uncertain.

## Proposed file map

The following new files/interfaces are **proposed**, not current code. Keep existing `src/liquid_mcdm/_vendor/d1_prompt.py`, source adapters, historical reports and user plans unchanged.

| File | Responsibility |
|---|---|
| `src/liquid_mcdm/training/order_factorial.py` | Validation-only ID/row/slot plan and four-cell rendering; default renderer untouched. |
| `scripts/diagnose_v2_order_factorial.py`, `configs/v2_order_factorial_s0b.yaml` | Frozen remote probe, repeated reads, complete coverage and receipts. |
| `src/liquid_mcdm/data/numeric_features.py` | Named domain schemas, train-fitted transforms, missing/unit flags and leakage rejection. |
| `src/liquid_mcdm/training/set_ranker.py` | Shared numeric encoder and independent or masked-mean context scorer. |
| `scripts/evaluate_v2_set_ranker.py`, `configs/v2_set_ranker_numeric.yaml` | Matched remote training/evaluation controls and evidence ledger. |
| `tests/test_order_factorial.py`, `tests/test_numeric_features.py`, `tests/test_set_ranker.py`, `tests/test_set_ranker_evaluation.py` | Interface and invariant fixtures, run remotely. |

### Task 1: Separate row order from alias assignment

**Files:** Create `src/liquid_mcdm/training/order_factorial.py`, `tests/test_order_factorial.py`; inspect [`behavioral_rendering.py`](../../../src/liquid_mcdm/training/behavioral_rendering.py) and vendored alias/readout code without changing their default path.

**Interface (proposed):** `render_factorial_cell(projected: DecisionInput, tokenizer, feature_policy: dict, *, row_ids: tuple[str, ...], id_to_slot: dict[str, int], max_length: int) -> dict` returns the existing rendered fields plus the ID-to-slot map for private diagnostics. Accept only a bijection over the same complete candidate set and canonical fixed slot/readout order.

- [ ] Write fixtures asserting C0A0 prompt bytes and token IDs equal `render_predictor_input` for the same projected request (and `render_behavioral` for its source episode); C1A0 changes only content-row order, C0A1 changes only ID-to-slot assignment, and C1A1 composes both. Preserve the option list, readout slot order, features, availability and candidate-ID probability mapping. A legacy joint shuffle is separately named.
- [ ] Write failure fixtures for duplicate IDs/slots, missing slot, changed candidate membership and unsupportable token alias count. Check actual pinned tokenizer alias/readout groups; do not assume a 26-option limit.
- [ ] Run focused remote fixture command `python -m pytest tests/test_order_factorial.py -q`; expect all cases pass and baseline prompt SHA equality. Commit only reviewed source/tests.

### Task 2: Run the frozen validation factorial

**Files:** Create `scripts/diagnose_v2_order_factorial.py`, `configs/v2_order_factorial_s0b.yaml`; reuse the selection, eligibility, model loading and guarded remote runner patterns in [`diagnose_v2_candidate_order.py`](../../../scripts/diagnose_v2_candidate_order.py).

**Interface (proposed):** `run(pilot_config_path: Path, pilot_output: Path, diagnostic_config_path: Path, output: Path) -> dict` writes a selection manifest, private per-cell scores and statuses, aggregate, receipt and failure ledger under a new `/workspace` directory. Freeze S0b checkpoint, the same 20 validation units/source and seed 42 from the existing probe; no test access or checkpoint selection.

- [ ] Test that the four cells use identical eligible IDs, fixed token group list and feature policy; require complete scored coverage per cell or explicit per-cell overflow/failure. Repeat an identical C0A0 prompt/inference to quantify stochastic/numerical variation. Record prompt, token-ID and readout-group hashes, token counts, max length, mapping, per-ID logits/probabilities/margins/ties, top-ID flips and L1. Compare matched scored intersection and report all exclusions.
- [ ] Remote preflight: inspect source diff, run `python -m pytest tests/test_order_factorial.py tests/test_candidate_order_diagnostic.py -q`, load pinned tokenizer and verify alias cardinality. Freeze config/code/archive and define paid-runtime ceiling/timeout/operator cleanup in the receipt before allocation.
- [ ] Run the guarded remote command `python scripts/diagnose_v2_order_factorial.py --help`, then the CLI arguments documented by the new script against a **new** `/workspace` output; verify exit status, selected hashes, exact four-cell coverage, read-back SHA-256 and retained private per-query data. Publish only sanitized aggregate. Do not infer a causal mechanism if cells differ in tokenized length or coverage without reporting it.

### Task 3: Build the smallest numerical shared scorer

**Files:** Create `src/liquid_mcdm/data/numeric_features.py`, `src/liquid_mcdm/training/set_ranker.py`, `tests/test_numeric_features.py`, `tests/test_set_ranker.py`. Reuse [`inference_input`](../../../src/liquid_mcdm/data/behavioral.py) for source allowlists; no new source adapter.

**Interfaces (proposed):** `fit_numeric_schema(train: list[DecisionEpisode], schema: FeatureSchema) -> FittedSchema`; `transform_numeric(request: DecisionInput, fitted: FittedSchema) -> NumericBatch` with named context/candidate features, missing flags, units, availability mask and output ID order. `score_candidates(batch: NumericBatch, *, arm: Literal["independent", "masked_mean"]) -> dict[str, float]` uses a shared encoder `z_i`; independent scores are `h(z_i)`, set scores are `h(z_i, masked_mean(phi(z_j)), log1p(count))`. Keep architecture dimensions and dtype in a versioned config before training.

- [ ] Test leakage rejection for ID, source/group, position/exposure, provenance and target fields; schema mismatch, unit mismatch, nonfinite present values and missing-value flags. Fit transforms on train only and prove validation values do not alter them.
- [ ] Test score-by-ID equivariance for full permutations, 2/5/20 candidates, unavailable and padded alternatives, and same content with changed IDs. Declare float32/BF16 tolerance in config before held-out evaluation; test finite precision against that bound and canonical-ID exact/near-tie reporting separately. Do not use candidate indices or positional embeddings.
- [ ] Run `python -m pytest tests/test_numeric_features.py tests/test_set_ranker.py -q` on remote CPU. Inspect actual parameter sharing and masks in review. Commit the passing slice before experiments. An optional attention arm waits for a documented interaction error slice.

### Task 4: Matched evaluation and decision-layer checks

**Files:** Create `scripts/evaluate_v2_set_ranker.py`, `configs/v2_set_ranker_numeric.yaml`, `tests/test_set_ranker_evaluation.py`; reuse [`evaluate_ranking`](../../../src/liquid_mcdm/evaluation/ranking.py), [`resolve_weights`](../../../src/liquid_mcdm/decision/weights.py), and [`verify_decision`](../../../src/liquid_mcdm/decision/verification.py).

**Interface (proposed):** Evaluator accepts frozen source/target/feature/pool/split manifests and paired prediction maps keyed by episode and candidate ID; it refuses an incomplete group. Output has source-specific metrics, natural-size bins, coverage/exclusions, paired unit-level interval inputs, run seeds and separate preference/verification outcomes.

- [ ] Test a mismatch in event (`combined_click_booking_relevance` versus `conditional_booking_choice`), search pool, feature budget, split hash, missing ID or criterion schema fails before comparison. The 5,000-search Expedia LambdaRank relevance NDCG reference must only pair with the same 0/1/5 relevance target and groups; booking 2/29→4/29 remains separate.
- [ ] Audit every source's prior final-test exposure, including classical benchmark consumption, and record it in a manifest. Test that uncertain/exposed cohorts cannot be labeled protected merely by subtracting the neural 50. Preserve train/selection/calibration/test roles; reserve a genuinely new or source-heldout protected cohort before a final claim.
- [ ] Test all-infeasible gives `no_feasible_alternative`; unknown/conflicting weights remain unknown/scenarios or abstain; user, approved-policy, externally supplied learned-evidence, objective and unknown modes retain provenance. `verify_decision` must check a claimed formula against the stated weights; a neural score alone does not pass. There is no preference learner in this task.
- [ ] Run `python -m pytest tests/test_set_ranker_evaluation.py tests/test_decision_verification.py -q` on remote CPU. Freeze matched baselines and the numeric arms before any bounded remote GPU training. For paid runs, record exact config, maximum cost/time, timeout, destination and operator deletion controls first; report actual cost afterward.
- [ ] On remote, use predeclared seeds 42/43/44 if funded. Report per-source point estimates and paired bootstrap intervals clustered by respondent (Swissmetro), query (ESCI) or search (Expedia), with seed variation separately. Report complete natural-group 5/10/20/50/100+ bins where supported, synthetic stress separately, missingness/latency/memory, and ID-mapped permutation error. Label validation/exploratory results honestly until protected-cohort gate passes.

### Task 5: Optional text and interaction research gate

**Files:** Extend the proposed numeric schema/scorer and versioned config only after Task 4 evidence; add focused tokenizer/text-sharing tests. Any new text file must have one responsibility, such as `src/liquid_mcdm/training/candidate_text.py` for shared per-candidate encoding.

- [ ] Document which numeric error slice requires text or candidate interactions and freeze a matched ablation. If the numeric arms are adequate, defer this task.
- [ ] On the remote runtime, pin the official Liquid encoder revision, review remote code and model license, verify `AutoModel` body/pooled shape and actual device/VRAM, then run a tiny frozen-body forward/reload fixture. Encode context and candidates independently with shared weights; test permutation after ID remapping. Begin with frozen encoder and head-only training; consider unfreezing or Set Transformer only after separate bounded evidence and cost review.
- [ ] Keep RL deferred until stable scores, constraints, reward definition and an independent test suite exist. A new RL plan must use a separate controlled design; this plan does not authorize or implement GRPO.

## Acceptance evidence

The diagnostic is complete only with frozen identities, four cells plus repeat, exact paired coverage, private per-ID outputs and read-back hashes. The numerical milestone is complete only when both arms pass mask/equivariance fixtures and a matched, source-specific evaluation with truthful exposure and uncertainty accounting. An architecture that passes mathematical permutation fixtures may still lose to a classical baseline; retain that negative result. No proposed benchmark establishes general-purpose decision intelligence, arbitrary-criteria support, calibrated preference authority, or production readiness by itself.
