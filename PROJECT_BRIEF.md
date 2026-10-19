# Codex Project Brief — Liquid MCDM Decision Models

**Project:** Liquid-MCDM Lab  
**Status:** Research + reproducible prototype  
**Environment:** VS Code + Codex for development; Google Colab managed GPU for training; Weights & Biases (W&B) for experiment tracking.  
**Date:** 2026-10-18

## 0. Instructions to Codex

Act as a research engineer and ML systems engineer. **Do current-source research first**, summarize evidence in `docs/research_findings.md`, and then implement an independently testable, reproducible research repository. Do not assert that undocumented fine-tuning mechanisms work. Inspect upstream model configurations and official sample notebooks before selecting model loaders, trainers, loss functions, option-formatting rules, or quantization implementations. Work incrementally; minimize paid Colab compute usage. Ask me before a long GPU run, downloading gated datasets, submitting forms, or spending money.

Treat this document as the initial project specification, not a claim that all upstream APIs or datasets have been validated. Record discovered incompatibilities and justified deviations in an ADR (`docs/decisions/`).

## 1. Research goal

Investigate whether a pretrained Liquid AI decision model can learn **multi-criteria decision making (MCDM)** tasks—COPRAS, WASPAS, VIKOR, and TOPSIS—and whether MCDM supervision helps predict *real* choices, initially hotel rankings. In the future, transfer the approach to healthcare vendor benchmarking; **healthcare procurement is out of scope for this phase**.

Separate three notions rigorously:

1. **Exact algorithm fidelity:** reproduce mathematically correct scores/rankings conditional on algorithm, weights, criteria directions, and alternative sets.
2. **Behavioral preference modeling:** predict observed clicks/bookings from a real ranking dataset; these observations are noisy, position-biased proxies for preferences, **not algorithmic ground truth**.
3. **Operational usefulness:** compare any learned model against exact solvers and strong classical learning-to-rank baselines. Do not claim learned math outperforms exact math at computing the same formula.

### Questions to answer

- Does fine-tuning an existing Liquid decision model improve MCDM winner/ranking prediction on held-out scenarios?
- Can it transfer across criteria counts, alternative counts, distributions, weighting schemes, and methods?
- Does a pretrained decision model offer value beyond a small numerical Set Transformer and LightGBM/LightGBM Ranker?
- Can a model with MCDM pretraining improve behavioral ranking on Expedia, compared with a model trained on Expedia labels only?
- Do models preserve constraints/invariances (alternative permutation, criterion direction, scale transformations, Pareto dominance when applicable)?

## 2. Prioritized model experiments

**Primary:** `LiquidAI/d1-3B`. Adapt the **official Liquid decision-model SFT examples** with TRL/PEFT LoRA, then evaluate Unsloth as an optional optimized path. Confirm the actual answer-token/option-logit mechanism, chat template, whether correct training loss masks non-answer tokens, and whether probabilities are formed from a restricted option vocabulary. Do **not** casually substitute a generic classification head or the training recipe for a different family.

**Secondary A:** `LiquidAI/d1-omni-600M`. Research whether its weights, custom heads, training code, and official fine-tuning support permit our task. If unclear, run inference-only and document the blocker; do not pretend it supports the d1-3B notebook unchanged.

**Secondary B:** `LiquidAI/LFM2.5-Encoder-350M` (prefer **350M**, not 230M). Design a *custom* permutation-equivariant alternatives/set-ranking head with numeric feature processing and/or a serialized-text baseline. This model does **not** already include the same task-specific d1 decision head. Confirm exact identifiers and compatibility before coding.

**Controls:** exact MCDM solver; a ~1M–10M parameter numerical Set Transformer (capacity baseline); LightGBM or LightGBM Ranker; optionally a simple per-alternative MLP. These are controls, not the project's flagship models. Do not fine-tune all variants before a cheap baseline works.

## 3. Upstream sources for initial verification

- Liquid TRL notebook: https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb
- Liquid Unsloth notebook: https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/sft_for_decision_models_with_unsloth.ipynb
- Unsloth guide: https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth
- Liquid d1-3B: https://huggingface.co/LiquidAI/d1-3B
- Liquid d1-omni-600M: https://huggingface.co/LiquidAI/d1-omni-600M
- Liquid 350M encoder: https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M
- Liquid cookbook: https://github.com/Liquid4All/cookbook
- Expedia ranking dataset: https://www.kaggle.com/competitions/expedia-personalized-sort/data
- UCI Car Evaluation: https://archive.ics.uci.edu/dataset/19/car+evaluation
- MSLR: https://www.microsoft.com/en-us/research/project/mslr/
- MCDM/LLM study: https://arxiv.org/html/2502.15778v1
- TRL SFT: https://huggingface.co/docs/trl/sft_trainer
- Colab FAQ: https://research.google.com/colaboratory/faq.html
- W&B: https://docs.wandb.ai/

In `docs/research_findings.md`, provide for each model: exact checkpoint name and revision, architecture/parameterization, license and intended use, available code, supported loaders, correct training data schema, decision scoring semantics, known runtime memory constraints, and links/commit hashes. Verify the Expedia access rules, feature meanings, availability, and permission for the intended research use before making it mandatory. Cite upstream links and note retrieval dates.

## 4. Dataset plan

**Stage 0 (no external access required):** Generate transparent synthetic MCDM *decision episodes* from sampled criteria matrices, weights, criteria directions, and algorithm identifiers. Labels must come from validated **exact** MCDM implementations. Generate correlated and diverse attributes, not exclusively IID uniform random numbers. Explicitly record seeds, generation settings, and distributions. Store `train/val/test` manifest with scenario-family splits; split templates/seeds **before augmentation**. Create 1k smoke examples first, then 10k–20k episodes for first learning-curve run; scale only after results warrant it.

**Stage 1:** UCI Car Evaluation dataset for simple multi-attribute decision-function sanity checks. Clearly distinguish its multi-attribute classification labels from vendor/hotel rankings.

**Stage 2 (primary real-data benchmark):** Expedia Personalized Hotel Search dataset, if terms and access permit. Reconstruct each search as a list of candidate hotels with context and available numerical/categorical features. Label clicks and bookings separately; document propensity/position bias and whether a controlled or randomized subset exists. Preserve complete search groups during splitting; **do not split alternative rows of the same search across train and test**. Don't invent unobserved criteria (e.g., amenities if not available); no leakage via future variables. If Expedia is inaccessible, use MSLR-WEB10K or another verified public grouped ranking dataset while retaining synthetic MCDM tasks; document domain limitations.

**Do not conflate:** synthetic solver labels, expert preferences, and observed behavior. Each gets its own field, provenance tag, loss, and evaluation slice.

### Canonical example schema

Use JSONL/Parquet with documented schema, for example:

```json
{
  "episode_id": "synthetic_000001",
  "source": "synthetic_solver",
  "context": {"domain": "hotel", "scenario_family": "price_quality_tradeoff"},
  "method": "WASPAS",
  "method_params": {"lambda": 0.5},
  "criteria": [
    {"name": "price", "direction": "min", "weight": 0.4, "unit": "USD"},
    {"name": "rating", "direction": "max", "weight": 0.6, "unit": "stars"}
  ],
  "alternatives": [
    {"id": "A", "values": {"price": 120, "rating": 4.5}},
    {"id": "B", "values": {"price": 100, "rating": 4.0}}
  ],
  "target": {
    "algorithm": "WASPAS",
    "ranking": [],
    "winner": null,
    "raw_scores": {},
    "tie_policy": "explicit"
  }
}
```

The target fields shown are **placeholders**, not computed facts. The generator must fill and verify them. Keep source measurements and exact solver scores unrounded; apply formatting/precision in prompts separately. Include tie and close-call metadata.

## 5. MCDM solver implementation requirements

Implement **COPRAS, WASPAS (configurable lambda), VIKOR (configurable v), and TOPSIS** using mathematically documented normalizations and criteria directions. For VIKOR, preserve the minimization semantics of its compromise score and explicitly implement/record the acceptable-advantage and stability conditions before claiming a unique compromise solution. For WASPAS, validate positive-domain assumptions for the weighted product; handle zero/non-positive costs safely. Define tie handling, missing criteria policy, zero-variance criteria, min/max feature normalization, and scale/units behavior.

Build hand-calculated unit cases plus independent cross-checks against reputable published examples or an established library. Cover method disagreement, sensitivity to weights, scale invariance where mathematically justified, alternative reordering, degenerate alternatives, cost-benefit reversal, and rank reversal where methods can exhibit it. **Do not assume every method is invariant to adding alternatives.**

## 6. Model inputs and training

First train the primary d1-3B on **single-choice winner prediction** with exact teacher labels, conditioning on method, parameters, weights, criteria directions, numeric values, and candidate options. Match the verified official Liquid training template, option tokenization, and inference scoring. Evaluate **zero-shot before fine-tuning** on the same held-out test distribution. Train LoRA (QLoRA only if compatible and needed). Keep prompt templates deterministic and versioned.

Do not claim winner classification equals full ranking. Full ranking should be a separate milestone: compare pairwise preference heads vs iterative choice vs direct listwise heads, checking cyclic pairwise choices and inference expense. For real Expedia labels, use query-group ranking losses or carefully justified click/booking objectives; report metrics by entire search group, not by isolated hotel row.

For the encoder-350M branch, prototype a custom set-aware, permutation-equivariant head over structured alternative features, plus serialized-text input experiment if justified. Specify how pretrained encoder embeddings interact with numeric variables, masks and set sizes. Compare frozen backbone / head-only against LoRA/partial unfreeze.

A practical training order:

1. CPU-only exact solver + data smoke tests.
2. Dataset/tokenizer inspection and baseline evaluations.
3. 50–200 example GPU overfit test on d1-3B with W&B logging.
4. Short LoRA experiment on ~1k then ~10k+ episodes, adapting sizes to compute budget.
5. Evaluate on held-out scenario families/criteria counts/number of alternatives.
6. Compare numerical/LightGBM controls.
7. Only then attempt Expedia preference experiments, d1-omni, and encoder-350M.

## 7. Colab + VS Code workflow

**VS Code + Codex:** local repo, code reviews, notebooks, tests, configs, git; no assumption Codex remotely controls a managed Colab runtime.

**Google Colab:** open generated notebook in browser, select available NVIDIA GPU (prefer A100; fall back to L4/T4 with memory-appropriate configurations), and run the exact repo scripts via a clone and parameterized commands. Do not hardcode Colab paths throughout package modules. Treat Colab VMs as **ephemeral**, and resume from persisted checkpoints. The user's UI displays A100/L4/T4/G4 selections, but real allocation, VRAM, runtime limits and compute-unit usage must be checked at runtime with `nvidia-smi` and Colab's UI. High RAM is host RAM, **not GPU VRAM**.

**Optional VS Code↔Colab integration:** research the currently supported VS Code Colab extension/remote connection mechanism, verify author, permissions, compatibility, and limitations with the user's managed Colab runtime. If unsupported, use the browser Colab workflow; do **not** set up unsupported public Jupyter endpoints, tunnels, policy bypasses, or exposed credentials. Google's documented *Colab local runtime* is the opposite direction (Colab UI using your own Jupyter runtime) and is not proof that VS Code can attach to a paid managed Colab GPU.

Produce notebooks:
- `notebooks/00_environment_check.ipynb` (GPU, CUDA, torch, VRAM, dependency checks)
- `notebooks/01_mcdm_data_and_baselines.ipynb`
- `notebooks/02_d1_3b_lora_colab.ipynb`
- `notebooks/03_evaluate_and_compare.ipynb`
- Optional `04_encoder_350m_experiment.ipynb`

Notebooks should be thin wrappers over importable functions/CLI scripts, not the only implementation. Use config files for model, dataset, and hyperparameters. Pin tested versions in lock/requirements files after confirming official compatibility. Provide instructions for Git clone, safe Hugging Face/Kaggle auth, W&B login, Drive checkpoint sync or approved alternative persistent artifact store, restart/recovery and resuming. No secrets in committed notebooks or logs; use Colab Secrets or environment variables.

## 8. Weights & Biases tracking and reproducibility

Use W&B as the primary experiment ledger (`project=liquid-mcdm-lab`; user chooses entity). Enable `report_to="wandb"` when using HF/TRL where appropriate, plus explicit evaluation logs. Track:

- Model checkpoint + revision, trainer type, LoRA config, dtype/quantization, optimizer, LR schedule, seeds and git commit.
- Dataset manifest hash/version, method mix, scenario distribution, prompt version, train/val/test counts, licenses/provenance.
- Train/eval loss, option accuracy, top-1 agreement with exact solver, Kendall tau/Spearman when full ranking exists, NDCG@5/10 and MRR where appropriate, ECE/Brier **only when probabilities and labels support these metrics**.
- Breakdown by MCDM algorithm, ties/near-ties, alternatives count, criteria count, OOD scenario group, numerical scale, and preference weighting.
- GPU device, memory, runtime/step, throughput, GPU-hours or compute-unit estimates when obtainable; don't invent usage data.
- Plots/confusion tables and **redacted** sample predictions, violations, method disagreements, and misranked cases.
- W&B Artifacts for safe model adapters, run configs, dataset manifests/synthetic-generator versions, result reports; keep large private datasets private. Don't upload raw datasets, credentials, or personal data by default.

Add optional W&B **Tables** for per-episode prediction analysis and **Weave** for traceable model-inference evaluations (inputs, scores, decisions, latency) if helpful; distinguish experiment telemetry from application tracing. Support `WANDB_MODE=offline` and later sync. Never write API keys into source files, `.ipynb`, or Git history.

## 9. Proposed repo shape

```text
liquid-mcdm-lab/
  README.md
  PROJECT_BRIEF.md
  pyproject.toml
  .gitignore
  .env.example
  configs/
    data.yaml
    d1_3b_lora.yaml
    evaluation.yaml
  docs/
    research_findings.md
    dataset_card.md
    model_card_draft.md
    decisions/
  src/liquid_mcdm/
    data/                 # loaders, generator, validation, grouping
    mcdm/                 # COPRAS, VIKOR, WASPAS, TOPSIS
    models/               # d1 adapters, custom encoder heads, baselines
    training/             # renderers, trainers, checkpoints
    evaluation/           # ranking, invariances, calibration
    tracking/             # W&B helpers and artifact versioning
    cli.py
  notebooks/
    00_environment_check.ipynb
    01_mcdm_data_and_baselines.ipynb
    02_d1_3b_lora_colab.ipynb
    03_evaluate_and_compare.ipynb
  scripts/
    smoke_test.sh
  tests/
    test_mcdm_reference.py
    test_data_splits.py
    test_prompt_format.py
    test_permutation.py
    test_cli_smoke.py
  reports/
    baseline_results.md
```

## 10. Milestones and acceptance criteria

**M0 — Research + feasibility:** `docs/research_findings.md` compares 3 Liquid checkpoints and two official fine-tuning notebooks with verified model formats/loaders, licensing, known limitations, and Colab environment concerns. A documented decision chooses the first trainer. No expensive training.

**M1 — Data + mathematics:** Correct reference MCDM implementations; tested edge cases; synthetic JSONL/Parquet generator; deterministic split manifest; CLI demo that runs CPU-only; first dataset card.

**M2 — Logging + Colab smoke:** Environment notebook; W&B login/config and offline mode; one tiny W&B run logging configuration, simple metrics, and artifacts; instructions tested in a clean runtime.

**M3 — Baseline + fine-tune:** Record untuned d1-3B performance; successfully overfit tiny labeled sample using the verified training recipe; then run a short, reproducible LoRA fine-tune with saved/reloadable adapter and evaluator. Require explicit permission before launching expensive runs.

**M4 — Meaningful evaluation:** Held-out MCDM correctness by method, sizes, perturbations, near ties, and seed; compare exact solver, numerical neural model, and LightGBM. Report generalization gaps, failure cases and GPU cost; don't confuse algorithm imitation with useful human choices.

**M5 — Optional behavioral extension:** Expedia/alternative dataset is legally and technically accessible; grouped train/validation/test workflow; real click/booking ranking benchmark with position-bias caveat; baseline comparison and interpretation.

Each milestone: executable commands, smoke tests, README instructions, logged W&B run links (where actually produced), and a concise progress summary with known blockers. No fabricated metrics or claims that jobs have been executed.

## 11. Immediate first task for Codex

1. Inspect the repo/environment and the referenced notebooks/model cards using current upstream sources.
2. Write `docs/research_findings.md` and propose a defensible first implementation (TRL vs Unsloth) plus a minimal package/Colab compatibility matrix.
3. Build MCDM reference solvers + tests + 1,000-example generator + simple CLI + W&B instrumentation that works offline.
4. Create an environment-check notebook and a training notebook shell; do not download the 3B checkpoint or launch a costly training run until the user approves and available GPU/VRAM are measured.
5. Report exact next Colab steps and any questions essential to proceed.

**Success criterion:** A Codex-generated repository where the math/data/tracking stack works on CPU, and the user can perform an explicit, controlled first d1-3B LoRA experiment on Colab with reproducible evaluation and W&B runs.
