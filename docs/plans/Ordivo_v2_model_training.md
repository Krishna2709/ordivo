You're identifying the most important research problem for Ordivo: we don't just want a model that predicts the winning alternative. We want a model that learns defensible decision behavior, respects constraints, handles uncertain preferences, and produces mathematically verifiable results.

That changes our training strategy.

My recommendation is SFT → verifier-guided GRPO → preference calibration, with RLCD as a research comparison. But reinforcement learning should optimize decision quality, not merely encourage the model to produce longer reasoning.

## 1. Which training algorithm should we use?

| Algorithm                                              | What it learns                                                       | Recommendation                              |
| ------------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------- |
| SFT (supervised fine-tuning)                           | Learns from labeled examples                                         | Necessary starting baseline                 |
| RLCD (Reinforcement Learning for Calibrated Decisions) | Improves probabilistic decision quality through scoring-rule rewards | Strong candidate for decision probabilities |
| GRPO (Group Relative Policy Optimization)              | Improves a policy using relative rewards across sampled outputs      | Primary RL experiment                       |
| DPO (Direct Preference Optimization)                   | Learns from preferred versus rejected decisions                      | Useful when we have expert preference pairs |
| Reward-model-based RL                                  | Optimizes a learned model of decision quality                        | Later, when real outcome data is sufficient |

These are not interchangeable.

GRPO is an optimization algorithm, not a guarantee of reasoning. RLCD focuses on decision calibration, while DPO requires preference comparisons.

For Ordivo, the strongest approach is likely a combination of objectives rather than one universal algorithm.

## 2. Proposed training architecture

Decision input

Criteria · alternatives · constraints · preferences · evidence

Liquid d1-3B + decision adapter

Predict candidate scores, preferences or decision actions

Deterministic verifier

MCDM formulas · hard constraints · arithmetic · Pareto dominance

Validated decision + feedback signal

Correctness, regret, ranking quality and calibrated confidence

The verifier is essential because a neural model's explanation is not proof that it actually followed the calculation it describes.

Instead of trusting a generated explanation, we should independently recompute intermediate and final numerical results.

## 3. How should reward-based training work?

I would use a verifier-based reward design, but hard constraints must remain enforced by the system, not merely encouraged through rewards.

An illustrative reward function:

\\[ R = 0.30R\_{\text{ranking}} +0.25R\_{\text{regret}} +0.20R\_{\text{math}} +0.15R\_{\text{consistency}} +0.10R\_{\text{calibration}} \\]

These coefficients are experimental, not validated hyperparameters.

| Reward                   | What it checks                                                     |
| ------------------------ | ------------------------------------------------------------------ |
| Ranking                  | Agreement with reference rankings or real preferences              |
| Regret                   | How much utility is lost relative to the best feasible alternative |
| Mathematical correctness | Whether reported numerical claims match exact solvers              |
| Consistency              | Whether equivalent inputs give equivalent decisions                |
| Calibration              | Whether predicted probabilities match observed frequencies         |

For decisions with hard constraints, infeasible selections receive a failure reward and are rejected at inference time.

One technical qualification: standard GRPO optimizes generated token sequences. Liquid d1's one-pass option-scoring interface does not automatically fit that workflow. We would need to implement a compatible policy objective or introduce a structured action-generation interface before using TRL's GRPO trainer.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



I would have Codex verify that compatibility before promising GRPO fine-tuning on d1-3B.

## 4. What happens when criterion weights are `null`?

This is one of the most important distinctions in MCDM.

There is no universally correct set of weights that the model can infer from the decision matrix alone.

Consider two people choosing the same hotel. One values affordability; another prioritizes location. Both may make rational but different decisions.

Ordivo needs a weight-resolution policy.

| Situation                                       | How weights are obtained                                                    |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| User explicitly supplies weights                | Use those weights                                                           |
| Organization has an approved preference profile | Retrieve and apply that profile                                             |
| Historical preference labels exist              | Predict a weight distribution or contextual utility                         |
| Objective weighting method requested            | Compute CRITIC, entropy weights, etc.                                       |
| No evidence or policy exists                    | Return `preferences_unknown` and request clarification or explore scenarios |

Entropy and CRITIC weights are data-derived weighting conventions, not proof of human priorities.

For learned preferences, I would model:

\\[ p(w\mid c,H) \\]

Where \\(c\\) is current context and \\(H\\) is relevant historical preference evidence.

Rather than inventing one precise weight vector, Ordivo could estimate several plausible profiles and show how recommendations change across them.

For example, it could say:

> Vendor A ranks highest when cost receives high priority. Vendor B ranks highest when quality and reliability dominate. There isn't enough preference evidence to select one definitively.

That would be more scientifically defensible than making up weights.

## 5. How do we know the model's reasoning is correct?

I would not train it to generate unrestricted chain-of-thought explanations and then assume those reflect its actual calculations.

Instead, require a verifiable decision trace.

Proposed decision trace

```

{
  "decision": "Vendor B",
  "weight_source": "approved_policy",
  "method": "WASPAS",
  "checks": {
    "constraints_satisfied": true,
    "weights_normalized": true,
    "scores_verified": true,
    "pareto_check": "passed"
  },
  "sensitivity": {
    "tested": true,
    "winner_stability": 0.87
  },
  "status": "verified"
}
  
```

Illustrative schema only. The stability value and verification status must come from actual computations, not model-generated assertions.

Every verification flag should be produced by code independently of the neural model.

The model can propose a decision, select a method, estimate preferences, or request missing evidence. The mathematical engine must verify the calculations.

This also means we can train on decision actions and verifiable intermediate outputs without trying to prove correctness from hidden neural reasoning.

## 6. The actual research plan I recommend

1. Stage 1 — Supervised decision learning. Train Liquid d1-3B using structured examples with explicit constraints and preference sources. Establish reliable supervised baselines before RL.

2. Stage 2 — RLCD-style probability optimization. Experiment with proper scoring-rule rewards to improve confidence estimates, followed by independent calibration. Laya's published RLCD recipe is a useful reference, but it does not guarantee calibrated probabilities automatically.&#x20;

   [image](https://www.google.com/s2/favicons?domain=https://laya.studio\&sz=32)

   Laya Studio

3. Stage 3 — Verifier-guided GRPO. Run a separate experiment for structured decision actions, mathematical correctness, constraint handling, and tool use. TRL supports custom reward functions and tool-using agent rollouts, provided the model and training interface are compatible.&#x20;

   [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

   GitHub

   +1

4. Stage 4 — Real preference learning. Use actual choice data, and eventually expert comparisons, to adapt contextual utilities. Add DPO only where suitable preference pairs exist.

## 7. What Codex needs to understand

# Ordivo — Decision Learning and RL Research Requirements

**Primary objective:** Build a generalizable, constraint-aware, preference-sensitive neural MCDM system whose decision outputs can be mathematically verified.

## Research requirements

1. Preserve the existing SFT baseline and use `LiquidAI/d1-3B` as the initial backbone.
2. Investigate RLCD-style proper-scoring optimization, verifier-guided GRPO, DPO, and supervised ranking losses. Compare their suitability rather than assuming reinforcement learning is superior.
3. Audit whether the existing d1 option-logit interface is compatible with each RL algorithm. Do not assume standard TRL `GRPOTrainer` works directly.
4. Implement an independent numerical verifier for MCDM formulas, hard constraints, Pareto checks, score validation, and sensitivity calculations.
5. Implement explicit weight-resolution modes: user-specified, approved policy, learned preferences, objective weighting, and unknown.
6. Never silently infer authoritative weights when preference evidence is absent. Track weight provenance and uncertainty.
7. Design structured, independently verifiable decision traces. Do not treat generated explanations as proof of mathematical reasoning.
8. Evaluate mathematical accuracy, constraint violations, ranking regret, calibration, decision consistency, generalization, and context-length robustness.
9. Test whether each training method improves held-out performance against SFT-only, exact MCDM, and LightGBM baselines.
10. Log reward components, verification failures, calibration metrics, ranking performance, model checkpoints, and computational costs in W&B.

## Required deliverables

- `docs/RL_RESEARCH_DESIGN.md`
- `docs/WEIGHT_INFERENCE_POLICY.md`
- `docs/DECISION_VERIFICATION.md`
- `docs/REWARD_DESIGN.md`
- `docs/EXPERIMENT_MATRIX.md`

**Execution principle:** Train decisions and verifiable actions—not merely plausible explanations. Preserve deterministic guarantees where mathematical computation is exact.

The research direction I'd commit to is:

Liquid d1-3B + supervised decision learning + calibrated probability training + verifier-guided RL + exact MCDM execution.

But we should only scale GRPO after confirming that its structured-action objective materially improves real held-out decisions. Otherwise, we risk spending GPU resources teaching a model to produce convincing decision traces without making better decisions.

That distinction—between appearing to reason and producing verifiably correct, useful decisions—should be foundational to Ordivo.