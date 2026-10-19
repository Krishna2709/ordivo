# Ordivo v2: decision objectives, RLCD and GRPO

Research note, 2026-10-18 local time. Proposed experiments only; no RL trainer,
compatibility result or RL-trained Ordivo checkpoint exists yet.

## Terminology and primary sources

In the user's Jev context, **RLCD** means **Reinforcement Learning for Calibrated
Decisions**, the name used by [TypeSafe's official introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
That page describes its goal and model family; it does not supply a portable,
verified Liquid d1 training recipe. An Ordivo implementation must be labeled
an independently specified calibration-focused method, not a reproduction of
TypeSafe's private implementation or guaranteed equivalent behavior.

There is also an unrelated earlier [RLCD paper on Contrastive Distillation](https://arxiv.org/abs/2307.12950).
Do not confuse its synthetic contrast-based alignment procedure with the
calibrated-decisions method discussed here. Record the exact expansion in configs.

**GRPO**, introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300), is a
policy-optimization method using groups of sampled outputs. The
[official TRL documentation](https://huggingface.co/docs/trl/grpo_trainer)
supports custom rewards and documents generation, reward normalization and
loss options. Pin these explicitly: selecting `GRPOTrainer` alone does not
freeze an objective or prove compatibility with d1.

RLCD expresses a calibration-oriented training approach; GRPO is an optimizer
family. They need not be mutually exclusive, but combining a proper scoring
reward with a particular policy-gradient estimator requires mathematical and
empirical checks. A method name is not evidence that calibrated probabilities
will result on unseen domains.

## Where learning belongs

Ordivo's exact engine evaluates known formulas, feasibility and Pareto relations.
RL should be considered for learned preferences, contextual policies or richer
decision procedures, not as a replacement for cheap deterministic arithmetic.
The first comparison remains supervised choice/ranking training plus strong
specialist baselines. Add RL only when its action, reward and advantage over
direct optimization are precisely defined.

| Objective | Signal | Verification |
|---|---|---|
| Preference agreement | Observed choices or stated preferences | Source-specific held-out choice/ranking metrics |
| Probability quality | Defined events and normalized distributions | NLL, Brier, reliability/ECE with uncertainty |
| Mandatory requirements | Typed constraints and availability | Deterministic feasible-action mask; no violation tradeoff |
| Pareto consistency | Verified numerical criteria and directions | Exact dominance analysis; no invented utility ordering within frontier |
| Robustness | Declared perturbations and preference uncertainty | Recomputed feasible sets/rankings and stability report |
| Explanation fidelity | Calculation trace and evidence references | Each factual claim matches a trace field |

A trade-off-sensitive system should respond to changed preferences. Rewarding
the same winner under every weight perturbation would teach the wrong behavior.
Likewise, multiple non-dominated options need not have equal human preference.

## Calibration-focused branch

For a well-defined categorical event with distribution `p` and observed class
`y`, direct baselines minimize `-log(p[y])` or
`sum_i (p[i] - 1[i=y])**2`. These are proper scoring losses: expected loss is
minimized at the true event distribution under the scoring assumptions. They
do not guarantee finite-sample calibration or calibration under domain shift.

For Expedia, distinguish conditional choice among exactly-one-booking groups
from unconditional purchase probability. For relevance grades, use a separately
defined categorical/ordinal event model; do not treat ranking scores as choice
probabilities. Fit temperature/calibration transformations on validation only.

First compare supervised NLL, supervised Brier where appropriate, and validation
temperature scaling against the same uncalibrated model. Direct differentiable
losses are necessary controls when a full probability vector and labels are
available. Do not call ordinary backpropagation through Brier loss RL.

An RLCD-inspired experiment must document its sampled object, conditional
distribution, proper score, gradient estimator, reference policy and update
budget. With a deterministic d1 distribution for a fixed input, evaluating the
same full-vector Brier reward for several sampled answer aliases gives identical
rewards: a group-relative baseline then produces no task advantage. Sampling
aliases alone does not create a meaningful calibration reward variation.
Any alternative involving stochastic latent decisions, rationales or perturbations
changes the experiment and must justify that change explicitly. Do not bolt on
unobserved hidden reasoning and claim the current d1 interface supports it.

Adding utility, correctness bonuses, reward clipping or group normalization can
change a proper-score objective. Evaluate those choices as new objectives;
do not inherit calibration guarantees from one component of a composite reward.

## GRPO branch

Start from a supervised checkpoint. The action must be named: a candidate alias,
a complete ranking, or a structured decision procedure. These have different
probabilities, costs and verifiers; do not mix them within one undocumented run.
GRPO groups are sampled rollouts of the same decision request, not the candidate
set and not a dataset's query grouping used for train/test splitting.

Define rewards from training-only observations or a declared simulator/exact
oracle. A logged booking is an observational label, not a known counterfactual
reward for every possible recommendation. Optimizing agreement with logged
labels does not establish causal improvements to customer outcomes. No inverse-
propensity or off-policy-value claims without a justified logging/overlap model.

Apply known hard constraints through action masks and an independent checker.
When the feasible set is empty, emit the explicit no-feasible result; do not
sample an invalid option and compensate with another reward. Other reward
components, weights and scaling must be versioned, with per-component telemetry
and ablations. Monitor invalid actions, reward variance, duplicate sampled
choices, entropy collapse, KL drift, reward hacking and costs.

For single-choice rewards, all samples can be the same/correct/incorrect and
yield zero task advantage. Report that frequency. With only a few fully scored
actions, direct expected-reward or supervised optimization may be simpler and
lower-variance; compare before adopting rollout-heavy optimization.

## d1 compatibility experiment required before adoption

The current [renderer](../../src/liquid_mcdm/training/rendering.py) supervises an
answer alias plus EOS. The upstream [readout](../../src/liquid_mcdm/_vendor/d1_prompt.py)
max-pools supported token aliases for each option and applies a restricted
softmax. Ordinary token-generation probabilities need not equal that distribution.
Moreover, the vendored helper converts logits to Python floats for inference;
it is not a differentiable calibration-training head.

A remote experiment must therefore establish:

1. Exact action vocabulary, alias groups, probability normalization, EOS/mask
   rules and candidate remapping, including unavailable/unknown candidates.
2. Equality of sampling policy and the action log probabilities used for updates
   and importance ratios. Sum-of-token probability, max-pooled option score,
   and canonical single-token restriction are different policies.
3. A differentiable score/readout path with gradient checks; match the declared
   serving distribution or version the policy change explicitly. Do not edit
   v1's vendored readout to hide a change in semantics.
4. Pinned loader/trainer compatibility, reference-policy behavior, finite loss,
   nonzero expected gradients, bounded GPU use, save/reload and repeated inference.
5. A short matched SFT versus RL comparison using validation for tuning, then
   untouched data for reporting. Log realized steps, tokens and GPU-hours.

## Experimental sequence and stopping criteria

Build real-data baselines and deterministic decision analysis first. Then run
SFT/ranking plus direct calibration controls, followed by small remote GRPO
and calibration-focused feasibility studies. Advance only the branch that
improves a declared endpoint at acceptable inference/training cost without
breaking hard constraints or degrading key domains beyond declared tolerances.

Freeze numerical acceptance tolerances and budgets in the later run config;
none is being approved or launched by this research note. Compare matched
initialization, downstream data, seeds, optimizer-update/token budgets and
evaluation populations. Report negative results and compute overhead as well
as improvements. W&B should keep reward, ranking, calibration, feasibility
and cross-domain metrics separate.
