# Ordivo v2 decision-learning research design

Accepted direction: 2026-10-18. This operationalizes the user-provided
`Ordivo_v2_model_training.md`; it does not claim that v2 training has occurred.
Ordivo is an MCDM intelligence framework across domains. The initial neural
backbone remains `LiquidAI/d1-3B`; Ordivo v1 artifacts remain the frozen reference.

## Execution order

1. Implement and remotely test weight resolution and numerical verification.
2. Freeze real-data adapters, grouped splits, rendering, and task-specific metrics.
   Train a supervised d1 baseline alongside exact MCDM and domain-appropriate
   LightGBM/choice/ranking controls. Preserve v1 separately.
3. Branch from the same v2 SFT checkpoint into direct proper-score optimization,
   RLCD-inspired probability-reporting experiments, and verifier-guided GRPO.
   These are controlled comparisons, not a mandatory serial RL curriculum.
4. Select using validation, fit final calibration on a separate calibration split,
   and evaluate once on untouched test groups and held-out domains.
5. Add DPO when authentic preferred/rejected decision pairs exist. Observed
   choices can support supervised learning from stage 2; they need not wait for RL.

This resolves the two different stage orders in the supplied plan. Mandatory
serial optimization would confound which algorithm helped and add GPU cost.

## Policy interfaces and compatibility

| Branch | Action / output | Objective | Required evidence before scaling |
|---|---|---|---|
| SFT reference | Alias token plus EOS | Existing completion CE | Mask, tokenizer, trainable layers, save/reload parity |
| Direct probability control | Option distribution | Differentiable NLL or Brier | Tensor-only alias aggregation, finite nonzero gradients, parity with inference |
| RLCD-inspired | Stochastic probability report | Proper score, explicit report-policy gradient | Distribution density, sampling, log probability, gradient estimator and variance audit |
| GRPO | Structured generated decision action | Verifier rewards over sampled completions | Generation/log-prob parity, masks, per-group reward variation, reload parity |
| DPO | Preferred/rejected outputs | Pairwise preference objective | Genuine provenance, matching prompts, suitable action representation |
| Ranking | Candidate utilities or ordered list | Pointwise/pairwise/listwise supervised loss | Grouped labels, ties, complete candidate coverage |

Current `training/rendering.py` and vendored `d1_prompt.readout` are inference
utilities. Their conversion to Python floats breaks autograd. Restricted
option probabilities use alias-group aggregation; they are not full-vocabulary
sequence probabilities. Do not feed those numbers to TRL as generated-token
log probabilities. A differentiable port must preserve the pinned readout,
including alias grouping, and verify gradients on RunPod.

For standard TRL GRPO, a structured action such as a method request or candidate
selection needs a sequence-policy adapter and supervised action examples. The
numerical engine executes it. An option-policy optimizer is a separate method,
not automatically `GRPOTrainer`. Pin TRL version, model class, precision,
loss_type, reward scaling, generation count, KL settings, batch divisibility,
maximum prompt/completion lengths and truncation policy in every run.

If several sampled aliases are scored using the same deterministic full-vector
Brier reward, all samples receive identical rewards and the group-relative
advantage vanishes. Probability-report exploration must actually change the
reported distribution. Compare its gradients and sample efficiency with direct
backpropagation before paying for scale.

## Primary-source research

[Laya's model card](https://huggingface.co/convaiinnovations/laya#training)
describes Gaussian perturbations to option logits and REINFORCE with a group-mean
baseline. Its [pinned reward implementation](https://github.com/NandhaKishorM/laya/blob/d74e8be46ef5241e4ba2b16cd25e896f24f71373/laya/common.py#L660)
combines log and spherical scores and adds an ordinal ranked-probability term.
It clamps the log term at -9.21. That clamp means we must not inherit an
unqualified strict-propriety guarantee from the unclipped mathematical score.
This is a concrete reference recipe, not demonstrated compatibility with d1.
The card also reports overconfidence and post-hoc calibration; we still need
our own controlled comparison against SFT and direct scoring losses.

[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces GRPO.
[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) documents custom rewards
and rollout interfaces; its available loss variants make explicit configuration
necessary. [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer) documents the
prompt/chosen/rejected contract. References checked 2026-10-18; pin exact library
versions and source commits again in the actual experiment receipt.

## Data and evidence boundaries

Use Expedia plus the diverse suite in [the dataset register](V2_DATASET_REGISTER.md).
Expedia booking choice, Swissmetro stated choice, ESCI relevance, MSLR relevance
and SUSHI explicit rankings remain distinct tasks. Never label a relevance score
as purchase probability. Source acquisition and rights records, complete candidate
sets, grouped splits, missingness, position exposure and context truncation belong
in each adapter manifest. Expedia access still requires an authenticated source;
its absence does not block testing the numerical layer.

The current implementation supplies the deterministic boundary only. No learned
weight estimator, real-data adapter, RL trainer or v2 checkpoint is implied.
See [experiment matrix](EXPERIMENT_MATRIX.md) and [verification](DECISION_VERIFICATION.md).
