# Verifier-guided reward design

Status: research specification, not an implemented trainer or validated reward.
The numerical verifier is implemented separately. Every experiment must name its
action space, observation, reference labels, feasible set and reward population.

## Hard constraints and action validity

Mask known infeasible actions and reject them at serving time regardless of score.
Unknown feasibility requires more evidence or abstention. For training rollouts,
record invalid JSON, unknown IDs, infeasible selections and missing evidence as
separate failures. A failure reward does not replace system enforcement.

Do not renormalize observed labels after filtering away the chosen outcome.
Record a data/specification conflict and repair its cause or exclude it under a
predeclared policy. Model-generated constraints cannot silently rewrite the
reference problem. Method choice must remain within the caller's allowed set.

## Components

| Component | Definition / admissible evidence | Failure to avoid |
|---|---|---|
| Ranking | NDCG@k or tie-aware pair agreement on declared reference labels | Converting a single booking into a complete human ranking |
| Regret | (best feasible utility - chosen utility) / declared positive utility range | Calling observed choice a counterfactual utility oracle |
| Mathematics | Recomputed score/ranking/constraint claims agree | Rewarding omission of all numerical claims |
| Consistency | Equivalent permutations/unit changes preserve mapped outputs | Rewarding unchanged decisions after genuinely changed preferences |
| Calibration | Negative Brier or log score on a defined event | Treating one sample's ECE or verbal confidence as proper reward |

Utility/regret needs an identified reference utility. A constant range gives
zero regret only when all feasible utilities are equal; an unavailable utility
makes the component inapplicable. VIKOR Q has reversed score orientation and
its unique-choice conditions must be respected. Report solver regret separately
from behavioral accuracy. A heuristic blend is not a human welfare measure.

The supplied coefficients 0.30 ranking + 0.25 regret + 0.20 mathematics + 0.15
consistency + 0.10 calibration define an **illustrative ablation**, not the default
validated objective. Freeze component scales and applicability per task before
training. Never renormalize a different set of active components per sample
without naming that changed objective. All components and masks are logged.

For mathematical rewards, the action schema names required claims. Missing a
required claim fails completeness; the serving verifier's null check is not a
training success. Tool execution already guarantees arithmetic, so rewards
should favor useful, valid actions and correct tool arguments rather than
teaching the model to imitate long calculation prose.

## Calibration and GRPO are separate mathematical questions

Proper scores optimize reported event distributions in expectation. The
positive weighted sum of proper scores remains proper under their assumptions;
adding a ranking bonus, clipping scores, or applying a sample-dependent
normalization need not preserve that optimum. Report calibration empirically.
Use full normalized option distributions and a separate calibration holdout.

A fixed d1 report scored repeatedly with a full-vector loss yields zero
within-group advantage. For report-policy RL, define a stochastic report and
its density, stop-gradient conventions, baseline and KL term explicitly.
For sequence GRPO, use the actual completion policy log probability. Record
zero-variance group fraction, invalid actions, reward distribution, clipping,
KL, entropy, gradients, and generated token counts. All-zero reward groups are
a diagnostic, never evidence that learning worked.

References: [proper scoring rules, Gneiting & Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf),
[DeepSeekMath](https://arxiv.org/abs/2402.03300),
[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer).

## W&B experiment record

Every real optimization run logs the code/base/adapter revisions, split and
source hashes, configuration, seeds, environment freeze, actual GPU, wall time,
provider price snapshot and estimated compute cost (separate from billed cost).
Log per-component reward statistics, failure counts with denominators, gradient
norms, group-variance statistics, checkpoint artifacts, validation ranking and
probability metrics, and calibration parameters. Final test metrics attach to
a frozen checkpoint after selection. Do not upload raw restricted data or secrets.

The verifier has no telemetry side effects. Existing v1 tracking remains intact.
The new v2 RL reward logger is still to be implemented with the trainer; no
v2 training metrics are currently claimed. Offline W&B must be synchronized and
its remote artifact IDs verified before claiming cloud persistence.
