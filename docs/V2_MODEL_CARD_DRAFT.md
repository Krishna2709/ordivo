# Ordivo v2 model card — draft, unreleased

**Research status:** a preliminary two-domain v2 adapter exists in a **private** research repository; no held-out neural performance or public release claim is established here. This is the intended public card template; replace remaining pending fields with immutable receipts before publication. [Release criteria](V2_RELEASE_CRITERIA.md) control promotion. `blackOrder/Ordivo-v1` and its [synthetic-only results](../reports/V2_RESEARCH_AUDIT.md) are a separate historical model.

A bounded remote compatibility check at integrated source commit
`22707ed34fd125e96aade9123f9baaf34cd64a18` completed one A40 BF16
forward/backward pass on a fixed prompt: finite loss, 166 trainable tensors with
nonzero gradients, unavailable-option probability zero, and zero reload delta
on that check. It took **no optimizer step** and produced no v2 trained adapter
or behavioral performance evidence. The remote/local metadata SHA-256 is
`2a9500911f60cf50635c31277b8994c8cebef92ab455174d12b9dea6f6317406`.
An actual [W&B cloud run](https://wandb.ai/black-order/liquid-mcdm-lab/runs/2f8r8pwf)
holds aggregate research telemetry. A Hugging Face research repository named
`blackOrder/Ordivo-v2-research` is **private** and has a readback-verified
22-artifact allowlisted classical evidence bundle. A separate preliminary
40-step Swissmetro/ESCI adapter was then retained privately at Hub commit
`bf1d425b4e6ffc392121d8fb0776e41b1af48c4a`; its nine allowlisted files
plus artifact manifest passed forced SHA-256 readback. The
[private research card](../hub/v2_research_model/README.md) records the exact
loading path and limits. Neither artifact set is a public v2 model release.

The pilot ran at source commit `854de0877c8e7e380b7353a578e95ba9865d4e25`
with config SHA-256 `c152e64d839082531d4ece9e66d330fa35ffddfbc442685742b9fd171de5462b`.
Its best checkpoint was step 40; selection-validation loss decreased on both
domains, and the reload probe delta was zero. ESCI prompt overflow removed
29 of 100 selected validation groups and 325 of 1,000 selected training groups
from neural loss. The [receipt](../hub/v2_research_model/README.md) is a
training-selection diagnostic only; held-out neural accuracy, NDCG and
calibration remain pending. Its [W&B pilot run](https://wandb.ai/black-order/liquid-mcdm-lab/runs/bnsgv73p)
is private research telemetry.

## Intended system

Ordivo v2 investigates a shared neural component for grouped candidate scoring across observed choice, stated choice, graded relevance, and explicit ranking. A deterministic layer handles declared MCDM formulas, hard constraints, feasible alternatives, Pareto checks and weight scenarios. The neural output proposes scores or choices; the verifier alone establishes arithmetic validity. No authoritative criterion weights can be inferred from a decision matrix without a user, policy, or preference evidence source.

The preliminary adapter uses [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) at revision `051bcc464b01b9f92942b364d9586b0ef5912432`. The intended public package must include adapter weights, tokenizer and prompt/readout versions, model config, dependency lock, source commit, checksums and a verified base-plus-adapter reload procedure. The adapter alone is not a complete model. Distribution must comply with the base [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/main/LICENSE) and all applicable data-derived restrictions; final license and notices are pending review.

## Intended uses and limits

Research comparisons of candidate ordering, conditional choice, calibration and verifiable numerical decisions are in scope. Tasks must be declared: a click/booking event, a stated selected mode, a relevance grade, and an explicit preference ranking have different meanings. Human review and approved policy remain necessary for consequential procurement or healthcare decisions. The model is not an exact MCDM calculator, a causal estimate of preference from observational clicks, or an autonomous policy authority. It should not be used to infer hidden personal preferences, to treat relevance as purchasing probability, to recommend unavailable candidates, or to act on unknown constraints.

## Interface planned for evaluation

Input: `schema_version=ordivo-behavioral-v1`, `task_type`, allowlisted context, declared criteria/constraints/preferences, and a complete variable-length set of alternatives with typed features and availability. Candidate IDs serve only as output keys and are remapped to aliases in neural prompts. Outcome labels, exposure/position, provenance, source/group identifiers and original row order cannot enter predictor features. Numeric units and missingness must be explicit; weights can be `null`.

Output: one finite score per eligible candidate keyed by ID, explicit unavailable/failed status, and task-specific probability only where a named event and normalization are defined. Ranking a score vector is not proof of listwise training. The deterministic decision report separately records weight source, feasibility, exact method and parameters, computed score/trace, uncertainty or sensitivity, and verification status. A failed or truncated group is an error, not a reduced successful prediction.

## Training and evaluation record

| Field | Public release value |
|---|---|
| v2 adapter repository/revision/SHA-256 | Private pilot `blackOrder/Ordivo-v2-research` at `bf1d425b4e6ffc392121d8fb0776e41b1af48c4a`; adapter safetensors SHA-256 `54e90e7b5c4dad3a470345c8462aa4ad7527829b702c81aa2a45ded7475066c7`. Public adapter pending. |
| Executed training commit, seeds, base revision | Pilot `854de0877c8e7e380b7353a578e95ba9865d4e25`, seed 42, d1 revision `051bcc464b01b9f92942b364d9586b0ef5912432`; final run pending. |
| Source and split manifest hashes; rights records | Pending |
| Supervision mixture, task losses, trainable parameters and budget | Pending |
| Train/selection/calibration/final-test counts by domain and group | Pending |
| Full candidate-set and context overflow accounting | Pending |
| W&B/offline run, hardware, wall time and billed cost | Pending |
| Held-out within-domain and unseen-domain results | Pending |
| Reload, permutation, verifier and failure-case receipts | Pending |

Required comparisons include exact math reference, domain-appropriate classical and heuristic baselines, base d1, frozen v1 transfer, matched behavioral-only training, and MCDM-pretrained continuation with equal downstream conditions. An extra-training control is needed before attributing a gain to MCDM content. A public shared-model generality claim requires at least three real domains covering choice and relevance and actual entire-domain holdout evidence; a two-domain baseline is preliminary. Final comparisons use multiple matched seeds and paired group intervals. Calibration uses an independent held-out calibration set and specifies the event population. ESCI/MSLR results describe source-sampled judged candidate pools, not all possible products or pages. Report violations, unknowns and failed queries with denominators. See the [experiment matrix](EXPERIMENT_MATRIX.md).

## Known and anticipated failure modes

The v1 synthetic pilot raised alias/order and calibration concerns; it provides no v2 evidence. For v2, inspect long or variable candidate sets, missing or incomparable units, repeated entities across splits, unavailable options, zero-positive and multi-positive groups, position bias in clicks, domain shift, language imbalance, near-ties, and conflicting/absent preferences. Record actual failures with sanitized examples only after remote evaluation. A lack of suitable evidence should yield `preferences_unknown` or explicit scenarios, not fabricated weights or a false `verified` label.

## Reproduction and rights

Public artifacts will include exact code/config/checksum references, a base-plus-adapter loading example, and commands for the permitted evaluation subset. Restricted source rows and individual predictions require source-specific authorization and remain out of the public package. The [v2 data card](V2_DATA_CARD_DRAFT.md) and per-source acquisition records govern source rights; an upstream dataset license cannot be assumed to grant rights to publish trained derivatives. The report may publish aggregate metrics only after rights and disclosure review.
