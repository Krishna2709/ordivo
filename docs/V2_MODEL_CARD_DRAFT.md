# Ordivo v2 model card — draft, unreleased

**Research status:** a preliminary two-domain v2 adapter exists in a **private** research repository; its bounded held-out neural probe measured negative/mixed results. A separate three-source S0b training run has completed with better selected-validation loss, and a bounded S0b held-out probe now shows mixed source-specific outcomes; the private S0b adapter passed file-level readback. No public release claim is established here. This is the intended public card template; replace remaining pending fields with immutable receipts before publication. [Release criteria](V2_RELEASE_CRITERIA.md) control promotion. `blackOrder/Ordivo-v1` and its [synthetic-only results](../reports/V2_RESEARCH_AUDIT.md) are a separate historical model.

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
training-selection diagnostic only. A later bounded held-out probe measured
lower adapted Swissmetro accuracy (0.66 versus base 0.68 on 450 scenarios) and
lower adapted ESCI NDCG@10 (0.752276 versus base 0.778290 on 27 admitted of
50 selected query groups); 23 ESCI groups overflowed. Swissmetro NLL improved
(0.805878→0.758958), while Brier worsened (0.459254→0.462190). This is mixed
and partly negative preliminary evidence, not a protected release test or a
generality claim. The [sanitized aggregate](../reports/runs/v2-neural-heldout-20261019/aggregate.json)
records exact denominators and hashes; calibration remains pending. Its [W&B pilot run](https://wandb.ai/black-order/liquid-mcdm-lab/runs/bnsgv73p)
is private research telemetry.

## Intended system

Ordivo v2 investigates a shared neural component for grouped candidate scoring across observed choice, stated choice, graded relevance, and explicit ranking. A deterministic layer handles declared MCDM formulas, hard constraints, feasible alternatives, Pareto checks and weight scenarios. The neural output proposes scores or choices; the verifier alone establishes arithmetic validity. No authoritative criterion weights can be inferred from a decision matrix without a user, policy, or preference evidence source.

The preliminary adapter uses [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) at revision `051bcc464b01b9f92942b364d9586b0ef5912432`. The intended public package must include adapter weights, tokenizer and prompt/readout versions, model config, dependency lock, source commit, checksums and a verified base-plus-adapter reload procedure. The adapter alone is not a complete model. Distribution must comply with the base [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/main/LICENSE) and all applicable data-derived restrictions; final license and notices are pending review.

## Intended uses and limits

Research comparisons of candidate ordering, conditional choice, calibration and verifiable numerical decisions are in scope. Tasks must be declared: a click/booking event, a stated selected mode, a relevance grade, and an explicit preference ranking have different meanings. Human review and approved policy remain necessary for consequential procurement or healthcare decisions. The model is not an exact MCDM calculator, a causal estimate of preference from observational clicks, or an autonomous policy authority. It should not be used to infer hidden personal preferences, to treat relevance as purchasing probability, to recommend unavailable candidates, or to act on unknown constraints.

The separate S0b pilot at `945fe6e07df4916f202108d7a56a543041808315`
completed 90/90 optimizer updates (30 per source), best selected-validation
step 90, with source-macro loss 2.6071312→2.2937011. Swissmetro loss fell
1.1358083→0.8775658 on 549 eligible validation episodes; ESCI
3.4676457→2.8772386 on 100 eligible groups; Expedia conditional-booking
choice 3.2179395→3.1262991 on 70/100 selected groups (30 had no target
supervised event). The receipt SHA-256 is
`85426d547c1fbc88cc52cf9c4fc9c9bfac99fb3bfbd7b256b177c3f069e24565`.
ESCI training overflow excluded 4/1,000 selected groups; Expedia target
exclusions were 315/1,000 selected training groups. Expedia used only the
predeclared 20k/2k/2k/5k complete-search sample, not the full source. This training loss is selection-validation evidence, not held-out accuracy, booking probability, calibration or cross-domain generality. Private S0b adapter readback passed at `blackOrder/Ordivo-v2-research` commit `df6405c575629dfaf9083f7b068ab8ef34b6eeb3`, directory SHA-256 `e9bc879b31a32e5546d238345babd7d9fc5eedad7e015510804e03fde8a77c50`; no S0b held-out inference is established by storage.

A separate [exploratory paired S0b probe](../reports/runs/v2-s0b-20261019/heldout-aggregate.json) at evaluation source
`10232089368296680c75461da2dffe5254ae7590` has aggregate SHA-256
`86753c1356cdf8524a3ea28a2f5b4ff31fb4ef20afaf3f96d6db1cb0fd4b086d`.
Swissmetro base→S0b accuracy fell 0.680000→0.662222 on 450/450 scenarios;
NLL improved 0.805878→0.782651 while Brier worsened 0.459254→0.463704.
ESCI NDCG@10 rose 0.705590→0.714332 on 50/50 selected groups, zero
overflow. Expedia conditional-booking accuracy rose 2/29→4/29 and Brier
improved 0.967797→0.941650, while NLL worsened 2.961689→3.016047;
21/50 selected searches lacked the exactly-one-booking target. These are
small, already-exposed test cohorts with no intervals/calibration/multi-seed or
protected-release test. Their 12 private files plus manifest passed forced-
download SHA readback at Hub commit `9f7110952dbd64aeb8eaa703a328cf196765fa8f`
(retention-manifest SHA-256
`2ab1be61a5709ca8b736d32c9085c8e152a58e65b208e887beb2902b358113c4`). A [private-ID aggregate intersection](../reports/runs/v2-s0b-20261019/esci-old27-match.json) confirms the old 27 ESCI groups are all in the
new 50; on those same 27, base NDCG@10 is 0.778290, S0 0.752276, and S0b
0.787842. The other 23 alter the new mean, so comparing the unpaired 27-
and 50-group means directly is invalid. Intersection receipt SHA-256
`494a1a0a6f2795a3c4f9962fc58f6194a7508f3aa5e9a7cc9ad24cde17479672`.

A corrected S0b candidate-order diagnostic on model-selection validation
pairs found top-choice flips in 65/180 Swissmetro scenarios (36.11%), 10/20
ESCI groups (50%) and 12/15 Expedia target-eligible searches (80%; five of
20 selected searches lacked the exactly-one-booking event), with zero
overflow/failures. Mean probability L1 changes were 0.34010, 0.30022 and
0.22336 respectively. This remains a material positional sensitivity despite
train-only shuffling; the diagnostic does not measure accuracy or establish
causation. Receipt SHA-256
`9be6417337ac574cc6fd38b95fe7f5c0b6e873848601db04eed3993110c344c4`.
On the exact same 180 Swissmetro scenarios, S0 flipped 52 versus S0b 65.
All 12 old eligible ESCI groups are in the new 20; on those same 12, flips
were S0 7 versus S0b 9. Mean L1 fell in both matched cohorts, so its
direction alone does not imply fewer decision changes. Match receipt SHA-256
`25689eff2d9aecebeed627c0bfdfee04f54e15bafce5cd46828a182369968207`.
The corrected source suite passed 255 tests/zero skips at `b6214613`, archive
SHA-256 `4be1e9c6587ea460435b8732bdb22a7944aa33a7584e6bbc95bdafcb9b912c41`.

## Research inference interface

Input: `schema_version=ordivo-behavioral-v1`, `task_type`, allowlisted context, declared criteria/constraints/preferences, and a complete variable-length set of alternatives with typed features and availability. Candidate IDs serve only as output keys and are remapped to aliases in neural prompts. Outcome labels, exposure/position, provenance, source/group identifiers and original row order cannot enter predictor features. Numeric units and missingness must be explicit; weights can be `null`.

The integrated label-free `render_predictor_input` rejects supervision,
exposure and provenance fields and requires the exact source feature allowlist;
it shares the executed pilot's prompt and max-alias option-readout core. Remote
source tests passed at `bd24814429211ad4dad4e9a58ba0137daa0bc76b` (244
tests, zero skips; archive SHA-256
`778a3689f90c44afa572fc4ecd968a8853e53f5c5bb81e3356338485dd57c2a0`).
The [private research card](../hub/v2_research_model/README.md) gives an
illustrative label-free request and pinned PEFT loading path. Its verified
private-adapter fixture playthrough returned train 0.24508503, Swissmetro
0.75491500 and unavailable car 0.0 (345 input tokens, finite scores summing
to one; metadata receipt SHA-256
`9f9832eddaf39df94079cd82a9458839ed4be28e13cde70727943e42c25c3137`).
These constructed-input scores do not measure benchmark quality or calibration;
this is not a production serving API.

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
| Held-out within-domain and unseen-domain results | Bounded paired within-domain probe measured negative/mixed results; protected full within-domain, entire-domain transfer, intervals and multiple seeds pending. |
| Reload, permutation, verifier and failure-case receipts | Pending |

Required comparisons include exact math reference, domain-appropriate classical and heuristic baselines, base d1, frozen v1 transfer, matched behavioral-only training, and MCDM-pretrained continuation with equal downstream conditions. An extra-training control is needed before attributing a gain to MCDM content. A public shared-model generality claim requires at least three real domains covering choice and relevance and actual entire-domain holdout evidence; a two-domain baseline is preliminary. Final comparisons use multiple matched seeds and paired group intervals. Calibration uses an independent held-out calibration set and specifies the event population. ESCI/MSLR results describe source-sampled judged candidate pools, not all possible products or pages. Report violations, unknowns and failed queries with denominators. See the [experiment matrix](EXPERIMENT_MATRIX.md).

The classical ESCI benchmark's canonical-ID tie rule gave LambdaRank/lexical
NDCG@10 of 0.659554/0.649744 on 14,496 groups. An exact uniform-within-tie
analysis of the same saved score files gave 0.656403/0.646555, with 13,360
and 13,730 metric-sensitive queries respectively. This is a tie-convention
sensitivity check, not a second model run or an uncertainty interval. The
[sanitized analysis](../reports/runs/v2-esci-ties-20261019/aggregate.json)
preserves its executed hashes; no superiority claim follows from either gap.

A matched bounded C1 direct-Brier versus R0 stochastic-report continuation
also ran for 40 updates each from the same pilot adapter and schedule. The
shared starting selection-validation macro loss was 1.5817367; best
**post-update** C1/R0 checkpoints at step 10 had 1.6459532/1.6163764,
respectively. Both degraded relative to the start; R0 degrading less does not
show a benefit. This is a selection diagnostic, with no continuation held-out
score. The [sanitized receipt](../reports/runs/v2-continuation-20261019/summary.json)
(SHA-256 `cc686cf66ee3d7daa777ecc8b3c1ccaedda5d546da96107ad1f18ea0bf055cc5`)
records the exact source, config, schedule and reload evidence. Both
[cloud runs](https://wandb.ai/black-order/liquid-mcdm-lab/runs/6ohymgld)
(C1; R0 run `57fzcrlp`) finished with sanitized cloud-readback SHA-256
`f9b02842db5e1d6d0747913a3f3e0a760cdcb95a147d17f45603f7b52c3d4580`.
The private C1/R0 adapters passed forced SHA readback at Hub commit
`4b43ec8767d2bb99ec835020c04d412c89cd26a4`, retention-manifest SHA-256
`4c5dc14a52a25e74acfa9a013d4ecc3ba14dcd96d9c21c44f0b61c857bfb367d`.

## Known and anticipated failure modes

The v1 synthetic pilot raised alias/order and calibration concerns; it provides no v2 evidence. For v2, inspect long or variable candidate sets, missing or incomparable units, repeated entities across splits, unavailable options, zero-positive and multi-positive groups, position bias in clicks, domain shift, language imbalance, near-ties, and conflicting/absent preferences. Record actual failures with sanitized examples only after remote evaluation. A lack of suitable evidence should yield `preferences_unknown` or explicit scenarios, not fabricated weights or a false `verified` label.

A validation-only candidate-order diagnostic on the frozen pilot found
52/180 Swissmetro top-choice flips (28.89%) across 20 selected respondents,
with mean probability L1 change 0.39663. ESCI admitted 12/20 selected query
groups after eight both-order prompt overflows; 7/12 admitted pairs flipped
(58.33%), mean L1 change 0.31887. There were no evaluation failures after
admission. This exposes positional sensitivity in the pilot; it does not
measure accuracy loss or show that fine-tuning caused the effect, because base
d1 was not evaluated in this diagnostic. The
[sanitized receipt](../reports/runs/v2-order-20261019/aggregate.json) has
SHA-256 `1caff28407551758975ab5ebcf67abd3708f44fa8f2fa4a29ead7f04beeabad7`.

## Reproduction and rights

Public artifacts will include exact code/config/checksum references, a base-plus-adapter loading example, and commands for the permitted evaluation subset. Restricted source rows and individual predictions require source-specific authorization and remain out of the public package. The [v2 data card](V2_DATA_CARD_DRAFT.md) and per-source acquisition records govern source rights; an upstream dataset license cannot be assumed to grant rights to publish trained derivatives. The report may publish aggregate metrics only after rights and disclosure review.
