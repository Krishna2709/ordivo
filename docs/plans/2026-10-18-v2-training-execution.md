# V2 training execution ledger

Source plan: [user-provided training plan](Ordivo_v2_model_training.md), preserved
verbatim. Architecture and sequencing rulings: [ADR 0008](../decisions/0008-v2-verifier-and-research-order.md)
and the four-role [ADR 0009](../decisions/0009-v2-four-role-splits.md).

| Work item | State | Evidence |
|---|---|---|
| Read source plan; reconcile existing architecture and datasets | Complete | Five required research documents and ADR 0008 |
| Weight provenance, missingness and objective conventions | Implemented | `decision/weights.py`, remote contract tests |
| Constraints, Pareto and numerical claim verification | Implemented | `decision/verification.py`, exact solver reference tests |
| Explicit weight scenarios | Implemented | Recomputed scores and stability, not confidence |
| Independent review | Complete, important issue fixed | Unknown feasibility regression reproduced then corrected |
| Real multi-domain data adapters/grouped baselines | Source implementation in progress; no real held-out benchmark | Data contract/reviewer-fix remote tests passed 19/0 at `2f417cb`; official Swissmetro acquisition/build receipt exists. Expedia official file list succeeded, download returned HTTP 403. ESCI and Swissmetro baseline results pending. |
| Differentiable d1 readout and training compatibility | Bounded remote check passed | Exact integrated commit `22707ed3`; one fixed A40 BF16 forward/backward, finite loss/gradients and reload parity. No optimizer step, v2 adapter, or behavioral training result. |
| V2 SFT and direct calibration baselines | Pending | Experiment S0/C0/C1 |
| RLCD/GRPO experiments and DPO pairs | Pending | Experiment R0/G0/G1/P0 |
| Final model, held-out evaluation and release | Pending | No v2 weights published |

## Verification chronology

- Baseline source before new feature: 130 tests passed on RunPod.
- New contract tests initially failed because `liquid_mcdm.decision` did not exist.
- First implementation: 154 tests passed on RunPod.
- Missing-feasibility proposal regression: one reproduced failure; fixed by
  preserving unknown as null, not false. Expanded suite: 155 passed.
- W&B verification record and exact final commit/results live in
  [V2_VERIFIER_VALIDATION.md](../../reports/V2_VERIFIER_VALIDATION.md).
- New RunPod A40 control replay at base `ddab9c4` exited 0 with zero skips after
  a pinned-tokenizer setup; this verifies the base suite, not integration of
  the v2 worker branches. Receipt: `v2-swarm-20261019T030009Z/baseline155_zero_skip`.
- V2 behavioral data reviewer-fix tests at `2f417cbe28e77e7303d3c7fbfdf3d3d80bcf720f`
  passed 19/0 on the remote pod. Release-report source tests at `178d1fd4`
  passed 7/0. These are component tests, not real-data performance evidence.
- Official Swissmetro source/build at `b352c14b21d4297194258c1cb43a54971bde3513`
  exited 0. Source SHA-256 is
  `27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2`;
  raw 10,728 survey rows yielded 10,719 retained choice groups after nine
  no-choice exclusions. Four partitions contain 8,532 train, 549 selection,
  594 calibration, and 1,044 final-test groups. Source-specific redistribution
  rights and any neural held-out model result remain unresolved. Receipt:
  `v2-swarm-20261019T030009Z/swiss_source`.
- First Swissmetro classical held-out benchmark at exact evaluator commit
  `9c532d98050e40ab3c9056e0bec1eeace123ccef` scored all 1,044 expected
  stated-choice groups, zero failures/exclusions. Conditional logit had 652/1,044
  top-1 agreement; travel-time heuristic 665/1,044, cost heuristic 294/1,044,
  seeded random 375/1,044. This is a single-source, single-run baseline, not a
  neural v2 or transfer result. No paired interval or billed cost is recorded.
  Sanitized [aggregate](../../reports/runs/v2-swissmetro-20261019/aggregate.json)
  and [remote/local readback](../../reports/runs/v2-swissmetro-20261019/readback.json)
  preserve the result. Source file SHA, split-manifest SHA and denominators were
  checked against the persisted remote task receipt.
- Official Expedia `DownloadDataFile` returned HTTP 403 after file-list access;
  no training rows were acquired and the cause is not established. Receipt:
  `v2-swarm-20261019T030009Z/expedia_diagnostic`.
- Official Amazon ESCI Task 1 acquisition/build at `b352c14` exited 0: 2,621,288
  example rows, 1,503,277 outside the reduced ranking pool omitted,
  1,118,011 candidate rows across 48,300 groups. Four partitions contain
  30,372 train, 1,688 selection, 1,744 calibration, and 14,496 official-test
  groups. This is a source-defined judged candidate pool, not retrieval recall.
  Subsequent source-bound audit confirmed zero query-and-locale overlap and
  official test isolation, with all four split hashes identical old/new.
  Receipt: `v2-swarm-20261019T030009Z/esci_source`.
- ESCI classical benchmark at exact evaluator `22707ed3` scored all 14,496
  expected official-test query groups, zero failures and zero zero-ideal
  exclusions, under frozen old manifest SHA
  `012fa5f58aaa6b75b0d73c1a9f84f811713e87c61df1dc5137867d34c2cba540`.
  Macro NDCG@10: LambdaRank 0.6596, lexical overlap 0.6497, seeded random
  0.5338. Any-tie groups: 14,495, 14,496 and 0 respectively; the tie rule is
  ascending canonical candidate ID. Any-tie counts do not establish whether
  top-ten scores change under tie averaging; no superiority or uncertainty
  claim follows. Sanitized [aggregate](../../reports/runs/v2-esci-20261019/aggregate.json)
  and [readback](../../reports/runs/v2-esci-20261019/readback.json) retain
  source/hash evidence. The later source-bound manifest's four split hashes
  were reconciled exactly against the old manifest in the
  [remote audit](../../reports/runs/v2-source-audit-20261019/audit.json).
- D1 option-readout/training compatibility at integrated `22707ed3` exited 0
  on A40 BF16: one fixed prompt, finite loss 0.57594, 166 trainable tensors
  with nonzero gradients, option-readout parity maximum delta 2.98e-8,
  unavailable-option probability 0, reload delta 0. No optimization or pilot
  training occurred. Remote/local metadata SHA-256 matches
  `2a9500911f60cf50635c31277b8994c8cebef92ab455174d12b9dea6f6317406`.
  Receipt: `v2-swarm-20261019T030009Z/gpu_compat_22707`.
- [W&B cloud run 2f8r8pwf](https://wandb.ai/black-order/liquid-mcdm-lab/runs/2f8r8pwf)
  was read back with aggregate test/data and Swissmetro baseline telemetry;
  sanitized [cloud receipt](../../reports/runs/v2-wandb-20261019/readback.json)
  has SHA-256 `b5074afd5532054c7a1b06f080f2042cf7654c6389fb5b490876344aec244d7f`.
  It is not a trained-model artifact. A Hugging Face v2 research repo exists
  privately; no public v2 model has been created.

No local workload was executed. The v2 worker branches have not yet passed a
single integrated remote suite. Initial GitHub CPU runs were superseded by the
user's explicit RunPod instruction; GPU execution guards were never bypassed.

## Deferred review improvements

The independent reviewer identified two minor follow-ups: dedicated tests for
proposed rankings/Pareto claims (including malformed ties and incomplete evidence),
and retaining a proposed-answer hash for distinguishing rejected proposals in
later audit logs. Existing input hashes bind the decision input, not the proposal.
These are recorded rather than claimed implemented.
