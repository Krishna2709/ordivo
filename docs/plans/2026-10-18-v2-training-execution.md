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
| Real multi-domain data adapters/grouped baselines | Two official grouped sources and classical held-out benchmarks complete; Expedia grouped build pending | Swissmetro and ESCI source-bound four-role manifests audited; source-specific classical metrics and denominators below. Expedia official archive and `train.csv` acquired after user-confirmed rules. |
| Differentiable d1 readout and training compatibility | Bounded remote check passed | Exact integrated commit `22707ed3`; one fixed A40 BF16 forward/backward, finite loss/gradients and reload parity. No optimizer step in this compatibility check. |
| V2 SFT and direct calibration baselines | Preliminary two-domain SFT pilot completed; calibration and matched controls pending | Pilot source `854de087`, 40 steps, private best adapter and selected-validation receipt below; S0/C0/C1 held-out conclusions pending. |
| RLCD/GRPO experiments and DPO pairs | Pending | Experiment R0/G0/G1/P0 |
| Final model, held-out evaluation and release | Exploratory paired probe measured; protected release evidence pending | Preliminary weights retained privately. Adapted pilot scored below base on Swissmetro accuracy and ESCI NDCG@10 on admitted held-out groups; no public v2 model or release claim. |

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
- Integrated research source suite at exact `22707ed34fd125e96aade9123f9baaf34cd64a18`
  passed 221 tests with zero skips on RunPod A40; source archive SHA-256
  `4ec592756c7346dbbfd1df3ab2ecb4c77e4e3072975f2b0ceb080da0e3ad7a2c`
  matched remotely. Receipt: `v2-swarm-20261019T030009Z/final_suite_22707`.
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
  no training rows were acquired at that point and the cause was not established. Receipt:
  `v2-swarm-20261019T030009Z/expedia_diagnostic`.
- After the user confirmed applicable rules, an official Expedia archive was
  acquired: 434,573,097 bytes, SHA-256
  `f16b9622a73c015e2693284a5939a82817417f23f284bfd69081f1f35cdd0602`.
  Python `zipfile` could not extract its Deflate64 CSV members. A versioned
  7-Zip 23.01 extraction succeeded: official `train.csv` is 2,358,114,793
  bytes, SHA-256 `84a53139ea92df1e9400f84852facbd78f3136f366a08440f4ab4518e71a49a9`.
  No grouped Expedia manifest or benchmark exists yet.
  [Archive inventory](../../reports/runs/v2-expedia-20261019/inventory.json)
  and [source receipt](../../reports/runs/v2-expedia-20261019/acquisition.json)
  contain metadata only.
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
  training occurred **in this compatibility task**. Remote/local metadata SHA-256 matches
  `2a9500911f60cf50635c31277b8994c8cebef92ab455174d12b9dea6f6317406`.
  Receipt: `v2-swarm-20261019T030009Z/gpu_compat_22707`.
- [W&B cloud run 2f8r8pwf](https://wandb.ai/black-order/liquid-mcdm-lab/runs/2f8r8pwf)
  was read back with aggregate test/data and Swissmetro baseline telemetry;
  sanitized [cloud receipt](../../reports/runs/v2-wandb-20261019/readback.json)
  has SHA-256 `b5074afd5532054c7a1b06f080f2042cf7654c6389fb5b490876344aec244d7f`.
  It is not a trained-model artifact. A private Hugging Face v2 research repo
  contains 22 readback-verified allowlisted classical artifacts at private repo
  commit `1a64f477f66413b25adad68b297c2cdcd4f5ba7e`.
- A subsequent bounded Swissmetro/ESCI supervised pilot at executed source
  `854de0877c8e7e380b7353a578e95ba9865d4e25`, config SHA-256
  `c152e64d839082531d4ece9e66d330fa35ffddfbc442685742b9fd171de5462b`,
  completed 40 optimizer steps and selected checkpoint step 40. It recorded
  nonzero LoRA updates, 20 updates per source and zero reload-probe loss delta.
  Swissmetro selected-validation loss changed 1.1358→0.8577; ESCI
  3.1603→2.6311. ESCI prompt overflow excluded 325 of 1,000 selected train
  and 29 of 100 selected validation groups. These are selection diagnostics,
  not held-out model performance. Sanitized remote/local receipt SHA-256 is
  `5f58e02b3259601d9663892f50e32460a57be2d424724ec283d87ecb8678f7c1`.
  The best adapter was retained in **private** `blackOrder/Ordivo-v2-research`
  at commit `bf1d425b4e6ffc392121d8fb0776e41b1af48c4a`, folder
  `supervised-pilot/20261019T030009Z/pilot-854de/`; nine allowlisted files
  plus manifest passed forced SHA readback. W&B pilot
  [run bnsgv73p](https://wandb.ai/black-order/liquid-mcdm-lab/runs/bnsgv73p)
  finished at step 40. Paired held-out neural evaluation, calibration, public
  weight rights and release were pending at this point.
- Exploratory paired held-out probe at the same executed source commit
  `854de0877c8e7e380b7353a578e95ba9865d4e25` compared base d1 with the
  frozen best pilot adapter. Swissmetro: 450 selected and scored scenarios from
  50 held-out respondents, base accuracy 306/450=0.68 versus adapted
  297/450=0.66; base/adapted NLL 0.805878/0.758958 and Brier
  0.459254/0.462190. ESCI: 50 selected held-out groups, 23 complete-group
  prompt overflows, 27 scored by each model; base/adapted NDCG@10
  0.778290/0.752276. No scoring failures after admission. The [sanitized
  aggregate](../../reports/runs/v2-neural-heldout-20261019/aggregate.json)
  has SHA-256 `a735c1db8810f9d9955152bbe16651e690c921e7330596cedc117d2358a19c8a`,
  evaluation config SHA-256 `524bfe7cde8135132417ed029a8fb2738bd5ffd068f9190d081c6246aaaadc06`,
  and source-manifest hashes. The [W&B probe run](https://wandb.ai/black-order/liquid-mcdm-lab/runs/3zdzt6ik)
  was read back as sanitized aggregate telemetry; eight probe artifacts are
  retained in private Hub folder `heldout-probe/20261019T030009Z/paired-854de/`.
  This is a bounded exploratory probe, not the protected final release test;
  intervals, multi-seed, calibration and entire-held-out-domain transfer remain
  pending.

No local workload was executed. The integrated suite proves source behavior,
not real-data neural efficacy or release readiness. Initial GitHub CPU runs were superseded by the
user's explicit RunPod instruction; GPU execution guards were never bypassed.

## Deferred review improvements

The independent reviewer identified two minor follow-ups: dedicated tests for
proposed rankings/Pareto claims (including malformed ties and incomplete evidence),
and retaining a proposed-answer hash for distinguishing rejected proposals in
later audit logs. Existing input hashes bind the decision input, not the proposal.
These are recorded rather than claimed implemented.
