# Ordivo v2 experiment matrix

Planning date: 2026-10-18. Preserve `blackOrder/Ordivo-v1` and its historical
results. No row below implies a completed new neural experiment.

| ID | Experiment | Data / target | Main comparison | Status |
|---|---|---|---|---|
| V0 | Weight policy + numerical verifier | Hand fixtures, adversarial inputs, independent formula references | Mathematical invariants | Implemented; see remote validation report |
| B0 | Exact MCDM | Synthetic and explicit-preference numeric specifications | Formula reference; not behavioral oracle | v1 available; new wrapper implemented |
| B1 | LightGBM / choice baselines | Source-specific real groups | Domain-specific choice/ranking performance | Planned |
| S0 | d1 supervised choice/ranking | Shared mixed-domain model, task-typed labels | B1, v1 transfer, zero-shot backbone | Planned |
| C0 | S0 + held-out temperature fit | Event probabilities | S0 raw probability quality | Planned |
| C1 | Direct NLL/Brier fine-tuning | Same S0 checkpoint and splits | C0 and matched-budget SFT | Planned |
| R0 | RLCD-inspired report policy | Perturbed probability reports | C1; score/exploration ablations | Not compatibility-validated |
| G0 | Structured-action SFT | Verifiable action examples | Action-interface control for G1 | Planned |
| G1 | Verifier-guided GRPO | Same action interface and SFT start as G0 | G0, R0 on applicable metrics | Not compatibility-validated |
| P0 | DPO | Genuine expert/observed preference pairs with provenance | Pairwise supervised control | Pending suitable pairs |
| X0 | Held-out-domain transfer | Entire unseen domains | Domain specialists and shared S0 | Planned |

## Dataset tasks and splits

- Expedia: search-group split, booking/click semantics separate; analyze randomized
  presentation subset and position bias. Exactly-one-booking conditional choice
  is distinct from purchase probability. Kaggle authentication remains needed.
- Swissmetro: stated mode choice, respondent-disjoint splits and availability.
- Amazon ESCI: graded relevance; preserve official test and query/locale groups.
- MSLR: fixed official fold and query groups; source agreement recorded.
- SUSHI: explicit rankings, respondent-disjoint; optional preference control.
- Synthetic MCDM: mathematical and metamorphic control, not real behavior evidence.

Pin hashes and source revisions at acquisition. A shared representation must
actually train across domains; collecting many datasets without a held-out-domain
test does not establish generality. Retain candidate identity across permutations.
No truncation may silently remove an alternative or its mandatory evidence.

## Evaluation contract

| Dimension | Metrics / checks |
|---|---|
| Mathematics | Score error, ranking/tie agreement, VIKOR compromise agreement |
| Constraints | Proposed violation rate and served violation rate separately; unknown counts |
| Ranking | Top-1, NDCG@k, MRR where applicable, tie-aware pairs; explicit denominators |
| Utility | Declared utility regret, feasibility-conditioned coverage; no fabricated counterfactuals |
| Probability | NLL, multiclass Brier, fixed-bin ECE, reliability by domain/option count |
| Robustness | Candidate/criterion permutations, positive unit changes under declared normalization, missingness |
| Preferences | Changed-weight responsiveness, sampled-profile agreement, explicit scenario stability |
| Generalization | Group/time/entity shift and entirely held-out domain |
| Context | Candidate-count and token-length slices, truncation/overflow failures, latency/memory |
| Cost | GPU hours, wall time, tokens, provider rate, estimated and actual billed cost separately |

Use train/model-selection validation/calibration/final-test partitions. If a
source only supplies train/test, derive selection and calibration groups from
train; preserve test. Freeze metrics and selection rule before accessing test.
A model is not selected by ECE alone: ranking quality, constraints, proper scores,
coverage and cost are joint evidence. Report paired group-level bootstrap
intervals and multiple seeds for final comparisons; tiny compatibility runs
establish execution, not performance superiority.

## Incremental remote execution

Run source/test checks directly on the approved RunPod A40. No local installs,
datasets, models, test workloads or fine-tuning. Run a small forward/backward and
reload check once a concrete trainer exists, then a bounded pilot, then compare
held-out results before scale. Tests are operational checks, not repeated user
approval gates. Reuse a healthy runtime where practical and terminate paid
resources after artifacts are persisted.

First implementation slice: weight policy, constraints, Pareto, exact scoring,
claim verification, explicit weight sensitivity, required documentation. Next:
real-data adapters and grouped baselines, differentiable d1 option readout,
SFT and C0/C1, then controlled R0/G0/G1. Preference inference training, arbitrary
constraint languages, evidence perturbation sweeps and a production serving API
are not part of the current foundation.
