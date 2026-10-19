---
license: cc-by-4.0
pretty_name: Liquid MCDM Synthetic Decision Episodes
task_categories:
- multiple-choice
- text-ranking
tags:
- decision-making
- mcdm
- synthetic
- copras
- waspas
- vikor
- topsis
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
  - split: test
    path: test.jsonl
---

# Liquid MCDM Synthetic Decision Episodes

A deterministic synthetic benchmark for single-choice imitation of four exact
multi-criteria decision-making algorithms: COPRAS, WASPAS, VIKOR, and TOPSIS.

## Contents

Each JSONL row is a complete decision episode containing the algorithm and its
parameters, weighted benefit/cost criteria, candidate alternatives with
full-precision numerical values, exact scores, ranking/tie groups, and a winner
only when the method's policy admits a unique choice. `manifest.json` records
split hashes, source hashes, seed lineage, generator settings, and counts.

Splits are assigned by scenario family before sampling. The generator crosses
method, scenario family, and weighting scheme instead of coupling them. The
published release contains 1,000 episodes generated with root seed 42.

## Intended use

- Evaluate whether a model can imitate an explicitly identified MCDM method.
- Test robustness across criteria counts, alternative counts, directions,
  scales, weighting schemes, near ties, and held-out scenario families.
- Reproduce solver, prompt, and model experiments from
  https://github.com/Krishna2709/liquid-mcdm-lab.

## Limitations

This is synthetic algorithmic supervision. It is not human preference data,
expert judgment, hotel click/booking behavior, or evidence of operational
usefulness. Winners are conditional on the complete candidate set, weights,
normalization, method, and tie policy. Some MCDM methods can exhibit rank
reversal. VIKOR nonunique compromise cases are not silently converted into
single-winner labels.

## License and provenance

Dataset records and manifests are released under CC BY 4.0. Generator and
solver code are versioned in the linked GitHub repository. Exact mathematical
policies and independent cross-checks are documented there.
