# Synthetic MCDM episodes v1

Source: generated numbers plus this repository's reference solvers. No external hotel, healthcare, personal or proprietary dataset. Labels are mathematical teacher outputs, not human preferences. No external dataset license is inherited; project distribution/license is not yet selected.

Configuration: `configs/data.yaml`; first smoke has 1,000 episodes, seed 42, float64 values, 3–8 alternatives, 2–6 criteria. The factor schedule crosses methods COPRAS/WASPAS/VIKOR/TOPSIS with every assigned family and equal, Dirichlet(1), sparse Dirichlet(0.3) weights; directions are sampled independently. Per-column units scale by 10**Uniform(-2,3).

| Split | Count | Families |
|---|---:|---|
| train | 800 | IID uniform, correlated lognormal quality, independent lognormal tradeoffs |
| val | 100 | clustered attributes |
| test | 100 | anticorrelated attributes, near-equal alternatives |

The generator source gives exact parameters, recorded by generator version; each episode records family, scale vector, weighting scheme and seed lineage. Family membership is assigned before sampling. No permutations or other augmentation in v1; future augmentations must retain parent split. Sizes overlap across splits: size extrapolation is a future version, not demonstrated by this smoke.

JSONL preserves float round-trip values; prompts use the same precision. `target.ranking` is a list of ordered tie groups of alternative IDs, `raw_scores` maps IDs to scores, and `winner` is nullable. VIKOR records S/R, advantage/stability and compromise indices. Ties/nonunique compromises remain in the dataset but are excluded from single-choice training with counts reported. A separate behavioral schema is required before real-data work; `source=synthetic_solver` cannot be reused for clicks/bookings.

Manifest: generator version, generator/solver source hashes, NumPy version, root seed, family split policy, counts and SHA-256 of each JSONL. `load_dataset` verifies hash/count and recomputes labels. Reproduction must use the locked environment and same generator source; a seed alone does not freeze algorithms or dependencies. No trained model or statistical generalization claim follows from teacher self-consistency.
