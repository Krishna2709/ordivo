# ADR 0002: explicit mathematical conventions and independent oracles

Status: accepted. Date: 2026-10-18.

Keep pure NumPy implementations with explicit normalization and tie groups. Use pymcdm 1.3.1 for WASPAS, VIKOR nondegenerate scores, and TOPSIS with vector normalization. Installed-wheel inspection and a cost-dominance example reproduced a COPRAS defect: the library's cost term collapses to a direct cost sum. Its docstring examples also follow that defective expression; they are not valid mathematical fixtures.

Do not alter our formula to match the defect. COPRAS tests instead use hand calculations and the audited standalone `copras_method` from pyDecision commit `07e59ab68bc37bf571211a97ece5685e3adb227a`, distributed with its license under `tests/reference`. Tests execute that unchanged function via AST extraction, omitting only plotting imports and helpers; `graph=False`. This keeps plotting dependencies out of the solver runtime. The source and SHA are recorded in `tests/reference/provenance.json`.

Project extensions: require at least two alternatives; reject missing/nonfinite data; require positive COPRAS/WASPAS inputs; normalize nonnegative weights; constant VIKOR dimensions contribute zero without weight redistribution. Near-zero S/R spans (<=32 float64 eps after normalized weighting) contribute zero to avoid magnifying roundoff. TOPSIS no-information closeness is 0.5 for every candidate. All-benefit COPRAS is its weighted normalized benefit sum. These extensions are not independently certified by library parity in degenerate cases.

Tie tolerance is absolute 1e-12 in raw-score space. Return ranked tie groups and no winner for first-place ties. VIKOR retains Q minimization and returns a unique winner only when acceptable advantage and stability both hold. Record the compromise set separately. Do not label every Q argmin as a unique compromise.

Adding candidates may change rankings. No blanket rank-reversal or translation invariance assumption. See [mathematics](../mathematics.md) and [data/method evidence](../data_colab_research.md).
