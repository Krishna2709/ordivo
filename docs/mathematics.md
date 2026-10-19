# Exact teacher conventions

“Exact solver” means deterministic float64 evaluation of specified formulas, not symbolic or arbitrary-precision arithmetic. See [primary references and independent implementations](data_colab_research.md). Rows are alternatives; columns are criteria; directions are min/max; finite nonnegative weights are normalized to sum one.

- **COPRAS:** column-sum normalize and weight, then sum benefit components P and cost components C. Q=P+sum(C)/(C*sum(1/C)), larger better. Return raw Q, not Q/max(Q). With no active costs use P.
- **WASPAS:** benefit x/max(x), cost min(x)/x. Q=lambda*weighted_sum+(1-lambda)*weighted_product. Evaluate product in log space; larger better. Lambda defaults to 0.5, bounded [0,1].
- **TOPSIS:** Euclidean column normalization, weighting, direction-aware ideal/anti-ideal, then distance_to_antiideal/(distance_to_ideal+distance_to_antiideal). Larger better. Zero columns contribute no distance; all-identical candidates receive 0.5.
- **VIKOR:** direction-aware range-normalized regret; S is weighted total regret and R maximum weighted regret. Q=v*range_normalize(S)+(1-v)*range_normalize(R), smaller better. v defaults to 0.5, bounded [0,1]. A constant criterion contributes zero. S/R spans within 32 machine eps are treated as numerical degeneracies with zero contribution.

VIKOR acceptable advantage is Q(second)-Q(first)>=1/(N-1), within tie tolerance. Stability means the leading candidate is also best (allowing ties) in S or R. If advantage fails, the compromise set contains candidates whose Q is less than the leader plus 1/(N-1); if only stability fails, include first and second (including boundary ties). Only both conditions and a unique leading score allow `winner`. Ranking and compromise set are different outputs.

All methods require >=2 alternatives and >=1 criterion. Reject missing/nonfinite values and invalid weights/directions. COPRAS/WASPAS strictly require positive measurements, including zero-weight dimensions; no epsilon repair. TOPSIS/VIKOR accept signed data. Constant criteria retain their declared weights; in WASPAS constant factors may change score scale, even when they do not supply a preference.

Stable-sort scores into tie groups anchored to each group's best score, absolute tolerance 1e-12; indices inside groups retain original input identity. No unique winner for a tied first group. `near_tie` means top-two raw-score gap <=1e-4, a method-dependent diagnostic, not a comparable cross-method probability.

Positive per-column rescaling preserves these formulas up to floating-point tolerance. VIKOR range regret also permits positive affine transforms in nondegenerate cases. TOPSIS, WASPAS and COPRAS generally are not translation invariant. Adding/removing alternatives can change normalizers/ideals and reverse rankings. Nonnegative weighted criteria generally respect dominance, but zero weights, VIKOR endpoints and tolerances permit ties; no strict-dominance assertion is made for every configuration.
