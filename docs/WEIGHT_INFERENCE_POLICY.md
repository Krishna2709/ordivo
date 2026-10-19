# Weight inference and provenance policy

Implemented boundary: `liquid_mcdm.decision.resolve_weights`. A matrix alone
cannot identify a user's priorities. Every result carries a source, status and
zero or more normalized profiles. Unknown preferences produce no invented vector.

| Request mode | Authority / input | Behavior |
|---|---|---|
| `user` | Request `values` keyed by every criterion | Validate finite nonnegative weights, require positive total, normalize |
| `approved_policy` | `policy_id`, `version`; trusted application registry | Require exact scope and version match; missing policy means unknown |
| `learned` | Trusted application `learned` argument | Require scope, evidence reference, profiles and probability mass summing to one |
| `objective` | Explicit `entropy` or `critic` request | Calculate named convention; never call it a human preference |
| `unknown` or null | No evidence | Return `preferences_unknown` |

The registry is passed separately from a model proposal. An `approved` boolean
in generated JSON cannot grant policy authority. Authentication/registry storage
is the calling application's responsibility; this package does not implement
identity management. Scope checks prevent accidental cross-scope application,
not a malicious caller from forging a registry.

Learned profiles retain each vector and its mass. The resolver consumes an
external estimate; it does not fit a preference model. An evidence URI is a
provenance pointer, not validation that a model is calibrated or that history
uniquely identifies weights. Contextual utility may later be preferable to
linear weights when compensatory assumptions fail. Different profile winners
produce `preference_uncertainty`; agreeing sampled profiles establish agreement
only within the supplied support, not all possible preferences.

## Objective conventions

Both methods use **all supplied alternatives**, including infeasible rows.
Filtering first changes the empirical information content; request a different
input population explicitly rather than silently switching populations.
The source trace names this population and the exact convention.

`entropy_raw_nonnegative_v1`: require complete, nonnegative data and at least
two alternatives. For each nonconstant column, divide by its sum to obtain
p; compute e = -sum(p log p)/log(n), with 0 log 0 = 0; normalize 1-e.
Constant columns contribute zero information. This convention does not reverse
cost columns. Negative values are rejected rather than silently transformed.

`critic_benefit_minmax_population_v1`: rescale each nonconstant column to [0,1],
reverse cost columns, compute population standard deviation and Pearson
correlation, then information j = std(j) * sum_k(1-correlation(j,k)). Constant
columns are excluded from correlation and assigned zero weight. Fewer than two
nonconstant columns, or total information at most 1e-12, is unresolved. Both
methods return unknown on degenerate information; neither substitutes uniform
weights. Scaling before sums/ranges protects finite extreme source units.

These formulas follow the [pymcdm authors' objective-weight documentation](https://pymcdm.readthedocs.io/en/master/modules/objective_weights.html).
The CRITIC primary paper is [Diakoulaki et al. (1995)](https://doi.org/10.1016/0305-0548(94)00059-H).
Our constant-column and degeneracy policies are explicit engineering choices.

## Missing preferences and sensitivity

With multiple feasible alternatives, unknown weights preserve the Pareto analysis
but do not select a weighted winner. Exactly one feasible alternative can be
identified without weights; this does not resolve the user's preferences.
A potentially feasible alternative with missing constraint evidence blocks a
unique recommendation. Explicit scenario vectors can explore trade-offs without
becoming an authoritative preference profile.

The sensitivity frequency is unweighted over the supplied scenarios. It is not
learned-profile posterior mass, a confidence interval, a calibrated correctness
probability, or a complete search of the weight simplex.
