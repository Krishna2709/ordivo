# Deterministic decision verification

The implemented API is `liquid_mcdm.decision.verify_decision(spec, proposal=None,
policies=None, learned=None, scenarios=None)`, with arguments after `proposal`
passed by keyword. It produces an `ordivo.decision.v2.1` JSON-compatible trace.
No model, network call, training run, or W&B side effect occurs in this boundary.
Execute examples/tests on the approved RunPod runtime.

## Example input and result

```python
from liquid_mcdm.decision import verify_decision

spec = {
    "scope": "example/vendors",
    "criteria": [{"id": "cost", "direction": "min"},
                 {"id": "quality", "direction": "max"}],
    "alternatives": [
        {"id": "A", "values": {"cost": 100, "quality": 90}},
        {"id": "B", "values": {"cost": 80, "quality": 75}},
        {"id": "C", "values": {"cost": 120, "quality": 98}},
    ],
    "constraints": [{"criterion": "cost", "op": "<=", "value": 110}],
    "weights": {"mode": "user", "values": {"cost": 1, "quality": 0}},
    "method": "WASPAS",
    "method_params": {"lambda_": 0.5},
}
trace = verify_decision(spec, {"decision": "B", "scores": {"A": 0.8, "B": 1.0}})
assert trace["status"] == "verified"
assert trace["decision"] == "B"
assert trace["feasibility"]["C"] == "infeasible"
assert trace["checks"]["scores_verified"] is True
```

Here A and B are both Pareto-optimal among feasible options. Cost-only WASPAS
scores are exactly 80/100 and 80/80. Changing weights can legitimately change the
winner. Neither this example nor unit tests are trained-model predictions.

## Contract

- Criteria and alternatives have unique nonempty IDs; values name every criterion.
  Unknown evidence is null, never zero imputation. Booleans and nonfinite numbers
  are invalid measurements. `scope` is required.
- Constraints support numeric `<`, `<=`, `>`, `>=`, `==`, with exact comparisons.
  Unknown evidence produces `unknown`; a known violation takes precedence.
- Pareto dominance uses every declared criterion and its direction: no worse in
  all and strictly better in one. Identical options both remain on the frontier.
  Missing possibly feasible rows mark the analysis incomplete. `dominated_by`
  is evidence among complete feasible alternatives, not a claim about unknown rows.
- Exact solvers receive only feasible alternatives. Zero feasible alternatives
  and a single feasible alternative bypass formulas requiring two rows. Scores
  are not fabricated for these cases. VIKOR compromise semantics remain intact.
- Weights are resolved independently of proposed scores. Multiple profiles retain
  separate rankings. The engine reports no unique decision when profiles disagree.
- Proposal fields are limited to `decision`, `scores`, `ranking`, `pareto_frontier`.
  Model-supplied `checks`, `status` or arbitrary reasoning fields are rejected.
  Scores/rankings must cover exactly the feasible population. Tie groups retain
  identity and do not imply a unique winner.
- Every check starts null. Only an executed comparison can make it true or false.
  A proposal is `verified` only for its supplied claims; omitted claims remain
  unchecked. `analysis_status` records recommendation completeness separately.
  A verified Pareto-only claim does not mean a weighted recommendation exists.
- Score tolerance is absolute 1e-9 plus relative 1e-7; solver ties use 1e-12.
  Tolerances are fixed by code, not chosen by the proposed answer.
- `decision` always describes the engine result. A rejected proposed decision
  never overwrites it. Unknown/uncertain preferences yield no weighted winner.
- `input_sha256` binds the supplied specification, resolved weights and scenario
  requests. It is not a signature and does not bind code: the run manifest must
  separately include the code commit and environment versions.

## Sensitivity and explanations

Pass a nonempty list of explicit user weight requests as `scenarios`. Each is
recomputed using the same criteria, constraints, method and parameters. Results
include weights, scores, ties and input hashes. Winner stability counts scenarios
retaining the base unique winner, including ties as non-retention; it is null
when the base has no unique winner or evidence is incomplete. No finite scenario
set certifies an exact switching boundary. Parameter/evidence perturbation
sweeps and boundary search remain future extensions.

Explanations should render these fields and their evidence references. Do not
claim to inspect hidden reasoning. The verifier is independent of model claims,
but intentionally reuses `mcdm/solvers.py`; it is not a second independent formula
implementation. Existing `pymcdm`/vendored reference tests and hand-computed
fixtures test the numerical implementation independently.

## Tests and limits

`tests/test_decision_verification.py` exercises the contracts, malformed inputs,
constraints, missing evidence, policies, objective degeneracy, uncertainty,
score tampering, permutations, ties and explicit sensitivity. Existing solver
reference tests continue to apply. [Run evidence](../reports/V2_VERIFIER_VALIDATION.md)
records actual remote outcomes. Neural compatibility and real-world decision
quality require separate experiments; a passing verifier does not establish either.
