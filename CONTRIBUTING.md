# Contributing

Open an issue before changing mathematical policy, prompt semantics, dataset
schema, or public model behavior. Solver changes require a hand fixture, an
independent reference or published example, and invariance/degeneracy tests.

This project keeps workloads off contributor laptops. Source editing and
packaging may run locally; dataset generation, the complete suite, model
weights, and training run on the documented remote environments. Never commit
credentials, `.env`, generated weights, raw restricted datasets, or private
W&B exports.

A pull request should state the affected invariant, commands executed, remote
runtime identity, test counts, and any changed artifact hashes. Learned-model
metrics must name the evaluated split and denominator.
