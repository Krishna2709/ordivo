# Architecture

The mathematical decision domain is independent of model training and telemetry. Public boundaries are ordinary Python functions over matrices or episode dictionaries, with thin CLI/notebook adapters.

| Module | Responsibility | Boundary |
|---|---|---|
| `mcdm/solvers.py` | Pure float64 formulas, validation, ties, VIKOR compromise | `solve(...) -> Result` |
| `data/synthetic.py` | Scenario families, independent factor schedule, seed lineage, JSONL/hash persistence | `generate_dataset`, `load_dataset`, `validate_episode` |
| `evaluation/metrics.py` | Winner probability metrics with explicit exclusions, slices and evaluated split identity | `evaluate_predictions`, `evaluate_dataset` |
| `training/rendering.py` | Whitelist source inputs; call upstream prompt/aliases/readout; enforce completion boundary | `render_example`, `option_probabilities` |
| `training/preflight.py` | Pinned small-file tokenizer cache and compatibility report | `tokenizer_preflight`; no weight downloads |
| `_vendor/d1_prompt.py` | Audited, unchanged upstream decision formatting and token-group scoring | Bundled license and source SHA |
| `tracking/wandb_run.py` | Construct sanitized telemetry/artifacts; offline default | `log_cpu_run`; no implicit `.env` loading |
| `runtime.py`, `environment.py` | Runpod/HF Jobs/Colab GPU execution guard and inventory | Never claim model compatibility from inventory |
| `cli.py`, notebooks, remote verification script | Orchestrate domain functions | All workload execution is remote |

## Invariants

Measurements and exact teacher scores retain float round-trip precision. A single-choice target is admitted only when unique under the method's policy. VIKOR compromise conditions cannot be replaced by argmin alone. Ordered tie groups keep candidate identity; no artificial single winner from stable sort.

Families are assigned to splits before sampling. Method, family and weighting factors are crossed rather than coupled. Future augmentation inherits parent split. Manifests carry generator/solver source hashes, version, seed lineage and split hashes; validation recomputes labels.

Prompt construction admits only method/parameters/criteria/alternatives and explicit normalization. Solver targets never enter state. Full-vocabulary completion CE and restricted-option inference are different operations. A GPU trainer must verify actual collator masks, trainable module names and save/reload behavior before acceptance.

Metrics require complete valid probability distributions and missing predictions fail. Report denominators and excluded nonunique cases. Teacher self-consistency is a plumbing control, not measured learned performance. Behavioral labels need a separate provenance contract and grouped split policy before inclusion.

## Extension points

Add a method by declaring its formula/domain/order/compromise contract and independent fixtures, then extend the generator/schema. Add behavioral datasets through a separate grouped loader; do not impersonate `synthetic_solver`. Add a model through the decision-input/probability boundary. Encoder set-ranking and full-ranking metrics are later distinct tasks, not implied by winner classification.

The user requires remote resources only and selected Hugging Face Jobs after Colab allocation failures; see ADR0005. Pure functions remain reusable, but CLI/notebook entry points reject local workload execution. Source preparation stays local; generated datasets, caches, environments, runs and checkpoints stay remote. Colab is ephemeral: persistent artifacts and resumable checkpoints require an explicit remote destination.

Provider orchestration is outside the domain. `runpod_control.py` packages an
allowlisted snapshot and controls one pod plus its network volume;
`runpod_bootstrap.py` creates isolated remote environments; `verify_remote.py`
coordinates CPU acceptance; and `train_d1_overfit.py` owns the model-specific
compatibility and adapter lifecycle. Paths and provider identities never enter
solver or dataset logic. Runpod model caches, environments, data, tests and
training stay under the remote `/workspace` volume.
