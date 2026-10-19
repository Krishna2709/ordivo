# ADR 0003: split scenario families before generation and keep telemetry offline

Status: accepted. Date: 2026-10-18.

Stage zero uses synthetic solver labels only. Generate 80/10/10 episodes with fixed, disjoint distribution families by split before sampling; seed lineage is [root, split index, episode index]. No augmentation in v1. Future permutations inherit parent split. Factorial cycling crosses methods, families and weighting schemes independently; review corrected an earlier modulo coupling. This intentionally tests distribution shift; it does not constitute a within-distribution validation estimate or an alternatives/criteria-count extrapolation experiment.

Keep unrounded JSON float values and use those same round-trip representations in prompts; retain raw solver scores. Persist SHA-256 per split and a deterministic manifest. Refuse overwriting dataset directories. Exclude ties and nonunique VIKOR compromises from single-choice training while retaining them for analysis; report denominator and exclusions.

W&B is the experiment ledger, offline by default. API keys are not needed for offline smoke and `.env` is not automatically loaded. Explicit safe config fields; disable console/code/git/system metadata capture; only constructed sanitized manifest/evaluation summary artifacts; detailed slices remain in the Colab report. Online logging is an explicit CLI mode; user supplies entity. No default raw dataset upload, no notebook capture, and no cloud run URL for offline runs.

Expedia's terms and external-data restriction leave the proposed synthetic-transfer use unresolved. MSLR also requires an agreement. Neither is a dependency of the working CPU pipeline. UCI classification is a future separate task, not a hotel-choice ground truth.
