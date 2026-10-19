# V2 verifier: remote validation evidence

Executed 2026-10-18 America/Chicago (receipts use UTC). This is numerical
verification, **not a v2 model-training run**.

- Tested source commit: `3e981b66c5f478e5a3fd2d531930af418488ba03`.
- Provider: RunPod, NVIDIA A40 48 GB class; actual driver inventory in the receipt.
- Existing baseline: **130 passed**, no skips.
- New module RED check: missing `liquid_mcdm.decision`, as expected before implementation.
- First implementation: **154 passed**, no skips.
- Review regression: missing constraint evidence was incorrectly reported false;
  a targeted test reproduced the error and the fix preserved null/unverifiable.
- Final suite: **155 passed**, zero failures/errors/skips, pytest wall time **10.41 s**.
- `pip check`: no broken requirements. The pinned tokenizer was tested on 36
  eligible synthetic examples; no model weights were loaded.
- W&B [run scg4z0ud](https://wandb.ai/black-order/liquid-mcdm-lab/runs/scg4z0ud)
  is finished. Artifact `black-order/liquid-mcdm-lab/ordivo-v2-verifier-evidence:v0`
  was independently read back as `COMMITTED` through the W&B API.

## Resources and persistence

Pod `r722brg4d1z8tr` ran for approximately 10.7 minutes
at $0.59/hour. Estimated compute charge: **$0.105**,
not a reconciled provider invoice. No network volume was created. Pod deletion
was verified through the provider API after evidence was saved. W&B records the
elapsed cost estimate at verification; the final lifetime estimate is in the
[runtime receipt](runs/v2-verifier-20261018/runtime-receipt.json).

[Verification receipt](runs/v2-verifier-20261018/verification.json),
[pytest output](runs/v2-verifier-20261018/pytest.log),
[JUnit results](runs/v2-verifier-20261018/pytest.xml),
[environment freeze](runs/v2-verifier-20261018/pip-freeze.txt), and
[file hashes](runs/v2-verifier-20261018/manifest.json) preserve the reproducibility
record. RED logs and the baseline XML are in the same directory. No secrets,
model weights or raw external datasets are included.

## Scope of the result

The suite covers deterministic solver references, data/rendering/control-plane
regressions and the new decision boundary. A fresh source-only reviewer found
the tri-state issue above; its fix passed the full remote suite. Minor follow-ups
are recorded in the [execution ledger](../docs/plans/2026-10-18-v2-training-execution.md).

The initial GitHub CPU attempt failed three CLI tests because those entrypoints
require a remote GPU runtime, while 124 tests passed and 3 tokenizer tests skipped.
That was not training or solver failure. The user directed testing to RunPod;
the temporary workflow was removed and the complete suite passed there. SSH did
not inherit the provider's pod identifier, so the control plane passed the actual
API-confirmed ID into the remote process; hardware checks remained active.

No real-data baseline, learned preferences, neural compatibility result, RL reward
optimization, v2 model accuracy or v2 model release is established by this run.
