# Liquid MCDM Lab

Research prototype for exact COPRAS/WASPAS/VIKOR/TOPSIS teachers and Liquid d1 winner prediction. **All workloads run remotely.** This local workspace is for source and documentation editing only; do not install packages, generate data, run tests or download models locally.

The release run `20261019T005111Z` trained a d1-3B LoRA adapter for 400 steps on a remote A40 after a 130-test CPU acceptance suite. Across all 87 eligible held-out episodes, accuracy improved from 28.74% to 58.62%; on the original 16-episode comparison slice it improved from 37.5% to 75.0%. The published adapter, dataset, source, complete evaluation evidence, and W&B telemetry are linked below. [de01.ipynb](de01.ipynb) remains the Colab alternative, and the [Hugging Face Jobs runbook](docs/huggingface_jobs.md) remains available.

Public release destinations:

- Source and experiment report: https://github.com/Krishna2709/liquid-mcdm-lab
- Synthetic dataset: https://huggingface.co/datasets/blackOrder/liquid-mcdm-synthetic
- LoRA adapter: https://huggingface.co/blackOrder/d1-3b-mcdm-lora
- Release experiment: https://wandb.ai/black-order/liquid-mcdm-lab/runs/4vblgw80

## What is implemented

- Pure NumPy teachers with explicit domains, normalization, tie groups and VIKOR compromise conditions.
- Synthetic JSONL generator, disjoint scenario families, seed lineage, source/data hashes, full-precision labels, and revalidation.
- Hand cases, independent cross-checks, invariance/degeneracy/rank-reversal tests.
- Single-choice metrics and CPU sanity controls; W&B offline telemetry with sanitized artifacts.
- Pinned upstream d1 prompt/readout helper with measured tokenizer, inference, training and adapter-reload checks.
- Four thin notebooks plus the Colab entry notebook, and a remote-only Runpod training entry point with bounded LoRA training and adapter persistence.

Remote CPU acceptance passed 130 tests before the release run. The 512-example training subset improved from 32.42% to 93.55% accuracy. The disjoint-family test result improved from 28.74% to 58.62% over 87 unique-winner episodes. These synthetic results measure exact-teacher imitation and do not establish real-world decision quality. See [progress](reports/progress.md) for hashes, metrics, cost, and reload evidence.

## Transfer source to your connected Colab runtime

Local VS Code files do not automatically exist in the remote kernel. Upload the prepared `dist/colab-source.zip` using the official extension's runtime file workflow, placing it at `/content/colab-source.zip`. The archive contains source/configs/tests/notebooks/docs only; it excludes `.env`, datasets, weights, environments and caches. The entry notebook has a remote-only bootstrap cell to extract and install it under `/content/liquid-mcdm-snapshots/<ZIP-SHA256>/decision-engines`; new uploads get a separate source directory so stale code is not silently reused. No Git remote is configured in this workspace.

If you later publish a repository, substitute a normal clone and exact commit checkout in the Colab runtime:

```bash
# Colab runtime terminal only; replace the remote and revision with actual values.
git clone YOUR_REPOSITORY_URL /content/decision-engines
cd /content/decision-engines
git checkout YOUR_COMMIT
```

Do not assume editing the local file updates the uploaded source. Re-upload a newly prepared source snapshot when code changes; the notebook selects its new hash-named directory. Preserve remote outputs/checkpoints separately before replacing source.

## Remote setup and verification

If the first cell fails with `FileNotFoundError: nvidia-smi`, it failed during the hardware check, **before dependency installation**. The kernel could be a CPU/TPU session or lack NVIDIA tooling; the log alone does not establish which. In VS Code use **Select Kernel → Colab → New Colab Server** to choose an available NVIDIA GPU machine, then attach this notebook to it. Auto Connect selects a default server; it is not a GPU guarantee. In browser Colab use **Runtime → Change runtime type → Hardware accelerator** and choose an available NVIDIA GPU. If using nbconvert, run it in that same GPU runtime. See [Google's extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) and [Colab FAQ](https://research.google.com/colaboratory/faq.html).

The updated first cell checks common NVIDIA driver paths and reports a clear runtime-selection error. It cannot allocate a GPU; installing a Python package or suppressing the check would not supply one. Upload both the updated entry notebook and the source ZIP before retrying. Frozen-module debugger warnings are separate from this failure.

The notebook checks managed-Colab environment markers and `nvidia-smi` before workloads. Select an NVIDIA GPU kernel. CPU mathematics still runs on the remote VM's CPU; model work later uses its GPU. This satisfies the local-resource constraint without pretending exact solvers need CUDA.

The entry notebook calls `scripts/verify_colab.py` to run these checks, plus a second generation for deterministic manifest comparison. For manual execution, run these **only in Colab** after extracting the source:

```bash
cd /content/decision-engines
python -m pip install -r requirements-cpu.lock
python -m pip install --no-deps -e .
python -m pip check
python -m liquid_mcdm.cli environment
python -m liquid_mcdm.cli generate --output outputs/smoke-v1 --count 1000 --seed 42
python -m liquid_mcdm.cli tokenizer-check --data outputs/smoke-v1 --output outputs/tokenizer_preflight.json
python -m pytest
python -m liquid_mcdm.cli baseline --data outputs/smoke-v1 --split test --output outputs/baseline.json --wandb offline
```

Restart the notebook kernel if package installation requires it. `requirements-cpu.lock` pins the CPU/tooling stack; it is not a Torch/CUDA training lock. Colab's preinstalled package conflicts must be resolved and recorded there, never assumed away. The generator refuses overwrite; use a new output directory for changed seed/config/source. `tokenizer-check` fetches only pinned tokenizer/config/template files, not weights. Run it before pytest so real-tokenizer tests do not skip.

For determinism, generate again to `outputs/repeat-v1` with the same seed/count and compare the two manifests and split hashes. Store the environment report, test output, manifest and evaluation results remotely. Full-ranking metrics are not computed from a winner-only prediction.

## Notebooks

1. `00_environment_check.ipynb`: GPU/VRAM, Torch/CUDA and dependency inventory; not a training-readiness certificate.
2. `01_mcdm_data_and_baselines.ipynb`: generation/revalidation, sanity controls and real offline W&B.
3. `02_d1_3b_lora_colab.ipynb`: candidate recipe, optional tokenizer check, and explicit model/GPU approval gates.
4. `03_evaluate_and_compare.ipynb`: held-out metrics from complete option probability files; clearly reports when d1 predictions are absent.

## Credentials and W&B

The local `.env` is excluded from source bundles. The HF control script reads only a recognized HF token for account/job requests; it never uploads the file. The read-only HF account preflight succeeded. The first GPU run logged step telemetry to [W&B run 56i7klyd](https://wandb.ai/liquid-mcdm-lab/runs/56i7klyd), but its complete post-training evidence remained in repository receipts. The [release run 4vblgw80](https://wandb.ai/black-order/liquid-mcdm-lab/runs/4vblgw80) includes 400 trainer rows, the dataset, adapter, evaluation JSON, environment freeze, configuration, and an 87-row before/after prediction Table as versioned W&B evidence. Both runs were recorded without printing or bundling the key. For a Colab run, use Colab Secrets without displaying their values:

```python
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')
os.environ['WANDB_PROJECT'] = 'liquid-mcdm-lab'
os.environ['WANDB_MODE'] = 'offline'
```

Do not print environment variables, embed keys in notebooks, upload `.env`, or capture notebook source in W&B. Choose the W&B entity and inspect project visibility before online logging. For explicit online logging, use `baseline ... --wandb online --entity YOUR_ENTITY`. Offline runs have no cloud URL. After reviewing staged files, `wandb sync PATH_TO_OFFLINE_RUN` uploads them. Uploaded artifacts are sanitized numeric/identity summaries; detailed slices remain in the local-to-Colab result JSON. No raw private dataset is uploaded by default.

## Release model experiment and persistence

Read [ADR0001](docs/decisions/0001-training-boundary.md). The official TRL recipe uses the standard image-text model, not a generic classifier. The final Runpod environment retained Torch 2.9.1+cu128 and omitted optional torchao; `pip check`, imports, BF16 forward/backward, training, saving and reload inference all executed on the A40. Revalidate the dependency dry run before using a different image.

The release run verified the pinned checkpoint, actual completion mask, frozen vision tower, finite forward/backward pass, nonzero LoRA gradients, custom/standard readout parity, and zero-shot metrics before executing 400 steps. The saved adapter retained every checked decision after reload. Repeated inference and two independent fresh reloads were numerically identical; the in-memory-to-reloaded comparison had maximum probability delta 0.02147 in BF16 and remained below the documented 0.05 tolerance.

Persist checkpoints, config, source snapshot, data manifest and environment freeze to a user-selected Google Drive folder or approved artifact store. A Colab disconnect can delete VM disk. Mounting Drive alone does not copy outputs; explicitly save/copy checkpoints there. Resume requires optimizer/scheduler/trainer state as well as adapter weights and the pinned base snapshot; verify reload predictions before scaling.

Expedia and MSLR are optional pending their access/use agreements. UCI Car is a classification sanity task. Numerical neural/LightGBM controls and behavioral ranking remain future milestones.
