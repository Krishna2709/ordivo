# Progress — 2026-10-18/19

## Release outcome

Run `20261019T005111Z` completed on a secure-cloud NVIDIA A40 48 GB Runpod at
$0.59/hour. It reused network volume `ngica7qkhd` and the previously validated
remote Python environments. No model, dataset generation, test suite, or
training workload ran on the local machine.

Remote CPU acceptance passed **130 tests in 31.45 seconds**. Two generated
1,000-episode datasets were byte-identical. The release dataset contains 800
training, 100 validation, and 100 test records; 87 test records have an
admissible unique winner. Its manifest SHA-256 is
`1235dd1439935f217ab7fca62746fa7ddf28806ace867810b349abc147e4e1b2`.

The pinned checkpoint was
`LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432`. The runtime used
Python 3.12.3, Torch 2.9.1+cu128, CUDA 12.8, BF16, rank-16 LoRA, and no
quantization. The compatibility pass supervised exactly the answer alias plus
EOS, found finite loss 0.80575, gave nonzero finite gradients to 166 of 332
trainable LoRA tensors, and left the vision tower frozen.

Training completed 400 optimizer steps in 556.5 seconds with aggregate loss
0.48215. The results were:

| Evaluation | Base | Adapted | Delta | Eligible |
|---|---:|---:|---:|---:|
| Training subset | 0.3242 | 0.9355 | +0.6113 | 512 |
| Full disjoint-family test | 0.2874 | 0.5862 | +0.2989 | 87 |
| Original comparison slice | 0.3750 | 0.7500 | +0.3750 | 16 |

Adapted test accuracy was 0.56 COPRAS, 0.56 TOPSIS, 0.6667 VIKOR, and 0.60
WASPAS. Accuracy improved for every method. The multiclass Brier score improved
from 0.7811 to 0.6231, but 10-bin ECE worsened from 0.1183 to 0.2394. Near-tie
accuracy was 0.4167 versus 0.6508 on non-near-tie cases. These results support
an algorithm-imitation claim only; they do not establish calibrated confidence
or operational decision quality.

## Reload evidence

All 16 checked saved-adapter decisions matched the in-memory model. Repeated
inference on one loaded adapter and two independent fresh reloads were
numerically identical. The in-memory-to-reloaded comparison had maximum
absolute probability delta 0.02147 and mean delta 0.00111. This is within the
documented 0.05 BF16 decision-equivalence tolerance, but it is not bitwise
probability parity.

## Publication and tracking

- Source and small receipts: https://github.com/Krishna2709/liquid-mcdm-lab
- Dataset: https://huggingface.co/datasets/blackOrder/liquid-mcdm-synthetic
- Adapter and evaluation evidence: https://huggingface.co/blackOrder/d1-3b-mcdm-lora
- First W&B telemetry run: https://wandb.ai/liquid-mcdm-lab/runs/56i7klyd
- Release W&B run: https://wandb.ai/black-order/liquid-mcdm-lab/runs/4vblgw80

The first W&B run contains trainer telemetry but not every final artifact. The
release run logs trainer telemetry plus the deterministic dataset, LoRA
adapter, all evaluation JSON, environment freeze, configuration, and an
87-row before/after prediction Table. Publication receipts contain SHA-256
hashes and read-back verification; the large adapter weights remain remote.

## Failure resolution

The official notebook's open-ended `torchao>0.16` constraint was incompatible
with the Runpod Torch 2.9.1 image. The successful non-quantized path omits
optional torchao. TRL 1.15 accepts a fractional value through `warmup_steps`,
so the configured 0.05 fraction is mapped there. The v2 preflight initially
requested more eligible test cases than existed; it was corrected to all 87
eligible episodes before model loading or training. See
[`docs/run_failure_analysis.md`](../docs/run_failure_analysis.md) for the full
attempt inventory.
