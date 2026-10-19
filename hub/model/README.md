---
license: other
license_name: lfm-open-license-v1.0
license_link: https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE
library_name: peft
base_model: LiquidAI/d1-3B
datasets:
- blackOrder/liquid-mcdm-synthetic
pipeline_tag: text-generation
tags:
- decision-model
- mcdm
- peft
- lora
- copras
- waspas
- vikor
- topsis
---

# d1-3B MCDM LoRA

PEFT LoRA adapter for single-choice imitation of COPRAS, WASPAS, VIKOR, and
TOPSIS decision episodes. It is trained from
[`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) at immutable revision
`051bcc464b01b9f92942b364d9586b0ef5912432`.

## Training contract

The prompt includes the selected method and parameters, criterion directions
and weights, full-precision candidate values, and answer aliases. Completion
loss supervises exactly the correct alias and EOS. Restricted option
probabilities use d1's upstream alias-group readout. The vision tower is frozen.
The adapter targets language attention, convolution, and MLP projections with
rank 16, alpha 32, and dropout 0.05. No quantization is used.

## Loading

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base_id = "LiquidAI/d1-3B"
revision = "051bcc464b01b9f92942b364d9586b0ef5912432"
adapter_id = "blackOrder/d1-3b-mcdm-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
base = AutoModelForImageTextToText.from_pretrained(
    base_id, revision=revision, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
```

Use the versioned prompt and option-probability helpers from
https://github.com/Krishna2709/liquid-mcdm-lab for comparable evaluation.

## Evaluation scope

Run `20261019T005111Z` used 512 training episodes, 400 optimizer steps, and an
NVIDIA A40 in BF16. Accuracy changed as follows:

| Split | Base | Adapted | Episodes |
|---|---:|---:|---:|
| Training subset | 32.42% | 93.55% | 512 |
| Disjoint-family test | 28.74% | 58.62% | 87 |
| Original comparison slice | 37.50% | 75.00% | 16 |

The full test set contains 100 records; 87 have an admissible unique winner and
are eligible for single-choice accuracy. Adapted accuracy by method was 56.0%
for COPRAS, 56.0% for TOPSIS, 66.7% for VIKOR, and 60.0% for WASPAS. The
release includes configuration, environment freeze, dataset manifest hash,
before/after metrics, per-method slices, per-episode predictions, and reload
diagnostics. Results are synthetic exact-teacher imitation and do not establish
human preference quality, full-ranking ability, procurement suitability, or
arithmetic superiority over exact solvers.

The complete training telemetry and release artifacts are in
[W&B run 4vblgw80](https://wandb.ai/black-order/liquid-mcdm-lab/runs/4vblgw80).

## Known limitations

The release adapter preserved all 16 checked winners after serialization.
Repeated inference and two independent fresh reloads were identical. The
in-memory-to-reloaded BF16 probability comparison had maximum absolute delta
0.02147, within the preregistered 0.05 decision-equivalence tolerance but not
bitwise probability parity. Test calibration error increased from 0.1183 to
0.2394 even as accuracy and Brier score improved, so probabilities should not
be interpreted as calibrated confidence. d1-3B and this adapter remain subject
to the LFM Open License v1.0; review its terms before use.
