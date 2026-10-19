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

The model release includes complete configuration, environment freeze, dataset
manifest hash, before/after metrics, per-method slices, prediction files,
reload diagnostics, and W&B run links. Results are from synthetic exact-teacher
supervision. They do not establish human preference quality, full-ranking
ability, procurement suitability, or arithmetic superiority over exact
solvers.

## Known limitations

The first diagnostic adapter preserved all four checked reload winners but
showed a maximum probability delta of 0.03060 against the in-memory model,
above the experiment's strict 1e-4 threshold. Release revisions document the
follow-up investigation. d1-3B and this adapter remain subject to the LFM Open
License v1.0; review its commercial and redistribution terms before use.
