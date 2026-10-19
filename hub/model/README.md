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

## Using the published model

The adapter accepts a structured numerical decision problem. It does not accept
a free-form business question. Each request supplies an MCDM method, its
parameters, weighted benefit/cost criteria, and the value of every alternative
on every criterion.

### Input fields

| Field | Required value |
|---|---|
| `method` | `COPRAS`, `WASPAS`, `VIKOR`, or `TOPSIS` |
| `method_params` | `{}` for COPRAS/TOPSIS, `{"lambda": 0.0..1.0}` for WASPAS, or `{"v": 0.0..1.0}` for VIKOR |
| `criteria` | Unique `name`, `direction` (`min` or `max`), and nonnegative `weight` for each criterion |
| `alternatives` | Unique `id` plus a `values` mapping containing every criterion name |

For inputs closest to the training distribution, use 3–8 alternatives, 2–6
criteria, weights summing to one, and finite strictly positive measurements.
Criterion names must match the keys in every alternative's `values` mapping.
The `target`, solver scores, ranking, provenance, and episode ID are not model
inputs.

### Sample input

This is published test episode `synthetic-v1-42-2-000000` with target fields
removed:

```python
decision = {
    "method": "COPRAS",
    "method_params": {},
    "criteria": [
        {
            "name": "criterion_0",
            "direction": "min",
            "weight": 1 / 3,
            "unit": "synthetic",
        },
        {
            "name": "criterion_1",
            "direction": "max",
            "weight": 1 / 3,
            "unit": "synthetic",
        },
        {
            "name": "criterion_2",
            "direction": "min",
            "weight": 1 / 3,
            "unit": "synthetic",
        },
    ],
    "alternatives": [
        {
            "id": "A",
            "values": {
                "criterion_0": 344.958426505142,
                "criterion_1": 174.71343616314272,
                "criterion_2": 4.374510512261219,
            },
        },
        {
            "id": "B",
            "values": {
                "criterion_0": 2074.7917654961116,
                "criterion_1": 40.98290161089295,
                "criterion_2": 24.87462780994877,
            },
        },
        {
            "id": "C",
            "values": {
                "criterion_0": 371.29539511555544,
                "criterion_1": 185.01329033444466,
                "criterion_2": 4.911120661391979,
            },
        },
        {
            "id": "D",
            "values": {
                "criterion_0": 4438.9639476302755,
                "criterion_1": 14.010542557524218,
                "criterion_2": 59.1768121163083,
            },
        },
    ],
}
```

### GPU inference

Use a remote CUDA environment with a compatible PyTorch installation. The
release environment used Torch 2.9.1+cu128, Transformers 5.19.0, PEFT 0.21.2,
Accelerate 1.15.0, and BF16.

```bash
python -m pip install \
  "transformers==5.19.0" \
  "peft==0.21.2" \
  "accelerate==1.15.0" \
  "safetensors==0.8.0" \
  "git+https://github.com/Krishna2709/liquid-mcdm-lab.git@v0.1.0"
```

After defining `decision` as above:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from liquid_mcdm._vendor import d1_prompt
from liquid_mcdm.training.rendering import decision_input, option_probabilities

BASE_MODEL = "LiquidAI/d1-3B"
BASE_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
ADAPTER = "blackOrder/d1-3b-mcdm-lora"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

state, question = decision_input(decision)
prompt = d1_prompt.render(
    tokenizer,
    state,
    question,
    bos=tokenizer.bos_token,
)

device = next(model.parameters()).device
inputs = tokenizer(
    prompt,
    add_special_tokens=False,
    return_tensors="pt",
)
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

with torch.inference_mode():
    logits = model(
        **inputs,
        logits_to_keep=1,
    ).logits[0, -1].float().cpu()

alternative_ids = [item["id"] for item in decision["alternatives"]]
probabilities = option_probabilities(tokenizer, alternative_ids, logits)
result = {
    "winner": max(probabilities, key=probabilities.get),
    "probabilities": probabilities,
}
print(result)
```

### Expected result

The recorded BF16 A40 result for this exact published episode was:

```python
{
    "winner": "A",
    "probabilities": {
        "A": 0.8771265580649198,
        "B": 0.003367434875200803,
        "C": 0.11870617117007114,
        "D": 0.0007998358898083066,
    },
}
```

Small floating-point differences are possible across GPU and software stacks.
The output is one winning alternative and a probability distribution over the
provided alternatives; it is not a complete ranking. The probabilities are
not calibrated confidence scores. The exact recorded row is available in the
[release prediction evidence](https://huggingface.co/blackOrder/d1-3b-mcdm-lora/blob/main/evidence/prediction_rows.json).

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
