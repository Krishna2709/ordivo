# Ordivo v2 research adapter (private)

**Status:** preliminary, access-controlled research. This repository is not a public v2 release or a claim of held-out neural quality. It is separate from the published Ordivo v1 synthetic adapter. The current bounded pilot trained one shared adapter on Swissmetro stated choice and Amazon ESCI graded relevance; it has no Expedia training, no three-domain generality result, and no completed public release dossier. The model repository is `blackOrder/Ordivo-v2-research` and should remain private.

## Exact model and task boundary

The pilot uses `LiquidAI/d1-3B` at immutable revision `051bcc464b01b9f92942b364d9586b0ef5912432`, with a PEFT LoRA `best_adapter` selected on the **model-selection validation** groups. The executed source commit is `854de0877c8e7e380b7353a578e95ba9865d4e25`; source archive SHA-256 is `38454c8d667183d57c9c48ed0bcc3d3b32f214e2485c6bdb6bf99f6af0204faa`; config SHA-256 is `c152e64d839082531d4ece9e66d330fa35ffddfbc442685742b9fd171de5462b`. Prompt contract `behavioral-choice-v1` lives in `liquid_mcdm.training.behavioral_rendering`; option readout lives in `liquid_mcdm.training.option_policy` and the pinned `liquid_mcdm._vendor.d1_prompt`. The base model, project code, tokenizer, and adapter are all required. This is an answer-slot candidate scorer, not a generated-text decision assistant.

Input is a complete `ordivo-behavioral-v1` episode with task type, declared context/criteria/constraints/preferences, and all candidates with typed source features and `available` masks. The rendering path uses a source-specific `feature_policy` allowlist. Candidate IDs are output keys; only positional aliases reach the model. Labels, exposure/position, provenance, source/group identifiers, and original row order must not be predictor features. The current evaluator accepts a validated full episode and projects its predictor-only input internally; an independent serving-input validator is not yet packaged. Do not pass arbitrary unvalidated dictionaries as if they were a supported public API.

For `stated_choice`, output is a conditional distribution over **the supplied available alternatives**. For ESCI `graded_relevance`, the current option distribution is a single-choice relevance probe over the **source Task 1 reduced judged candidate list**; it is not product retrieval recall, purchase probability, a calibrated confidence, or a universal utility scale. If an `observed_choice` source is added, `prediction_event` must distinguish `conditional_booking_choice` from `combined_click_booking_relevance`; a conditional booking distribution applies only to exposed groups with exactly one booking. Unavailable options are masked, and no complete candidate group may be truncated to fit the prompt.

## Research artifact ledger

The remote pilot receipt records 40 optimizer steps, best step 40, 20 updates per source, a nonzero LoRA update, and zero observed reload-probe loss delta. Swissmetro selection-validation loss changed from 1.1358 to 0.8577; ESCI from 3.1603 to 2.6311 on the selected eligible groups. ESCI rendered 71 of 100 selected validation groups; 29 complete groups exceeded the 2,048-token cap and were excluded explicitly. These are training-selection diagnostics, **not held-out accuracy, ranking quality, calibration, or transfer evidence**. Neural held-out evaluation and paired uncertainty remain pending. The receipt SHA-256 is `5f58e02b3259601d9663892f50e32460a57be2d424724ec283d87ecb8678f7c1`.

| Expected private run-folder item | State / verification rule |
|---|---|
| `supervised-pilot/20261019T030009Z/pilot-854de/best_adapter/` and tokenizer files | Private upload and forced SHA readback passed; `adapter_model.safetensors` SHA-256 `54e90e7b5c4dad3a470345c8462aa4ad7527829b702c81aa2a45ded7475066c7` |
| `supervised-pilot/20261019T030009Z/pilot-854de/pilot_receipt.json` | Private upload/readback passed; SHA-256 `5f58e02b3259601d9663892f50e32460a57be2d424724ec283d87ecb8678f7c1` |
| `supervised-pilot/20261019T030009Z/pilot-854de/effective_config.yaml` and `environment.json` | Private upload/readback passed; exact hashes in manifest |
| `supervised-pilot/20261019T030009Z/pilot-854de/artifact_manifest.json` | Private upload/readback passed; nine allowlisted files plus manifest; Hub commit `bf1d425b4e6ffc392121d8fb0776e41b1af48c4a` |
| Held-out neural evaluation, calibration, cost, and rights decision | Pending; no value should be inferred from validation loss |

The private dataset repository is `blackOrder/Ordivo-v2-research-data`. Its source manifests define four disjoint roles: train, model-selection validation, calibration, and final test. The pilot consumed Swissmetro manifest SHA-256 `dd3dc4b0713506f1664149993ca11022614a3ee850c335944ae37b5525af592a` and ESCI manifest SHA-256 `9fd4065fc0db4586453a412123eb9ef393b1750cd8196bb11cad794d020f07cc`. Keep exact executed hashes, even if a later metadata-only manifest revision preserves split bytes.

## Loading and scoring a validated research episode

Use the pinned project source and dependencies from the executed run. The private Hub files were force-downloaded and matched the upload SHA-256 values at commit `bf1d425b4e6ffc392121d8fb0776e41b1af48c4a`; inspect the private artifact manifest when reproducing. `episode` below is one authorized, complete `ordivo-behavioral-v1` episode validated under its source contract; do not upload or log it. Choose the matching allowlist from the frozen run config, not a superset constructed from labels.

```python
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from liquid_mcdm.training.behavioral_rendering import render_behavioral
from liquid_mcdm.training.option_policy import option_probabilities

BASE = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
RUN = "supervised-pilot/20261019T030009Z/pilot-854de"
snapshot = snapshot_download(
    "blackOrder/Ordivo-v2-research",
    revision="bf1d425b4e6ffc392121d8fb0776e41b1af48c4a",
    allow_patterns=[f"{RUN}/best_adapter/*", f"{RUN}/artifact_manifest.json"],
)
adapter_dir = Path(snapshot) / RUN / "best_adapter"
# `episode` and `feature_policy` are supplied by the authorized source adapter.
tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
base = AutoModelForImageTextToText.from_pretrained(
    BASE, revision=REVISION, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()
rendered = render_behavioral(episode, tokenizer, feature_policy, max_length=2048)
inputs = tokenizer(rendered["prompt"], add_special_tokens=False, return_tensors="pt")
inputs = inputs.to(next(model.parameters()).device)
with torch.inference_mode():
    answer_logits = model(**inputs, logits_to_keep=1).logits[0, -1]
    eligible = torch.tensor(rendered["eligible"], dtype=torch.bool, device=answer_logits.device)
    probabilities = option_probabilities(
        answer_logits, rendered["readout_groups"], eligible
    )
scores_by_candidate_id = dict(zip(rendered["candidate_ids"], probabilities.cpu().tolist()))
```

This softmax is conditional on this candidate group and rendering policy. Do not interpret it as calibrated real-world event probability. No expected prediction is asserted without a separately checked episode-level inference receipt.

## Use and distribution

The d1 base is under [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE). Source-specific dataset terms and notices must travel with any private artifacts as applicable. The ESCI source repository declares [Apache-2.0](https://github.com/amazon-science/esci-data/blob/main/LICENSE); Swissmetro source and derived-weight distribution terms need a source-specific record. Expedia competition data is subject to competition rules and is absent from this pilot. Private research storage does not establish permission for public adapter redistribution. No license is assigned to this adapter by this draft README.
