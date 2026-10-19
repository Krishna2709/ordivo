# Ordivo v2 research adapter (private)

**Status:** preliminary, access-controlled pilot, **not a release candidate**. This repository is not a public v2 release or a claim of held-out neural improvement. It is separate from the published Ordivo v1 synthetic adapter. The current bounded pilot trained one shared adapter on Swissmetro stated choice and Amazon ESCI graded relevance; it has no Expedia training, no three-domain generality result, and no completed public release dossier. The model repository is `blackOrder/Ordivo-v2-research` and should remain private.

## Exact model and task boundary

The pilot uses `LiquidAI/d1-3B` at immutable revision `051bcc464b01b9f92942b364d9586b0ef5912432`, with a PEFT LoRA `best_adapter` selected on the **model-selection validation** groups. The executed source commit is `854de0877c8e7e380b7353a578e95ba9865d4e25`; source archive SHA-256 is `38454c8d667183d57c9c48ed0bcc3d3b32f214e2485c6bdb6bf99f6af0204faa`; config SHA-256 is `c152e64d839082531d4ece9e66d330fa35ffddfbc442685742b9fd171de5462b`. Prompt contract `behavioral-choice-v1` lives in `liquid_mcdm.training.behavioral_rendering`; option readout lives in `liquid_mcdm.training.option_policy` and the pinned `liquid_mcdm._vendor.d1_prompt`. The base model, project code, tokenizer, and adapter are all required. This is an answer-slot candidate scorer, not a generated-text decision assistant.

Input is a complete `ordivo-behavioral-v1` episode with task type, declared context/criteria/constraints/preferences, and all candidates with typed source features and `available` masks. The rendering path uses a source-specific `feature_policy` allowlist. Candidate IDs are output keys; only positional aliases reach the model. Labels, exposure/position, provenance, source/group identifiers, and original row order must not be predictor features. The label-free research renderer accepts only an exact predictor envelope and frozen feature-policy allowlist, without supervision, exposure or provenance. Its scoped/full source tests and one private-adapter illustrative inference playthrough passed. A production serving API remains pending.

For `stated_choice`, output is a conditional distribution over **the supplied available alternatives**. For ESCI `graded_relevance`, the current option distribution is a single-choice relevance probe over the **source Task 1 reduced judged candidate list**; it is not product retrieval recall, purchase probability, a calibrated confidence, or a universal utility scale. If an `observed_choice` source is added, `prediction_event` must distinguish `conditional_booking_choice` from `combined_click_booking_relevance`; a conditional booking distribution applies only to exposed groups with exactly one booking. Unavailable options are masked, and no complete candidate group may be truncated to fit the prompt.

## Research artifact ledger

The remote pilot receipt records 40 optimizer steps, best step 40, 20 updates per source, a nonzero LoRA update, and zero observed reload-probe loss delta. Swissmetro selection-validation loss changed from 1.1358 to 0.8577; ESCI from 3.1603 to 2.6311 on the selected eligible groups. ESCI rendered 71 of 100 selected validation groups; 29 complete groups exceeded the 2,048-token cap and were excluded explicitly. These are training-selection diagnostics, **not held-out accuracy, ranking quality, calibration, or transfer evidence**. A bounded paired held-out probe has now measured mixed/negative results below; paired intervals and protected release evaluation remain pending. The receipt SHA-256 is `5f58e02b3259601d9663892f50e32460a57be2d424724ec283d87ecb8678f7c1`.

| Expected private run-folder item | State / verification rule |
|---|---|
| `supervised-pilot/20261019T030009Z/pilot-854de/best_adapter/` and tokenizer files | Private upload and forced SHA readback passed; `adapter_model.safetensors` SHA-256 `54e90e7b5c4dad3a470345c8462aa4ad7527829b702c81aa2a45ded7475066c7` |
| `supervised-pilot/20261019T030009Z/pilot-854de/pilot_receipt.json` | Private upload/readback passed; SHA-256 `5f58e02b3259601d9663892f50e32460a57be2d424724ec283d87ecb8678f7c1` |
| `supervised-pilot/20261019T030009Z/pilot-854de/effective_config.yaml` and `environment.json` | Private upload/readback passed; exact hashes in manifest |
| `supervised-pilot/20261019T030009Z/pilot-854de/artifact_manifest.json` | Private upload/readback passed; nine allowlisted files plus manifest; Hub commit `bf1d425b4e6ffc392121d8fb0776e41b1af48c4a` |
| Bounded paired held-out neural probe | Measured below; adapted model did not improve accuracy or NDCG on admitted groups; not the protected release test |
| Full evaluation, calibration, cost, and rights decision | Pending; no release decision follows from this pilot |

The private dataset repository is `blackOrder/Ordivo-v2-research-data`. Its source manifests define four disjoint roles: train, model-selection validation, calibration, and final test. The pilot consumed Swissmetro manifest SHA-256 `dd3dc4b0713506f1664149993ca11022614a3ee850c335944ae37b5525af592a` and ESCI manifest SHA-256 `9fd4065fc0db4586453a412123eb9ef393b1750cd8196bb11cad794d020f07cc`. Keep exact executed hashes, even if a later metadata-only manifest revision preserves split bytes.

## Bounded held-out probe: mixed and negative results

An exploratory paired probe used the same selected held-out units for base d1 and the private adapted pilot. The [sanitized aggregate](https://github.com/Krishna2709/ordivo/blob/1955a2ced9fa44cb71e5bee5e19f3002fe522f67/reports/runs/v2-neural-heldout-20261019/aggregate.json) has SHA-256 `a735c1db8810f9d9955152bbe16651e690c921e7330596cedc117d2358a19c8a`, executed source `854de0877c8e7e380b7353a578e95ba9865d4e25`, evaluation config SHA-256 `524bfe7cde8135132417ed029a8fb2738bd5ffd068f9190d081c6246aaaadc06`, and the frozen source-manifest hashes above. A private [W&B run](https://wandb.ai/black-order/liquid-mcdm-lab/runs/3zdzt6ik) records sanitized aggregates; per-query files remain private.

| Endpoint | Base d1 | Adapted pilot | Coverage |
|---|---:|---:|---|
| Swissmetro stated-choice accuracy@1 | 0.68 (306/450) | 0.66 (297/450) | 450/450 selected scenarios from 50 held-out respondents |
| Swissmetro choice NLL ↓ | 0.805878 | 0.758958 | Same 450 scenarios |
| Swissmetro choice Brier ↓ | 0.459254 | 0.462190 | Same 450 scenarios |
| ESCI judged-pool NDCG@10 | 0.778290 | 0.752276 | 27/50 selected held-out query groups; 23 prompt overflows |

The adapted model loses 2 percentage points of Swissmetro top-1 accuracy and 0.0260146 ESCI NDCG@10 on admitted groups. NLL improves while Brier worsens on Swissmetro. There are no paired intervals, calibration fit, multi-seed replications, or entire-held-out-domain transfer result. ESCI prompt overflow is a material coverage failure; do not interpret 27/50 admitted queries as full test coverage. This probe is not the protected final release test. Private retention includes eight files plus manifest under `heldout-probe/20261019T030009Z/paired-854de/` at Hub commit `04656dae99ff3d28a0e8a7ac3f22275335cb6ad3`; every file passed forced-download SHA readback. The local retention-manifest SHA-256 is `606510b5833b6264755ff805ea50df1f4bf90e76449831489f3553631dc3d163`. No per-query record belongs in a public report.

A subsequent matched C1 direct-Brier and R0 stochastic-report continuation
completed 40 optimizer updates per arm from the same starting pilot adapter.
Their best **post-update** checkpoints (step 10; step 0 excluded from that
selection rule) both worsened selected-validation macro loss relative to the
shared start: start 1.5817367, C1 1.6459532, R0 1.6163764. R0 degrading less
than C1 is not evidence of improvement. The sanitized remote receipt SHA-256
is `cc686cf66ee3d7daa777ecc8b3c1ccaedda5d546da96107ad1f18ea0bf055cc5`
at source `f69643794a932b14dac46b2e866667f5a7814bc4`; no continuation
held-out result, cloud readback or private Hub retention is asserted here.

A separate S0b three-source supervised follow-up is configured after token
coverage review, but no executed S0b result is recorded in this card yet. Its
setup does not change the frozen pilot results above.

A validation-only candidate-order probe of this frozen pilot found
52/180 Swissmetro top-choice flips (28.89%) across 20 selected respondents,
and 7/12 admitted ESCI flips (58.33%) from 20 selected queries; eight ESCI
query groups overflowed in both orders. Mean probability L1 change was
0.39663/0.31887. This is positional sensitivity, not an accuracy comparison;
base d1 was not evaluated for this probe. Sanitized aggregate SHA-256:
`1caff28407551758975ab5ebcf67abd3708f44fa8f2fa4a29ead7f04beeabad7`.
Per-query retention and cloud readback remain pending; no public result should
omit the 8/20 ESCI overflow denominator.

## Loading and scoring a label-free research request

Use integrated project source `bd24814429211ad4dad4e9a58ba0137daa0bc76b` or a reviewed descendant with its dependency lock. The label-free `render_predictor_input` uses the same prompt/readout core as the executed pilot; the remote scoped 16-test and integrated 244-test suites passed at this source commit (archive SHA-256 `778a3689f90c44afa572fc4ecd968a8853e53f5c5bb81e3356338485dd57c2a0`). The private adapter files were force-downloaded and matched upload SHA-256 at Hub commit `bf1d425b4e6ffc392121d8fb0776e41b1af48c4a`; inspect its private artifact manifest when reproducing. The **illustrative** Swissmetro-style input below has no target, supervision, exposure, provenance, respondent, group or source identifier. Its numbers are examples, not a dataset row or expected model prediction. Apply the exact feature allowlist from the frozen pilot config to real authorized inputs.

Run from an authenticated, access-controlled **CUDA GPU** environment with
compatible PyTorch, Transformers, PEFT and Hugging Face Hub packages; the private folder contains the [retained environment inventory](https://huggingface.co/blackOrder/Ordivo-v2-research/blob/bf1d425b4e6ffc392121d8fb0776e41b1af48c4a/supervised-pilot/20261019T030009Z/pilot-854de/environment.json) and [effective pilot config](https://huggingface.co/blackOrder/Ordivo-v2-research/blob/bf1d425b4e6ffc392121d8fb0776e41b1af48c4a/supervised-pilot/20261019T030009Z/pilot-854de/effective_config.yaml) at the pinned Hub commit. Authenticate to the private Hub repository through your
normal secret-backed environment or `hf auth login`; do not paste or print a
token in a notebook or command log. The code below imports the versioned
project package, so run it from a checkout of the pinned project source with
`PYTHONPATH=src` (or install that exact checkout as a package):

```bash
git clone https://github.com/Krishna2709/ordivo.git
cd ordivo
git checkout bd24814429211ad4dad4e9a58ba0137daa0bc76b
# In the approved GPU environment with the pinned dependency set and Hub auth:
PYTHONPATH=src python private_v2_probe.py
```

Save the Python block below as `private_v2_probe.py` in that checkout. The
remote source test suite used the same `PYTHONPATH=src` import boundary. The
illustrative inference playthrough was verified with the pinned source, base,
private adapter and environment inventory. This constructed input is a fixture,
not a row from a benchmark or a measurement of prediction quality.

```python
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from liquid_mcdm.training.behavioral_rendering import render_predictor_input
from liquid_mcdm.training.option_policy import option_probabilities

request = {
    "schema_version": "ordivo-behavioral-v1",
    "task_type": "stated_choice",
    "context": {"purpose": 1, "ga": 0, "who": 1, "luggage": 0},
    "criteria": [
        {"name": "travel_time_minutes", "direction": "min", "weight": None},
        {"name": "cost_chf", "direction": "min", "weight": None},
    ],
    "constraints": [],
    "preferences": {},
    "alternatives": [
        {"id": "train", "features": {"travel_time_minutes": 90.0, "cost_chf": 70.0, "headway_minutes": 30.0}, "available": True},
        {"id": "swissmetro", "features": {"travel_time_minutes": 75.0, "cost_chf": 85.0, "headway_minutes": 20.0}, "available": True},
        {"id": "car", "features": {"travel_time_minutes": 110.0, "cost_chf": 60.0, "headway_minutes": None}, "available": False},
    ],
}
feature_policy = {
    "context_features": ["purpose", "ga", "who", "luggage"],
    "alternative_features": ["travel_time_minutes", "cost_chf", "headway_minutes"],
}
BASE = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
RUN = "supervised-pilot/20261019T030009Z/pilot-854de"
snapshot = snapshot_download(
    "blackOrder/Ordivo-v2-research",
    revision="bf1d425b4e6ffc392121d8fb0776e41b1af48c4a",
    allow_patterns=[f"{RUN}/best_adapter/*", f"{RUN}/artifact_manifest.json"],
)
adapter_dir = Path(snapshot) / RUN / "best_adapter"
tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
base = AutoModelForImageTextToText.from_pretrained(
    BASE, revision=REVISION, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()
rendered = render_predictor_input(request, tokenizer, feature_policy, max_length=2048)
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

The remote A40 playthrough produced `train: 0.24508503079414368`,
`swissmetro: 0.7549149990081787`, and unavailable `car: 0.0` for this
constructed request. It rendered all three candidates in 345 input tokens;
all scores were finite and summed to one. The metadata-only receipt SHA-256 is
`9f9832eddaf39df94079cd82a9458839ed4be28e13cde70727943e42c25c3137`
at exact project source `bd24814429211ad4dad4e9a58ba0137daa0bc76b`,
base revision above, private Hub adapter revision above, and adapter directory
SHA-256 `831fb319d8a2144ef26ed86a7c985ed3ca4b9662180ced32e28b6f4c469c612c`.
These are fixture-specific outputs, not benchmark accuracy or calibrated
real-world event probabilities. The unavailable car remains in the complete
prompt but receives probability zero.

## Use and distribution

The d1 base is under [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE). Source-specific dataset terms and notices must travel with any private artifacts as applicable. The ESCI source repository declares [Apache-2.0](https://github.com/amazon-science/esci-data/blob/main/LICENSE); Swissmetro source and derived-weight distribution terms need a source-specific record. Expedia competition data is subject to competition rules and is absent from this pilot. Private research storage does not establish permission for public adapter redistribution. No license is assigned to this adapter by this draft README.
