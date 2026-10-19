# Ordivo v2 research adapter (private)

**Status:** preliminary, access-controlled pilot, **not a release candidate**. This repository is not a public v2 release or a claim of held-out neural improvement. It is separate from the published Ordivo v1 synthetic adapter. The retained bounded pilot trained one shared adapter on Swissmetro stated choice and Amazon ESCI graded relevance. A separate three-source S0b pilot has now completed training with sampled Expedia conditional-booking choice, and a separate bounded held-out probe now has mixed results; its private best adapter passed file-level readback. Neither establishes three-domain generality or a public release dossier. The model repository is `blackOrder/Ordivo-v2-research` and should remain private.

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
than C1 is not evidence of improvement. The [sanitized remote receipt](https://github.com/Krishna2709/ordivo/blob/b6a353549f478b806b50d979bbfc4a5d5d5907d3/reports/runs/v2-continuation-20261019/summary.json) SHA-256
is `cc686cf66ee3d7daa777ecc8b3c1ccaedda5d546da96107ad1f18ea0bf055cc5`
at source `f69643794a932b14dac46b2e866667f5a7814bc4`; no continuation
held-out result is asserted here. Private C1/R0 adapter retention passed forced-download SHA readback under `continuations/20261019T030009Z/matched-f696/` at Hub commit `4b43ec8767d2bb99ec835020c04d412c89cd26a4` (retention-manifest SHA-256 `4c5dc14a52a25e74acfa9a013d4ecc3ba14dcd96d9c21c44f0b61c857bfb367d`). Both [C1](https://wandb.ai/black-order/liquid-mcdm-lab/runs/6ohymgld) and [R0](https://wandb.ai/black-order/liquid-mcdm-lab/runs/57fzcrlp) cloud runs finished; sanitized cloud-readback SHA-256 `f9b02842db5e1d6d0747913a3f3e0a760cdcb95a147d17f45603f7b52c3d4580`.

A separate S0b three-source supervised follow-up completed 90/90 nonzero
optimizer steps at exact integrated source
`945fe6e07df4916f202108d7a56a543041808315`, archive SHA-256
`d655f68a053264cf4c37665504ceecd25c097bc95813c0804342c9bf006e29c4`,
and config SHA-256
`1ad185a26779248dc0b9bf1635eb4c2a6460304cd34479de77941b898ab6403c`.
It used 30 updates per source, a 6,144-token complete-group cap, training-only
candidate permutation and best selected-validation checkpoint step 90. The
mean of source validation losses fell 2.6071312→2.2937011; Swissmetro
1.1358083→0.8775658 (549/549 selected validation episodes eligible), ESCI
3.4676457→2.8772386 (100/100), and Expedia conditional-booking choice
3.2179395→3.1262991 (70/100 selected validation searches eligible; 30 had
no exactly-one-booking supervised event). ESCI training excluded 4/1,000
selected groups for prompt overflow; Expedia training excluded 315/1,000
selected groups for the target event. The [metadata-only pilot receipt](https://github.com/Krishna2709/ordivo/blob/47000cd8bf30318d60912efd863a8e2b18f551d0/reports/runs/v2-s0b-20261019/pilot-receipt.json) SHA-256 is
`85426d547c1fbc88cc52cf9c4fc9c9bfac99fb3bfbd7b256b177c3f069e24565`.
The S0b Expedia source is the predeclared bounded 20k/2k/2k/5k grouped
sample (manifest SHA-256
`c72552b8fe1acd0cab0480610dd65b5dd6aed575313047d2da2f2786953b4a2b`),
not the full 399,344-search source. A [W&B run](https://wandb.ai/black-order/liquid-mcdm-lab/runs/0sq2jfj9)
was online at launch. The best adapter is retained privately under
`supervised-s0b/20261019T030009Z/s0b-945fe6e/` at Hub commit
`df6405c575629dfaf9083f7b068ab8ef34b6eeb3`: eight allowlisted files
plus manifest passed forced-download SHA readback. Adapter directory SHA-256
`e9bc879b31a32e5546d238345babd7d9fc5eedad7e015510804e03fde8a77c50`;
retention-manifest SHA-256
`1bbd441f8ccb2d276ed4028681cb5d550276fd7b690a47642aa02973d24801e5`.
Final cloud readback, calibration and cross-domain transfer remain pending.
These are model-selection validation losses only; a separate bounded S0b
held-out probe follows below.

### S0b bounded paired held-out probe

A separate, exploratory paired probe compared base d1 and the S0b best-step-90
adapter on the same selected units within each source. Exact [sanitized aggregate](https://github.com/Krishna2709/ordivo/blob/47000cd8bf30318d60912efd863a8e2b18f551d0/reports/runs/v2-s0b-20261019/heldout-aggregate.json) SHA-256:
`86753c1356cdf8524a3ea28a2f5b4ff31fb4ef20afaf3f96d6db1cb0fd4b086d`;
executed evaluation source `10232089368296680c75461da2dffe5254ae7590`,
config SHA-256 `b4d889f639cdeaafd46fb24e9b2dccb078e5958178d1553f03acc4a9573594c1`.

| Endpoint | Base d1 | S0b | Eligible coverage |
|---|---:|---:|---|
| Swissmetro choice accuracy@1 | 0.680000 (306/450) | 0.662222 (298/450) | 450/450 scenarios from 50 held-out respondents |
| Swissmetro choice NLL ↓ | 0.805878 | 0.782651 | Same 450 |
| Swissmetro choice Brier ↓ | 0.459254 | 0.463704 | Same 450 |
| ESCI judged-pool NDCG@10 | 0.705590 | 0.714332 | 50/50 selected test groups; zero overflow |
| Expedia exactly-one-booking accuracy@1 | 0.068966 (2/29) | 0.137931 (4/29) | 29/50 selected searches; 21 lack target event |
| Expedia booking-choice NLL ↓ | 2.961689 | 3.016047 | Same 29 |
| Expedia booking-choice Brier ↓ | 0.967797 | 0.941650 | Same 29 |

All admitted units were scored by both models with zero inference failures.
The private per-query outputs and aggregate are retained under
`heldout/20261019T030009Z/s0b-paired-50-1023208` at Hub commit
`9f7110952dbd64aeb8eaa703a328cf196765fa8f`: 12 files plus manifest
passed forced-download SHA readback; retention-manifest SHA-256
`2ab1be61a5709ca8b736d32c9085c8e152a58e65b208e887beb2902b358113c4`. Swiss
accuracy and Brier worsened; ESCI NDCG and Expedia accuracy/Brier improved on
small cohorts, while Expedia NLL worsened. The old S0 ESCI probe scored only 27/50 selected groups; a private-ID
intersection confirms all 27 are in the new 50. On those same 27, base NDCG@10
was 0.778290 in both, old S0 0.752276 and S0b 0.787842. The 23 newly
admitted groups change the 50-group mean, so do not directly compare the old
27-group mean with the new 50-group mean. [Intersection receipt](https://github.com/Krishna2709/ordivo/blob/47000cd8bf30318d60912efd863a8e2b18f551d0/reports/runs/v2-s0b-20261019/esci-old27-match.json) SHA-256 is
`494a1a0a6f2795a3c4f9962fc58f6194a7508f3aa5e9a7cc9ad24cde17479672`. No intervals, calibration fit, multiple seeds or
entire-domain transfer were measured. This is not a protected release test or
a general improvement claim. Expedia is conditional booking within its bounded
exposed-search sample, not general booking probability.

A validation-only candidate-order probe of the original frozen two-source S0 pilot found
52/180 Swissmetro top-choice flips (28.89%) across 20 selected respondents,
and 7/12 admitted ESCI flips (58.33%) from 20 selected queries; eight ESCI
query groups overflowed in both orders. Mean probability L1 change was
0.39663/0.31887. This is positional sensitivity, not an accuracy comparison;
base d1 was not evaluated for this probe. Sanitized aggregate SHA-256:
`1caff28407551758975ab5ebcf67abd3708f44fa8f2fa4a29ead7f04beeabad7`.
The [sanitized order aggregate](https://github.com/Krishna2709/ordivo/blob/b6a353549f478b806b50d979bbfc4a5d5d5907d3/reports/runs/v2-order-20261019/aggregate.json) is metadata-only. Its four private diagnostic output files plus manifest passed forced-download readback under `diagnostics/20261019T030009Z/candidate-order-a19d/` at Hub commit `4434f85c12ab6f7f7bb5894747f3b8f003c075e8` (retention SHA-256 `4a93087601c3ca1b32c7b580d512e2778d4a060cbc92f071066aa03e7a4af54f`). Cloud readback is not asserted here; no public result should
omit the 8/20 ESCI overflow denominator.

## Loading and scoring the frozen S0 label-free research request

The executable fixture below loads only the earlier, two-source S0 adapter at
Hub commit `bf1d425b4e6ffc392121d8fb0776e41b1af48c4a`. It does not load
or validate the separate S0b checkpoint. S0b requires its own Hub folder/revision (`supervised-s0b/20261019T030009Z/s0b-945fe6e/`
at `df6405c575629dfaf9083f7b068ab8ef34b6eeb3`) and run-specific feature
policy, including Expedia `conditional_booking_choice`. Its private retention
is verified, but that checkpoint has not been exercised through this tested
fixture or a held-out result; its parameters are not silently substituted
into this S0 example.

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
all scores were finite and summed to one. The [metadata-only receipt](https://github.com/Krishna2709/ordivo/blob/b6a353549f478b806b50d979bbfc4a5d5d5907d3/reports/runs/v2-inference-20261019/summary.json) SHA-256 is
`9f9832eddaf39df94079cd82a9458839ed4be28e13cde70727943e42c25c3137`
at exact project source `bd24814429211ad4dad4e9a58ba0137daa0bc76b`,
base revision above, private Hub adapter revision above, and adapter directory
SHA-256 `831fb319d8a2144ef26ed86a7c985ed3ca4b9662180ced32e28b6f4c469c612c`.
These are fixture-specific outputs, not benchmark accuracy or calibrated
real-world event probabilities. The unavailable car remains in the complete
prompt but receives probability zero.

For an authorized S0b inspection, use the **separate** verified private
checkpoint parameters below with the same pinned d1 base and the run's
`effective_config.yaml` and source-specific `feature_policy`. No numeric
fixture output has been verified for S0b, so the S0 scores above are not
expected S0b scores:

```python
S0B_HUB_REVISION = "df6405c575629dfaf9083f7b068ab8ef34b6eeb3"
S0B_RUN = "supervised-s0b/20261019T030009Z/s0b-945fe6e"
S0B_ADAPTER_DIRECTORY_SHA256 = "e9bc879b31a32e5546d238345babd7d9fc5eedad7e015510804e03fde8a77c50"
```

For Expedia, the S0b pilot receipt specifies
`prediction_event=conditional_booking_choice`, context features
`srch_length_of_stay`, `srch_booking_window`, `srch_adults_count`,
`srch_children_count`, `srch_room_count`, `srch_saturday_night_bool`, and
alternative features `price_usd`, `prop_starrating`, `prop_review_score`,
`prop_location_score1`, `prop_location_score2`. That event's output is a
conditional distribution only over an exposed complete search group eligible
under the exactly-one-booking policy; it is not a marginal booking rate. The
S0b readout was evaluated on the bounded held-out groups above, but this
illustrative S0 request was not rerun with S0b.

## Use and distribution

The d1 base is under [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE). Source-specific dataset terms and notices must travel with any private artifacts as applicable. The ESCI source repository declares [Apache-2.0](https://github.com/amazon-science/esci-data/blob/main/LICENSE); Swissmetro source and derived-weight distribution terms need a source-specific record. Expedia competition data is subject to competition rules. It is absent from the frozen S0 adapter used in the tested example, but a bounded, sampled Expedia conditional-booking source was included in separate S0b training. Private research storage does not establish permission for public adapter redistribution. No license is assigned to this adapter by this draft README.
