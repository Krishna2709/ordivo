# Ordivo v2 d1 training compatibility

Status: source implementation prepared; remote tensor and GPU execution receipts pending.

## Readout and objective

The pinned `LiquidAI/d1-3B` revision is
`051bcc464b01b9f92942b364d9586b0ef5912432`. Its
[prompt implementation](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/prompt.py)
scores each option by the maximum answer-slot logit among the single-token alias
and available leading-space form, then applies a softmax over options. The
vendored `d1_prompt.readout` returns Python floats for inference. The new
`training.option_policy` uses Torch operations for the same MAX → softmax path
and supports an availability mask. NLL and multiclass Brier are conditional
choice objectives over complete groups. They are not sequence-policy log
probabilities. We do not use them as a GRPO completion log probability.

The supervised reference remains alias-plus-EOS completion CE through
`AutoModelForImageTextToText` and PEFT LoRA, following the
[official Liquid TRL notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb).
Direct NLL/Brier is a separate controlled experiment. Temperature calibration
fits only a fourth, disjoint calibration partition for an explicitly named
event; SFT/early-stop validation and final test cannot enter the fitting function.
The fitting API reads all four frozen split files, checks hashes and group
disjointness against the manifest, verifies the selected calibration group IDs,
requires the full eligible calibration population, and records its digest.
Applying the temperature requires the same event,
candidate-population protocol, and manifest hash. Accepted events are
`conditional_booking_choice` (only among exposed, eligible, exactly-one-booking
groups) and `stated_choice`. Graded relevance is not a choice event.

The source-only `train_v2_supervised.py` defines a bounded shared-adapter pilot:
equal-domain round-robin sampling, one complete query per update, categorical
NLL for stated/conditional booking choices and normalized-gain listwise loss
for observed/graded relevance. Zero-gain groups are counted as excluded from
loss; they are not converted into a false preference target. It uses train and
validation only, caps optimizer steps and runtime, selects on macro-domain
validation loss, saves the best adapter, and checks reload. The sample config
is not an execution authorization. Real source acquisition, rights and baseline
evidence must be reviewed before starting this pilot; two domains would still
be preliminary research.
The identical frozen validation subset is scored before any optimizer update
and at the selected best adapter. Base/best macro and per-source losses are
recorded in the remote receipt and W&B; these are optimization diagnostics,
not an accuracy or transfer claim.
Model adapters and receipts remain on the remote runtime. The trainer rejects
`publish_model_artifacts: true`; changing W&B mode can publish aggregate metrics
but cannot silently upload derived weights. Artifact publication needs a
separate source-rights and destination-visibility review.
The pilot streams all four source files to verify hashes and split-unit
disjointness, then selects whole groups by a fixed seed/hash before rendering.
Defaults cap selection at 1,000 train and 100 validation split units per source;
panel respondents remain intact. It records source totals, selected counts and
digests, and loss exclusions including zero/no event and prompt overflow. Actual
selected IDs remain in a remote-only file outside W&B metrics. The wall-clock cap
covers integrity passes, rendering, model loading, updates and validation.
The controller must supply `ORDIVO_SOURCE_COMMIT` (40 hex) and
`ORDIVO_SOURCE_SHA256` (64 hex) from the verified source archive; the runtime
does not need a `.git` checkout. The Swissmetro plus ESCI fallback config uses
reviewed allowlists and remains a preliminary two-domain pilot if Expedia
access is blocked. Its ESCI evidence budget includes query and locale context,
plus title, brand and color for every candidate; descriptions and bullets are
excluded explicitly to limit context overflow. It does not silently remove
candidates, and records overflows. A comparative baseline claim requires a
baseline rerun with the same feature policy. Swissmetro uses survey context
(`purpose`, `ga`, `who`, `luggage`) and recorded travel time, cost and headway;
these are observed features, not a learned cost or GA preference rule.

## Remote evidence gates

On the operations-owned RunPod A40, first run tensor-only tests with a
committed failing state and then the implementation state. Run the bounded
GPU script against the pinned checkpoint and record its JSON receipt. It loads
the same base checkpoint with the standard training class, compares Torch
readout with pinned inference, checks unavailable-option masking, backprops one
NLL example through LoRA, and saves/reloads an adapter. A pass establishes only
mechanical compatibility for one fixed prompt. It does not establish behavioral
learning, calibration, ranking performance, or GRPO compatibility.

Required command, after the source commit is copied to the remote checkout:

```bash
python -m pytest -q tests/test_option_policy.py tests/test_behavioral_rendering.py tests/test_calibration.py
python scripts/check_v2_training_compatibility.py --output /workspace/receipts/v2-training-compatibility
```

The GPU receipt must show finite loss, nonzero finite LoRA gradients, exact
masking, pinned readout agreement, and adapter save/reload agreement. The
operations owner records actual package versions, runtime ID, costs and cleanup.
No behavioral optimization pilot is authorized by this document; that requires
data and baseline evidence reviewed by the root task owner.
