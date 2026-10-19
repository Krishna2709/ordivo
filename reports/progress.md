# Progress — 2026-10-18/19

## Current outcome

The first bounded Liquid d1-3B LoRA fine-tune is complete. Run
`20261019T002734Z` executed on a secure-cloud NVIDIA A40 48 GB Runpod at
$0.59/hour. The paid pod was deleted and verified absent after approximately
0.628 hours, an estimated $0.37 of pod time. Network volume `ngica7qkhd`
remains because it contains the adapter and full remote evidence; storage
charges continue until that volume is deleted.

The adapter is at
`/workspace/runs/20261019T002734Z/training/adapter` on that volume. It contains
115,818,206 bytes. The weights file is 97,894,520 bytes with SHA-256
`ae1ba85962d6f4c87f14b055c6290ef142a25d6794207b9d5ebcc45d8493b282`.
The complete file manifest is in
`dist/runpod/results-20261019T002734Z/training__artifact_manifest.json`; large
weights were intentionally not copied to the local machine.

[W&B run 56i7klyd](https://wandb.ai/liquid-mcdm-lab/runs/56i7klyd) was logged
offline on the pod and explicitly synced after training. No credential was
printed or included in the source archive.

## Remote acceptance and training evidence

- Remote CPU acceptance passed **124 tests in 36.94 seconds**, with zero skips.
- Two 1,000-episode generations were byte-identical.
- The pinned checkpoint was
  `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432`.
- Runtime: Python 3.12.3, Torch 2.9.1+cu128, Transformers 5.19.0,
  TRL 1.15.0, PEFT 0.21.2, CUDA 12.8, BF16.
- The real collated batch supervised exactly the answer alias plus EOS.
- Compatibility loss was finite at 0.91372; 166 of 332 trainable LoRA tensors
  received nonzero finite gradients. The vision tower had zero trainable
  tensors.
- The run completed all 25 optimizer steps in 40.62 seconds with training loss
  0.72068. Peak compatibility-check allocation/reservation was approximately
  6.63/14.24 GB.

The intentionally tiny 64-example training subset improved from 0.28125 to
0.671875 accuracy, and winner NLL fell from 1.57356 to 0.99330. On the
16-example held-out diagnostic, accuracy moved from 0.375 to 0.500. These
numbers demonstrate that the bounded optimization changed the model in the
intended direction; they do not establish generalization.

The custom d1 wrapper and standard image-text loader agreed on the argmax for
7 of 8 base examples; maximum probability delta was 0.01562. After saving and
reloading the adapter, all 4 checked argmax decisions matched, but maximum
probability delta was 0.03060. That exceeds the deliberately strict 1e-4
serialization threshold, so `run.json` correctly records the final validation
as failed even though training completed and the adapter was saved. Exact
probability reload parity remains the next issue to isolate before scaling.

## Compatibility findings

The official notebook's open-ended `torchao>0.16` resolved to 0.18.0, which
failed against Torch 2.9.1. Torchao 0.15.0 imported with that Torch build, but
PEFT 0.21.2 rejected it during LoRA injection because PEFT requires 0.16 or
newer. The successful non-quantized path omits optional torchao. TRL 1.15 also
uses fractional `warmup_steps` rather than `warmup_ratio`; the training entry
point now maps the configured 0.05 fraction to that API.

## Historical execution routes

The initial local CPU suite passed 73 tests before the user prohibited local
workloads. Colab then failed before dependency installation because its runtime
did not expose `nvidia-smi`. A Hugging Face Jobs workflow was prepared but not
submitted. The user subsequently authorized Runpod, which supplied the final
remote CPU acceptance and GPU training evidence above.
