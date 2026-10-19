# ADR 0006: single-node Runpod execution with fail-closed training gates

Status: accepted for the first authorized GPU experiment. Date: 2026-10-18.

The managed Colab and Hugging Face Jobs paths did not provide a dependable
execution route. The user authorized Runpod. A single secure-cloud A40 48 GB is
selected because the 3B-parameter, non-quantized LoRA recipe is a single-device
workload. A multi-node cluster would add orchestration and synchronization
failure modes without serving this experiment.

The remote run has two environments. The first installs the hash-locked CPU
requirements and must pass deterministic generation, tokenizer boundary,
solver, W&B-offline, and zero-skip test checks. The second reuses the pod
image's CUDA-matched PyTorch and installs the candidate Liquid/TRL stack. It
must prove the actual GPU, exact LoRA target scope, TRL completion mask, finite
loss, and nonzero finite LoRA gradients before training begins.

The experiment remains the brief's 64-example, 25-step overfit check. It saves
only a PEFT adapter and reproducibility evidence. Zero-shot and post-training
metrics are diagnostic for this tiny split and cannot establish generalization.
W&B remains offline so no account secret is needed on the pod. Model access is
public, so the Hugging Face token is not transferred either.

The first live compatibility resolution selected `torchao==0.18.0` beside the
image's `torch==2.9.1+cu128`; importing Transformers failed because that
torchao release requires newer PyTorch APIs. `torchao==0.15.0` imported with
PyTorch 2.9.1, but PEFT 0.21.2 rejects versions below 0.16 during adapter
injection. The run does not use quantization, so the final environment omits
the optional torchao package and preserves the image's CUDA/PyTorch stack.
TRL 1.15 also exposes a fractional warmup through `warmup_steps`, not
`warmup_ratio`; the training entry point maps the configured 0.05 fraction to
that API.

The controller starts a detached local standard-library reaper with the pod ID
and a two-hour deadline; it reads the API key from `.env` only at the deadline
and deletes compute if the interactive controller disappears. This is a cost
backstop while the workstation remains awake and online; current Runpod REST
v1 does not expose a provider-side termination deadline. The active controller still
deletes the pod immediately after a terminal run. Results
and caches are on a separate 20 GB network volume. That volume is intentionally
preserved after compute cleanup so an interrupted controller cannot destroy
the only adapter. Base/post metrics on the 64 training examples record whether
the bounded run actually improved rather than equating adapter reload with
overfitting. The volume continues incurring storage charges until explicitly
deleted.

The authorized run `20261019T002734Z` completed 25 optimizer steps on an NVIDIA
A40 in 40.62 seconds. Training-subset accuracy increased from 0.28125 to
0.671875 and winner NLL decreased from 1.57356 to 0.99330. The adapter was
saved and hashed on network volume `ngica7qkhd`; W&B run `56i7klyd` was synced.
Reload preserved all four checked argmax decisions, but probability deltas
reached 0.03060, above the strict 1e-4 parity threshold. The run therefore
proves completed LoRA training and a loadable decision-equivalent adapter on
that sample, while exact probability serialization parity remains unresolved.
The paid pod was deleted and verified absent; the volume was preserved.

