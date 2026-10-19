# Runpod d1-3B gated overfit plan

**Goal:** Run the authorized 64-example, 25-step d1-3B LoRA experiment on one
remote NVIDIA GPU while keeping all installations, model weights, generated
data, tests, and training off the workstation.

**Architecture:** A local standard-library control client creates one private
Runpod network volume and one secure-cloud pod, records every provider resource
identifier before the next mutation, and transfers an allowlisted source ZIP
over SSH. A remote bootstrap creates separate CPU-acceptance and training
virtual environments on `/workspace`. Training is unreachable until the full
CPU acceptance suite and a real TRL/PEFT forward-backward compatibility gate
pass. Results and the adapter remain on the network volume; the pod is deleted
at terminal success or failure.

**Reviewed bounds:** one A40 48 GB, secure cloud, on-demand, 25 optimizer steps,
64 training examples, a detached local two-hour compute reaper, offline W&B. The expected compute
ceiling is 2 GPU-hours. The volume is 20 GB and continues billing until it is
explicitly deleted.

## Tasks

1. Extend the runtime guard with an explicit `RUNPOD_POD_ID` branch and an
   actual `nvidia-smi` query.
2. Add a training entry point which evaluates the pinned base checkpoint,
   applies the official Liquid LoRA target list, constructs an official TRL
   prompt/completion dataset, inspects the collated labels, runs a finite
   forward/backward pass, and only then calls `trainer.train()`.
3. Save adapter-only weights, tokenizer, dependency freeze, W&B offline run,
   per-episode predictions, metrics, and a reload parity report.
4. Add a remote bootstrap which checks source integrity, executes the full CPU
   acceptance pipeline in an isolated environment, resolves training
   dependencies without replacing the image CUDA/PyTorch stack, then runs the
   compatibility gate and bounded training command.
5. Add a Runpod control client with prepare, preflight, launch, status, logs,
   and cleanup actions. Never serialize API keys or send `.env`.
6. Run all tests and workloads on the remote pod. Stop before training on any
   failed acceptance, masking, LoRA-target, gradient, memory, or reload check.

