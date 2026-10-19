# Release run `20261019T005111Z`

This directory contains the small, reviewable receipts from the remote A40
release run. Large dataset files and adapter weights were transferred directly
from Runpod to Hugging Face and were never copied to the local machine.

- CPU acceptance: 130 tests passed in 31.45 seconds.
- Training: 512 examples, 400 optimizer steps, rank-16 LoRA, BF16, no
  quantization.
- Full eligible test: 0.2874 base accuracy to 0.5862 adapted accuracy over 87
  episodes.
- Original comparison slice: 0.375 base to 0.750 adapted over 16 episodes.
- Reload: 16/16 decisions preserved; maximum saved-versus-reloaded probability
  delta 0.02147.
- W&B: https://wandb.ai/black-order/liquid-mcdm-lab/runs/4vblgw80
- Dataset: https://huggingface.co/datasets/blackOrder/liquid-mcdm-synthetic
- Model: https://huggingface.co/blackOrder/d1-3b-mcdm-lora

`hf_publish_receipt.json` records every published dataset and adapter payload
hash plus the successful Hub read-back checks. `training/run.json` is the
terminal run state; the remaining JSON files contain complete before/after,
slice, prediction, and reload evidence.
