# Run 20261019T002734Z

First bounded d1-3B MCDM LoRA run on a Runpod NVIDIA A40.

- 64 unique-winner training episodes, 25 optimizer steps.
- Base revision: `051bcc464b01b9f92942b364d9586b0ef5912432`.
- W&B: https://wandb.ai/liquid-mcdm-lab/runs/56i7klyd
- Training accuracy: 0.28125 to 0.671875.
- Held-out diagnostic accuracy: 0.375 to 0.500 on 16 episodes.
- Adapter reload argmax agreement: 1.0 on four episodes.
- Maximum reload probability delta: 0.030595, above the strict 1e-4 gate.

`training__run.json` is the unmodified terminal receipt and therefore records
`status=failed` for the reload probability gate even though all 25 training
steps completed and the adapter was saved. `summary.json` separates completed
training from the unresolved probability-parity issue.

The adapter weights are not stored in GitHub. Their SHA-256 and remote file
manifest are in `training__artifact_manifest.json`.
