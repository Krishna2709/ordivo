# Announcement draft: Liquid MCDM d1-3B LoRA

We are releasing Liquid MCDM Lab: a reproducible experiment that teaches
LiquidAI/d1-3B to imitate four exact multi-criteria decision methods—COPRAS,
WASPAS, VIKOR, and TOPSIS—from structured decision episodes.

The release includes:

- audited float64 solver implementations and 124 remote acceptance tests;
- a deterministic synthetic dataset with exact teacher provenance;
- a PEFT LoRA adapter based on a pinned LiquidAI/d1-3B revision;
- before/after evaluations, per-method slices, failure analysis, W&B telemetry,
  and a standalone interactive experiment report;
- Runpod, Colab, and Hugging Face execution material with no embedded secrets.

The first 25-step diagnostic raised accuracy from 28.1% to 67.2% on its
64-example training subset and from 37.5% to 50.0% on a 16-example held-out
slice. These are algorithm-imitation results on synthetic data, not evidence
of human preference modeling or real-world decision quality.

Links:

- Code: https://github.com/Krishna2709/liquid-mcdm-lab
- Dataset: https://huggingface.co/datasets/blackOrder/liquid-mcdm-synthetic
- Model: https://huggingface.co/blackOrder/d1-3b-mcdm-lora
- W&B project: https://wandb.ai/liquid-mcdm-lab
- Experiment report: `reports/experiment_report.html`
