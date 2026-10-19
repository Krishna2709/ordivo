# Announcement draft: Ordivo v1 — Neural Decision Intelligence Model

We are releasing **Ordivo v1**, the first version of our Neural Decision
Intelligence Model. It is a reproducible experiment that teaches LiquidAI/d1-3B
to imitate four exact multi-criteria decision methods—COPRAS, WASPAS, VIKOR,
and TOPSIS—from structured decision episodes.

The release includes:

- audited float64 solver implementations and 130 remote acceptance tests;
- a deterministic synthetic dataset with exact teacher provenance;
- a PEFT LoRA adapter based on a pinned LiquidAI/d1-3B revision;
- before/after evaluations, per-method slices, failure analysis, W&B telemetry,
  and a standalone interactive experiment report;
- Runpod, Colab, and Hugging Face execution material with no embedded secrets.

The 400-step release run raised accuracy from 32.42% to 93.55% on its
512-example training subset and from 28.74% to 58.62% across all 87 eligible
disjoint-family test episodes. On the original 16-example comparison slice,
accuracy improved from 37.5% to 75.0%. Every one of 16 checked decisions was
preserved after adapter reload. These are algorithm-imitation results on
synthetic data, not evidence of human preference modeling or real-world
decision quality.

Links:

- Code: https://github.com/Krishna2709/ordivo-neural-decision-intelligence-model
- Dataset: https://huggingface.co/datasets/blackOrder/liquid-mcdm-synthetic
- Model: https://huggingface.co/blackOrder/Ordivo-v1
- W&B release run: https://wandb.ai/black-order/liquid-mcdm-lab/runs/4vblgw80
- Experiment report: `reports/experiment_report.html`
