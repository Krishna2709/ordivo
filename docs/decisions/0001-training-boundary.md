# ADR 0001: preserve d1 answer-token training with official TRL

Status: accepted for initial implementation; GPU compatibility pending. Date: 2026-10-18.

The brief prioritizes d1-3B while requiring primary-source verification. Compared: generic causal-LM/classifier training, official Liquid TRL, and official Liquid Unsloth. Select TRL + PEFT with `AutoModelForImageTextToText`; use the pinned custom inference wrapper only for evaluation. The standard model and wrapper are distinct interfaces over the checkpoint. Preserve upstream prompt rendering and max-pooled alias/leading-space readout, and full-vocabulary completion CE on alias **and EOS**. Explicitly mask prompt/padding.

The generic Unsloth decision-head tutorial changes the head/format; the Liquid Unsloth notebook additionally contains a datasets/TRL dependency conflict. It remains an optional optimization after parity, not the initial installer. See [model audit](../model_research.md) for primary sources and revisions.

Reuse the audited upstream prompt helper unchanged, with original license and attribution, rather than manually approximating its tokenization rules. Tokenizer-only checks may run on CPU. A reference mask is not proof of a TRL collated mask; actual collator, forward/backward, LoRA target names and reload still require GPU preflight. Candidate dependencies are separate from the tested CPU lock.

Keep the configured warmup fraction at 0.05; TRL 1.15/Transformers 5 exposes
that fraction through `warmup_steps`, where values below one are interpreted as
a ratio. Start batch size one, disable packing, keep checkpoints, and use W&B.
These are project adaptations. No QLoRA claim.

Omni remains inference-only research. Encoder-350M requires a new set-aware head; neither is silently substituted for the primary model.
