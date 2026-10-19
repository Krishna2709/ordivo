# Model card draft — bounded d1-3B LoRA adapter

Base candidate: LiquidAI/d1-3B, revision 051bcc464b01b9f92942b364d9586b0ef5912432.
Task: single-choice imitation of specified MCDM teachers on synthetic positive-valued matrices.
Status: run `20261019T002734Z` completed 25 LoRA steps on 64 synthetic
examples using an NVIDIA A40. The adapter is preserved on Runpod network
volume `ngica7qkhd`; its weights SHA-256 is
`ae1ba85962d6f4c87f14b055c6290ef142a25d6794207b9d5ebcc45d8493b282`.

Training: official Liquid TRL/PEFT text-only prompt/completion recipe for
LFM2-VL; completion-only CE on answer alias and EOS; no quantization. The
training subset moved from 0.28125 to 0.671875 accuracy and held-out accuracy
moved from 0.375 to 0.500 on 16 examples. Restricted option probabilities use
upstream max-pooled alias groups. This is winner classification, not full
ranking.

Limitations: synthetic labels are conditional on method, normalization,
weights and candidate set. Do not infer human preferences, procurement
suitability, calibration or arithmetic superiority over exact solvers. VIKOR
nonunique compromises/ties are excluded from winner training. The adapter
reload preserved all four checked argmax decisions but had maximum probability
delta 0.03060, so exact serialization parity is unresolved. License: base LFM
Open License v1.0; review its conditions and preserve notices before adapter
distribution. External datasets require separate eligibility review.
