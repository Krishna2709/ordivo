# Model card draft — bounded d1-3B LoRA adapter

Base candidate: LiquidAI/d1-3B, revision 051bcc464b01b9f92942b364d9586b0ef5912432.
Task: single-choice imitation of specified MCDM teachers on synthetic positive-valued matrices.
Status: release run `20261019T005111Z` completed 400 LoRA steps on 512
synthetic examples using an NVIDIA A40 and passed full evaluation and reload
acceptance. The adapter is published at `blackOrder/d1-3b-mcdm-lora`; its exact
file hashes are recorded in the Hugging Face publication receipt.

Training: official Liquid TRL/PEFT text-only prompt/completion recipe for
LFM2-VL; completion-only CE on answer alias and EOS; no quantization. The
training subset moved from 0.3242 to 0.9355 accuracy. Across all 87 eligible
test episodes, accuracy moved from 0.2874 to 0.5862; on the original 16-example
comparison slice it moved from 0.375 to 0.750. Restricted option probabilities
use upstream max-pooled alias groups. This is winner classification, not full
ranking.

Limitations: synthetic labels are conditional on method, normalization,
weights and candidate set. Do not infer human preferences, procurement
suitability or arithmetic superiority over exact solvers. Test ECE worsened
despite improved accuracy, so probabilities are not calibrated confidence. VIKOR
nonunique compromises/ties are excluded from winner training. The adapter
reload preserved all 16 checked argmax decisions; repeated inference and two
fresh reloads matched exactly. The in-memory-to-reload maximum probability
delta was 0.02147 in BF16. License: base LFM Open License v1.0; review its
conditions and preserve notices before adapter distribution. External datasets
require separate eligibility review.
