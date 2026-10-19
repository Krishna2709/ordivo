# M0 research and feasibility findings

Retrieved **2026-10-18**, using primary model repositories, exact official notebook revisions, package metadata and provider documentation. Full evidence: [model/notebook audit](model_research.md), [mathematics, datasets, Colab and W&B audit](data_colab_research.md), and [source hashes](../research/upstream_models.json).

## Decision

Start with **official Liquid TRL + PEFT LoRA**, preserving d1-3B's answer-token mechanism. Use `AutoModelForImageTextToText` to train the standard LFM2-VL model and the custom `AutoModel` wrapper for inference. Do not substitute a generic classification head or assume that a causal-LM loader accepts the checkpoint. [ADR 0001](decisions/0001-training-boundary.md) records alternatives and deviations.

**Execution constraint, updated during implementation:** tests, generation,
dependency installation, model downloads and training run on remote provider
resources. Colab failed to provide a dependable execution route, so the first
authorized experiment ran on a Runpod A40. Local work remained limited to
source, control-plane actions and small evidence files.

## Checkpoints

| Checkpoint and revision | Architecture and loader | Decision/training semantics | Status |
|---|---|---|---|
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B/tree/051bcc464b01b9f92942b364d9586b0ef5912432), `051bcc464b01b9f92942b364d9586b0ef5912432` | ~3.12B LFM2.5-VL; hybrid convolution/attention text backbone plus SigLIP2 vision; standard training class `AutoModelForImageTextToText`, custom inference `AutoModel(trust_remote_code=True)` | One choice question over state/options. Pinned `prompt.py` renders prompt; completion is answer alias + EOS. Full-vocabulary CE only on completion; inference max-pools alias/leading-space token forms, then softmaxes over options. | First 64-example/25-step LoRA run completed on Runpod A40 |
| [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M/tree/02b55d7076f15129e59ab3f94783f32c4b088674), `02b55d7076f15129e59ab3f94783f32c4b088674` | ~587M, shared bidirectional trunk and custom decision head plus vision/audio encoders; custom AutoModel | Marker-position hidden states, attention head and shared scorer with option masks/softmax. Public inference is no-grad and converts results to lists; no verified supervised loss contract. | Research/inference-only; d1-3B SFT recipe is not interchangeable |
| [LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/tree/b886781f7c6f10ca9b7096e21b83e30a073c2f39), `b886781f7c6f10ca9b7096e21b83e30a073c2f39` | ~354.5M bidirectional LFM2 encoder/MLM, 16 layers, hidden 1024; custom AutoModel or AutoModelForMaskedLM | No pretrained d1 choice head. Proposed shared alternative encoder combines text embeddings, numeric values, directions and weights, with masked set attention and shared scalar scoring. | Custom head design only; frozen backbone/head-only first, then optional adaptation |

All three include **LFM Open License v1.0**, not an unconditional Apache/MIT license. Commercial revenue and redistribution conditions apply; see the [pinned license](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE) and detailed model audit. The encoder card's trained context is 8,192 despite a larger configuration position limit; do not claim 128k trained capability. d1-3B card specifies 32,768; omni 16,384 combined positions.

Weight-storage lower bounds alone are ~6.24 GB for d1-3B BF16, 1.174 GB for omni FP16, 1.418 GB for encoder FP32. They exclude activation/gradient/optimizer/kernel overhead and are not measured VRAM requirements. The measured d1-3B compatibility pass used about 6.63 GB allocated and 14.24 GB reserved on an A40; this is evidence for the bounded recipe, not a guarantee for other runtimes.

## Official notebooks and compatibility matrix

Both official notebooks were inspected at cookbook commit [`744c5b628a27adb4f219ff4a1d11e3d70259b369`](https://github.com/Liquid4All/cookbook/tree/744c5b628a27adb4f219ff4a1d11e3d70259b369).

| Layer | Evidence / version | Verification boundary |
|---|---|---|
| CPU domain stack | Python 3.12.11, NumPy 2.2.6, pytest 8.4.2; `uv.lock` and `requirements-cpu.lock` | Prior local tests passed before Colab-only request; final source requires remote rerun |
| W&B | 0.22.3 offline SDK run with safe artifacts | Prior genuine offline run; no cloud run URL or sync |
| Tokenizer and model | Transformers 5.19.0, tokenizers 0.23.3, pinned d1 revision | Tokenization, custom/standard inference, real collator labels, BF16 forward/backward and adapter reload inference executed remotely |
| [TRL notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb) | Transformers 5.19.0, TRL 1.15.0, PEFT 0.21.2; Python 3.12.3, Torch 2.9.1+cu128 on the measured pod | Imports, real collator mask, BF16 forward/backward, LoRA gradients and 25 training steps verified. Optional torchao omitted after incompatibility findings |
| [Liquid Unsloth notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_unsloth.ipynb) | FastVisionModel + standard image-text loader; Transformers 5.17.0, no 4-bit in inspected recipe | Installer pins datasets 4.3.0 but TRL 1.15 requires >=4.7; `--no-deps` hides incompatibility. Deferred |
| [Generic Unsloth guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) | FastDecisionModel/DecisionTrainer, state/questions/gold, custom decision head | Different head/format; not a drop-in d1 adaptation |
| Torch/CUDA/torchvision | Preserve a compatible measured Colab runtime combination | Not installed/tested locally; no QLoRA/Flash Attention claim |

Official TRL LoRA targets q/k/v/out/in_proj and w1/w2/w3, excludes vision tower, rank16/alpha32/dropout0.05. The measured run used batch1, gradient accumulation4, no packing, max length2048, BF16, checkpointing, 64 examples and 25 steps. TRL 1.15 accepts the 0.05 warmup fraction through `warmup_steps`. The real batch contained exactly alias plus EOS supervision; 166 trainable tensors had nonzero finite gradients.

The pinned prompt helper is bundled unchanged with its license. Labels A–Z and larger-set aliases must be tokenizer-verified; inference uses max over token groups including valid leading-space variants. Non-answer prompt/padding labels must be -100; EOS remains supervised. A reference mask check is not proof that the real TRL collator agrees. Custom inference must load a revision-pinned **local snapshot** because its internal tokenizer loader otherwise can resolve main.

## MCDM, data and experiment design

Chosen normalizations: COPRAS column sums; WASPAS max-benefit/min-cost ratios; VIKOR range regret; TOPSIS vector norms. [Mathematical policy](mathematics.md) defines domains, ties and degeneracies. VIKOR score minimization and acceptable compromise are separate. Strictly positive COPRAS/WASPAS inputs avoid undefined ratios/products. No blanket invariance to adding alternatives.

Installed pymcdm 1.3.1 COPRAS failed a hand cost-dominance case; its source omits reductions in the proportional cost expression. Use independently audited pyDecision instead for COPRAS; retain matched-convention pymcdm checks for other methods. See [ADR 0002](decisions/0002-mathematics-and-oracles.md). Test rank reversal rather than assuming it away.

Synthetic stage zero is unblocked: 1,000 episodes, exact teacher provenance, full precision, fixed disjoint scenario families, explicit seed lineage and hashes. Final review corrected accidental method/family/weighting confounding by factorial cycling; previous generated data were removed and must be regenerated remotely. [Dataset card](dataset_card.md) records limitations: no size-extrapolation claim, no human-preference ground truth.

| Dataset | Verified primary evidence | Decision |
|---|---|---|
| [UCI Car](https://archive.ics.uci.edu/dataset/19/car+evaluation) | 1,728 examples, six categorical attributes, hierarchical classification labels, CC BY 4.0 | Future classification sanity task, not hotel rankings |
| [Expedia](https://www.kaggle.com/c/expedia-personalized-sort/data) | Sign-in/rule agreement required, competition terms and outside-data restriction; search IDs and click/booking outcomes, random-order indicator | Permission for intended synthetic-pretrained transfer remains unresolved. No download/agreement. Preserve complete search groups, separate outcomes, exclude leakage |
| [MSLR](https://www.microsoft.com/en-us/research/project/mslr/) | 136 features, qid grouping, relevance0–4, predefined folds; requires online agreement | Candidate fallback only after terms verified; web relevance differs from hotel behavior |

Randomized displayed order does not eliminate examination/position bias. Do not invent unverified amenities or uniform nightly price semantics. The [referenced MCDM/LLM paper](https://arxiv.org/html/2502.15778v1) motivates multi-attribute classification; it does not validate these four exact solvers, d1 training or hotel-choice transfer.

## Colab, credentials and tracking

[Google's official VS Code Colab extension](https://github.com/googlecolab/colab-vscode) supports Select Kernel → Colab → Auto Connect and managed notebook kernels via OAuth. Publisher, source version, scopes and limitations are documented in the companion audit. A connected kernel does not automatically synchronize local source or expose execution to this agent. [Colab FAQ](https://research.google.com/colaboratory/faq.html) says allocations/limits vary; high RAM is host RAM; VMs are ephemeral. Measure nvidia-smi and persist checkpoints/config/manifest/environment outside the VM. Do not expose Jupyter tokens or tunnels.

Source archives exclude `.env`. The GPU run logged to W&B offline first and was
then explicitly synced as [run 56i7klyd](https://wandb.ai/liquid-mcdm-lab/runs/56i7klyd)
using the configured key without printing or bundling it. W&B telemetry builds
sanitized summaries rather than uploading arbitrary input JSON.

## Acceptance status

M0 research and the M1/M2 implementation are complete. Runpod CPU acceptance
passed 124 tests with zero skips and reproduced the dataset byte-for-byte. Run
`20261019T002734Z` completed the first M3 overfit experiment: train accuracy
0.28125 to 0.671875, train winner NLL 1.57356 to 0.99330, and held-out
accuracy 0.375 to 0.500 on only 16 examples. Custom/standard base readout had
0.875 argmax agreement over eight examples. The saved 115,818,206-byte adapter
preserved all four checked reload argmax decisions, but maximum probability
delta was 0.03060, failing the strict 1e-4 parity gate. This is evidence of a
completed bounded fine-tune, not generalization or exact probability reload
parity. The paid pod was deleted; network volume `ngica7qkhd` remains.
