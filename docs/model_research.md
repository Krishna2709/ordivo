# Liquid model and training feasibility audit

Retrieved 2026-10-18. This audit inspected public metadata, model code/configuration, two official notebook sources, package metadata, and the Unsloth guide. No weights were downloaded, no upstream Python was executed, no GPU packages were installed, and no model training/inference was performed. Source inspection establishes an implementation candidate, not a tested Colab environment.

## Recommendation and source identity

Start with the official **TRL + PEFT LoRA** recipe for `LiquidAI/d1-3B`: standard `AutoModelForImageTextToText` for training; custom `AutoModel(..., trust_remote_code=True)` for System One evaluation. Retain its pretrained answer-token readout. Use the official Unsloth `FastVisionModel` route only as a subsequent optimization. The generic Unsloth `FastDecisionModel` guide adds a different decision head and is not a drop-in recipe for preserving d1-3B. The current official notebooks already distinguish these routes.

| Source | Exact revision retrieved |
|---|---|
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B/tree/051bcc464b01b9f92942b364d9586b0ef5912432) | `051bcc464b01b9f92942b364d9586b0ef5912432` |
| [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M/tree/02b55d7076f15129e59ab3f94783f32c4b088674) | `02b55d7076f15129e59ab3f94783f32c4b088674` |
| [LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/tree/b886781f7c6f10ca9b7096e21b83e30a073c2f39) | `b886781f7c6f10ca9b7096e21b83e30a073c2f39` |
| [Liquid4All/cookbook](https://github.com/Liquid4All/cookbook/tree/744c5b628a27adb4f219ff4a1d11e3d70259b369) | `744c5b628a27adb4f219ff4a1d11e3d70259b369` |

[research/upstream_models.json](../research/upstream_models.json) records per-file URLs, byte sizes and SHA-256 hashes, package metadata, and tokenizer vocabulary evidence. Small original source/config/card/license snapshots are under `research/upstream/`. They are research evidence, not imported by the CPU stack. The notebook copies are upstream material; no root cookbook license was present in the pinned GitHub directory listing, so do not assume a blanket permissive license for redistributing the cookbook.

## d1-3B architecture, interface and exact answer semantics

The [card](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/README.md) describes approximately 3.12B parameters, LFM2.5-VL-3B ancestry, SigLIP2 NaFlex vision tower, 128,000 vocabulary, and 32,768 context. The [configuration](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/config.json) is `lfm2_vl`, architecture `Lfm2VlForConditionalGeneration`, with 30 text layers, hidden size 2048, 32 attention heads/8 KV heads, interleaved short convolutions and attention. `auto_map` registers only `AutoModel: modeling_d1.D1Model` for the custom API. The card's supported inference floor is Transformers 5.14; the notebook pins a newer version.

The [custom model](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/modeling_d1.py) subclasses an optimized LFM2-VL implementation and exposes `system_one(state, questions)` and batched inference. The training notebook instead loads the same weights through Transformers' standard `AutoModelForImageTextToText`. Using `AutoModelForCausalLM`, a generic classification head, or the custom inference API as a training loop would deviate from the verified recipe.

All details below come from pinned [prompt.py](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/prompt.py), [runner.py](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/runner.py), and the tokenizer JSON:

- Input is a state plus named questions. A winner task is one `choice` question with instructions and insertion-ordered `{alternative_id: description}` criteria. Keep solver targets outside the state.
- `render(tokenizer, state, as_question(question), bos=tokenizer.bos_token)` builds the exact prompt. Default is no system turn, JSON with indentation for object states, and an assistant header with no reasoning prefix. Generic chat-template application alone does not add the required state/question/options layout.
- Native single-letter alphabetic labels retain their labels as codes. Otherwise up to 26 options use `A` through `Z`, larger option sets use `00`, `01`, and onward. `aliases()` checks single-token encoding and distinct IDs, falling back through a fixed pool when necessary. Never assume all arbitrary option IDs are tokens or that 27 options use the same coding scheme as 26.
- At the final prompt position, each choice option has a token **group**: its alias token plus a leading-space alias when that variant is one token. The readout takes the maximum logit/log-probability within each group, then softmaxes over option groups. It is not the full-vocabulary softmax probability of a word, and selecting only the unspaced alias silently changes upstream scoring.
- Vocabulary evidence: `A=41`, `B=42`, `C=43`, `Z=66`, `00=555`, `01=4040`, `26=1616`, `76=9024`; space-prefixed `A=334`, `B=378`. BOS `<|startoftext|>=124894`, `<|im_start|>=124899`, EOS `<|im_end|>=124900`, PAD `<|pad|>=124893`. These are direct vocabulary lookups; an authorized tokenizer preflight must additionally verify actual tokenization, especially concatenated completion boundaries.
- `noul` pools single-token yes/no case variants; `score` uses digits 0 through level-count minus one, intended for 2–10 levels. Winner prediction only needs `choice`.

Template shape, with literal newlines represented below:

```text
<BOS><|im_start|>user
<state rendered by state_block>


QUESTION:
<instructions>

Options:
A <first description>
B <second description>

Reply with the option code only.<|im_end|>
<|im_start|>assistant
```

Use the source renderer, not a transcription of this illustration. One training row has `prompt` equal to the complete rendered string and `completion` equal to the correct **current option alias** plus EOS. The next-token logits at the final prompt token predict the alias. In causal teacher forcing, the subsequent position predicts EOS.

## Official TRL recipe and candidate environment

The pinned [TRL notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb) has:

- `transformers==5.19.0`, `trl==1.15.0`, `peft==0.21.2`, `torchao>0.16.0`, and unpinned `datasets`.
- BF16 if the GPU supports it natively, otherwise FP16; the notebook uses `torch.cuda.is_bf16_supported(including_emulation=False)`.
- Standard `AutoModelForImageTextToText.from_pretrained(..., dtype=dtype, device_map="auto")`.
- PEFT causal-LM LoRA: rank 16, alpha 32, dropout 0.05, no bias; target `q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3`; explicitly exclude `.*vision_tower.*`. Confirm actual trainable names and frozen vision weights after wrapping.
- TRL `SFTTrainer(..., processing_class=tokenizer)` with a plain string prompt/completion dataset; batch size 4, gradient accumulation 4, gradient checkpointing, 1 epoch, LR 2e-4, cosine scheduler, and `warmup_steps=0.05` in this notebook version. These are upstream example settings, not a measured allocation for our scenarios.
- Training/evaluation datasets are split by semantic cases before rendering. Option subsets/order vary by example. For MCDM, permutation augmentation must retain all alternatives: arbitrary subsets change the exact solver task and require recomputing labels.

Prompt/completion SFT masks prompt positions. [TRL v1.15.0 source](https://github.com/huggingface/trl/blob/v1.15.0/trl/trainer/sft_trainer.py) chooses completion-only loss for this schema, constructs a zero/one completion mask, and assigns `-100` to masked labels. Set `completion_only_loss=True` explicitly and inspect a real collated batch: precisely alias and EOS remain supervised; prompt and padding are ignored. This is standard **full-vocabulary causal cross-entropy over the completion positions**, not restricted-option cross-entropy. Restricted option scoring occurs at inference. Do not claim every non-answer token is excluded: EOS is deliberately trained too.

| Component | Candidate constraint and evidence | Validation remaining |
|---|---|---|
| Python | >=3.11, required by [TRL 1.15 metadata](https://pypi.org/pypi/trl/1.15.0/json) | Record actual Colab version |
| Transformers | 5.19.0, official TRL notebook | Import, loader and forward parity |
| TRL / PEFT | 1.15.0 / 0.21.2, official notebook | Collator masks, gradients and checkpoint round trip |
| Datasets / Accelerate | >=4.7.0 / >=1.4.0, TRL package metadata | Resolve and freeze exact versions in Colab |
| Hugging Face Hub / tokenizers | >=1.31,<3 / >=0.23.1,<0.24 from [Transformers metadata](https://pypi.org/pypi/transformers/5.19.0/json) | Record resolved versions |
| PyTorch / CUDA / torchvision | Use a mutually compatible Colab runtime stack; Transformers torch extra requires >=2.5 | Measure GPU/VRAM and record build/runtime versions |
| torchao | >0.16.0 in notebook | Resolve against PyTorch; no QLoRA assertion |
| bitsandbytes / Unsloth / Flash Attention | Optional, absent from first TRL path | Separate compatibility experiment |

These are verified upstream pins/metadata, **not a tested lockfile**. Do not replace Colab's Torch/CUDA pair blindly. Resolve dependencies normally, run `pip check`, restart if needed, then freeze the proven environment. Use `report_to="wandb"` or offline W&B rather than inheriting the notebook's tracking disablement. Save/resume checkpoints during our runs rather than inheriting `save_strategy="no"`.

Pin all model/tokenizer/code loads to the listed revision. An important custom-loader subtlety: `D1Model.engine` loads its tokenizer from `self.name_or_path` without passing revision. Load custom inference from a local **revision-pinned snapshot directory** so that internal lookup cannot silently resolve current `main`. Review downloaded code before executing it. Adapter export requires testing reload against that same base. The notebook's merged export also copies custom Python modules, processor configuration and license; verify auto-map and tokenizer identity on reload.

## Unsloth comparison and discovered blocker

The pinned [Liquid Unsloth notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_unsloth.ipynb) imports Unsloth first, evaluates through custom AutoModel, and trains with `FastVisionModel.from_pretrained(..., auto_model=AutoModelForImageTextToText, load_in_4bit=False, max_seq_length=2048)`. Its LoRA regex restricts adaptation to language-model attention, convolution and MLP modules; it uses response-only masking over text rows, leaving alias plus EOS. This preserves the d1 readout.

A concrete dependency inconsistency exists: its Colab installer pins `datasets==4.3.0`, then installs `trl==1.15.0` with `--no-deps`; that TRL release requires `datasets>=4.7.0`. It also pins Transformers 5.17.0, unlike the TRL notebook's 5.19.0. Do not reproduce this inconsistent environment or combine installer fragments blindly. Resolve it in an isolated optional experiment and establish parity before claiming an optimized training path.

The [generic Unsloth guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) documents `FastDecisionModel`/`DecisionTrainer`, `state/questions/gold` datasets and a Clef-style head over option representations. Liquid's notebook explicitly explains that applying that route to d1-3B creates a new random decision head and saves a different model format. Its example memory/speed figures concern other checkpoints; they do not establish d1-3B memory requirements or QLoRA compatibility.

## Secondary models

| Property | d1-omni-600M | LFM2.5-Encoder-350M |
|---|---|---|
| Architecture | 587M total: 381M shared trunk/head, 94M vision, 112M audio; custom bidirectional encoder plus decision head | ~354.5M bidirectional LFM2 encoder/MLM, 16 layers, hidden size 1024 |
| Context | 16,384 combined positions; image text capped at 896 | Card trained/support context 8,192; config's 128,000 positional limit is not evidence of 128k trained capability |
| Loader | `AutoModel`, custom `D1OmniConfig`/`D1OmniModel`, `trust_remote_code=True` | `AutoModel` encoder body or `AutoModelForMaskedLM`, both custom code |
| Dependencies | Card `transformers>=5.15`, torch/torchvision/Pillow/soundfile | Card latest Transformers; config records 5.12.0; actual compatibility still untested |
| Decision semantics | Gather option-marker hidden states after a two-layer Transformer head, shared MLP scorer, mask absent options, softmax; text temperature by question type/count | No pretrained d1 decision head; MLM vocabulary predictions and backbone hidden states |
| Precision | Card prefers float32 CPU / float16 GPU, warns BF16 can change top answers | Config float32; reduced precision requires validation |
| First scope | Inference-only feasibility | Proposed custom numeric/text set head, not yet trained |

Evidence: pinned [omni model](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/modeling_d1.py), [omni encoder/head](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/encoder.py), [omni prompt](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/prompt.py), [encoder card](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/blob/b886781f7c6f10ca9b7096e21b83e30a073c2f39/README.md), and [encoder code](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/blob/b886781f7c6f10ca9b7096e21b83e30a073c2f39/modeling_lfm2_bidirectional.py).

Omni weights and head implementation are public. However, its public probability method is `no_grad`, converts softmax results to Python lists, and exposes no ordinary supervised training `forward(labels=...)` contract or matching official SFT example in the inspected sources. Training would require a separately designed differentiable forward/loss and proven padding/marker/calibration handling. The d1-3B answer-token notebook cannot be reused unchanged. Temperature constants learned upstream are not guaranteed calibrated after adaptation.

For encoder-350M, propose encoding criterion descriptions/context and each alternative's text independently with shared weights; combine pooled embeddings with normalized numeric values, direction and weight features in a shared alternative encoder; use masked set attention without alternative-position embeddings and a shared scalar scorer. Permuting alternatives must permute output scores. Include candidate masks in attention and loss. A serialized whole-episode encoder is a useful comparison, but is not automatically permutation-equivariant. Start frozen-backbone/head-only, compare the numerical Set Transformer, and only then test partial unfreeze/LoRA. This is a design proposal, not upstream supplied ranking functionality.

## License, memory and go/no-go gates

All three pinned model repositories include **LFM Open License v1.0**. Its grant covers use/modification/distribution subject to conditions; commercial use has an annual-revenue threshold of USD 10 million and qualified nonprofit research provisions. Redistribution requires the license, retained notices, and prominent modification notices. Do not describe these weights as Apache/MIT or unconditionally commercial-use permissive. See the actual [d1 license](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE), [omni license](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/LICENSE), and [encoder license](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/blob/b886781f7c6f10ca9b7096e21b83e30a073c2f39/LICENSE) before distribution or commercial deployment.

Arithmetic lower bounds for weights alone: d1-3B BF16 approximately 6.24 GB; omni FP16 1.174 GB; encoder FP32 1.418 GB (decimal units). These exclude activations, gradients, optimizer, temporary tensors, vision components' runtime work, and allocator overhead. Both official fine-tuning notebooks say a T4 is sufficient for their example; this is an upstream claim, not a measured guarantee for MCDM prompts. Start with batch size 1 and bounded context after measuring allocation; prefer A100 BF16 where available, assess L4/T4 with actual VRAM and FP16 behavior. No model performance, throughput, cost or calibration has been measured here.

Before approving a first GPU overfit run, require: dependency/import checks, frozen revision/local snapshot, measured GPU/VRAM, tokenizer and alias-group parity, a collated-batch loss-mask assertion, standard-vs-custom readout comparison, zero-shot held-out evaluation, correct LoRA trainable names, a 50–200 example bounded training budget, adapter save/reload parity, and W&B offline/online policy chosen by the user. Then update the compatibility matrix with actual evidence rather than upgrading candidate pins to tested claims.
