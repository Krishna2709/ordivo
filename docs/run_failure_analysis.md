# Run failure analysis

## Attempt inventory

| Attempt | Stage reached | Failure | Root cause | Resolution |
|---|---|---|---|---|
| `20261018T235414Z` | Remote CPU acceptance | One reproducibility test rejected a COPRAS metadata value differing by about `3e-17` after alternative reordering | Metadata comparison incorrectly required bit-identical derived floats while the actual decision result was invariant | Preserve exact winner/ranking checks and compare derived numeric metadata with a documented floating tolerance |
| `20261019T000332Z` | Training environment import | `torchao==0.18.0` imported against missing PyTorch APIs | Official notebook's open `torchao>0.16` constraint resolved beyond the Runpod image's Torch 2.9.1 compatibility | Preserve the image's tested CUDA/Torch stack; do not install optional torchao for this non-quantized path |
| `20261019T001641Z` | CPU acceptance | Cancelled | Redundant retry after the earlier CPU suite had already passed | Reuse the validated environment and immutable dataset receipt |
| `20261019T002233Z` | LoRA injection | PEFT rejected torchao 0.15.0 | Torchao 0.15 matched Torch 2.9.1, but PEFT 0.21 requires torchao 0.16+ whenever it is installed | Remove optional torchao entirely |
| `20261019T002516Z` | SFT configuration | `warmup_ratio` absent from TRL 1.15 `SFTConfig` | Transformers 5 consolidated fractional warmup into `warmup_steps`, where values below one represent a ratio | Map the configured 0.05 fraction to `warmup_steps` |
| `20261019T002734Z` | Post-training adapter reload | Argmax agreement 1.0, maximum probability delta 0.03060 exceeded 1e-4 | Training and save completed; exact source of probability variance is unresolved. Cache-mode alone did not explain it | In v2, compare repeated inference, two independent fresh reloads, and saved-vs-reloaded decisions separately; preserve strict numerical evidence without treating decision-equivalent reload as exact probability equality |
| `20261019T004757Z` | V2 preflight after CPU acceptance | Requested 128 unique-winner test episodes but only 87 existed | The 100-record test split includes 13 nonunique outcomes that are correctly ineligible for single-choice accuracy | Evaluate all 87 eligible episodes; training and model loading never started for the failed attempt |

## Accuracy diagnosis

The first successful optimization used 64 examples and 25 optimizer steps, or
100 effective training examples with gradient accumulation four. It completed
about 1.56 passes over the tiny subset. Accuracy improved strongly on that
subset but reached only 0.50 on a 16-example held-out diagnostic. The held-out
denominator is too small for stable conclusions, and per-method slices contain
as few as two examples.

Release run `20261019T005111Z` used 512 unique training episodes and 400
optimizer steps while retaining the proven rank-16 LoRA scope and 2e-4 cosine
schedule. Training accuracy rose from 0.3242 to 0.9355. More importantly,
accuracy across every eligible disjoint-family test episode rose from 0.2874
to 0.5862, while the original 16-example comparison slice rose from 0.375 to
0.750. The broad result improved for every method. Near-tie accuracy remained
the hardest slice at 0.4167, and test ECE worsened from 0.1183 to 0.2394, so
the next accuracy work should target hard-case data and calibration rather
than simply extending this run.

The reload investigation separated three effects. Repeated inference on the
same loaded adapter was identical, and two independent fresh reloads were also
identical. Only the in-memory training model versus the saved/reloaded adapter
showed BF16 probability drift: maximum 0.02147 and mean 0.00111, with 16/16
decisions preserved. This supports a 0.05 probability tolerance for
decision-equivalent BF16 serialization while continuing to report the numeric
drift rather than calling it exact parity.

## W&B coverage

Run `56i7klyd` contains the complete TRL step-level training curve and runtime
telemetry. Post-training JSON evaluations and the adapter manifest were saved
on the Runpod volume and copied into the repository, but were not all logged as
W&B Tables or Artifacts. Release run `4vblgw80` adds explicit logging for
dataset identity, base/post metrics, all 87 before/after probability rows,
parity diagnostics, environment freeze, configuration, and the complete LoRA
adapter. Its final receipt is attached after offline sync so W&B and the
published Hugging Face files share the same terminal evidence.
