# Alternatives to Colab — 2026-10-18

Status: user selected Hugging Face Jobs. Implementation and runbook prepared; read-only identity/hardware preflight succeeded. No source uploaded, paid resource created or job launched. See [runbook](huggingface_jobs.md) and ADR0005. All workload execution must remain remote; the user has proposed RunPod or Hugging Face as alternatives to the earlier Colab-only restriction.

## Recommended first route: Hugging Face Jobs

[Jobs](https://huggingface.co/docs/hub/jobs-overview) runs custom Python scripts or Docker commands on selected remote hardware. It supports training, not merely hosted inference. [Configuration](https://huggingface.co/docs/hub/jobs-configuration) supports secret injection, timeouts and mounted inputs/outputs. This suits reproducible command-line acceptance runs without notebook-kernel attachment. A positive credit balance is required; possession of an API token does not establish credits or sufficient token permissions.

Current [published compute rates](https://huggingface.co/docs/hub/jobs-pricing), checked 2026-10-18:

| Hardware | GPU memory | Compute/hour | Arithmetic estimate for 10 billable minutes |
|---|---:|---:|---:|
| T4 small | 16 GB | $0.40 | ~$0.07 |
| L4 | 24 GB | $0.80 | ~$0.13 |
| A10G small | 24 GB | $1.00 | ~$0.17 |
| A100 large | 80 GB | $2.50 | ~$0.42 |

These are listed hardware charges, not a quotation for a complete experiment. Storage/other applicable charges are separate. Jobs bill Starting/Running time by the minute; use an explicit timeout. Installation and startup also consume runtime. No d1 training fit or throughput is established by these figures. A T4 smoke would run mathematical tests/data/tokenizer checks only, without downloading model weights; later model preflight can use a larger approved device if required.

The [official training guide](https://huggingface.co/docs/hub/jobs-training) supports mounting an entire project directory; a UV script upload alone does not transfer our package/configs. Stage only the existing allowlisted source archive or an equally sanitized source directory. Never mount the workspace root with `.env`. Jobs discard their filesystem at completion, so acceptance reports and checkpoints need an approved persistent bucket/repository before a real run. Secrets are supplied through the provider's secret mechanism, never command-line literal values or source files.

## Alternative: RunPod Pod

[Pods](https://docs.runpod.io/pods/overview) give direct control over a remote GPU container and support SSH, JupyterLab and VS Code. They suit interactive dependency debugging. Use an official PyTorch template, record its actual image digest/Python/Torch/CUDA versions and create an isolated project environment on the remote machine. The template itself does not establish compatibility with the newer official Liquid TRL pins.

Use an on-demand Pod for the initial debug run. Select GPU/price/region from actual availability before approval; no universal RunPod rate is assumed. [Pod billing](https://docs.runpod.io/pods/pricing) and [storage lifetimes](https://docs.runpod.io/pods/storage/types) differ: container disk is temporary, a volume disk lasts until Pod deletion, and a network volume survives independently. Preserve checkpoints/results before terminating. Stopping GPU compute can leave storage charges.

## Migration requirements (implemented; runtime validation pending)

1. Replace the Colab-only workload guard with explicit supported remote-provider guards, retaining local-execution rejection and NVIDIA availability checks. Do not simply remove the guard.
2. Rename/generalize `scripts/verify_colab.py` into a remote acceptance runner; record provider, hardware, image, Python/dependency versions, source identity and output destination.
3. Keep solver/data/evaluation logic unchanged. Use scripts as the entry point, not notebook execution.
4. Prepare a source-only upload, frozen runtime specification, provider secret mapping, durable result destination, hardware choice and bounded timeout for review before spending.
5. First run: package/import validation, solver/data tests, deterministic 1,000-episode generation, tokenizer-only checks and offline W&B, with no weights/fine-tuning.
6. After those pass: separately approved pinned-weight download, real TRL collator/gradient checks and zero-shot evaluation; only then a bounded overfit run.

The official d1 architecture/loader/readout findings remain binding across providers. Generic AutoTrain SFT is not a verified substitute for the project's custom Liquid recipe. Provider migration and successful hardware allocation alone do not establish model-training compatibility.
