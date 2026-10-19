# ADR 0005: Scripted Hugging Face Jobs acceptance

Date: 2026-10-18. Status: implementation prepared; execution pending approval.

## Context and alternatives

Colab nbconvert failed before installation because nvidia-smi was unavailable. Improved diagnostics cannot allocate a GPU. The user selected Hugging Face Jobs over RunPod for time-limited scripted runs. This supersedes ADR0004's provider restriction; its no-local-workload requirement remains binding. RunPod offers interactive debugging but requires separate lifecycle management. Staying on Colab would require another kernel/allocation handoff.

## Decision

Keep mathematical/data/model-formatting domains unchanged. Admit HF JOB_ID plus a non-CPU ACCELERATOR at the execution boundary, then require a successful NVIDIA inventory. Environment flags are operational checks, not trustworthy hardware attestation. Preserve the Colab notebook guard. Use one shared acceptance runner with a thin Colab compatibility wrapper.

Pin the public Python3.12 Linux amd64 image by OCI digest; install the hashed CPU lock in a fresh remote venv with wheels only. Import the source using explicit PYTHONPATH instead of invoking an unpinned build-isolation environment. This smoke image has no training stack. No Torch, CUDA compatibility, model loading, forward/backward or fine-tuning success is inferred.

Stage an allowlisted ZIP, bootstrap and receipt locally. Use the already installed huggingface-hub1.12.0 API for control operations (no local dependency installation). Newer CLI local-directory auto-sync is not assumed supported by this client. On explicit paid approval, create one new private bucket in the authenticated user's namespace, upload only those three files, and mount it at /job-store. Save setup logs, per-step outputs, datasets and offline W&B data under results/JOB_ID. The source archive is hash-checked before extraction and installation; a submission receipt prevents accidental duplicate attempts. Credentials are used only by the local control client, with no job secrets or .env upload.

The first job is t4-small, 600 seconds. Re-query hardware pricing before submission and reject an estimate above $0.07 compute. Storage is separate and survives job completion. There are no automatic resubmissions, timeout extensions or training steps.

## Acceptance and failure semantics

Require pip check, frozen dependency inventory, two byte-identical 1,000-episode outputs, pinned tokenizer checks, full pytest with zero skips, and offline W&B baseline artifacts. Record each active/completed step. A failure must not write passed=true. A forced provider timeout may prevent finalizers and leave status=running; reconcile HF terminal status with complete acceptance/bootstrap reports. Mount/setup failure can leave only provider logs. Durable evidence must be read back after the job before claiming acceptance.

## Evidence and limits

The [official Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration) documents JOB_ID/ACCELERATOR, timeouts and bucket mounts. [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets) documents private creation and mutable/nonversioned storage. [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing) describes billing; authenticated hardware metadata returned $0.006667/minute for t4-small on 2026-10-18. Identity lookup succeeded for blackOrder, but credit balance, job/bucket write permission, image startup and driver injection have not been validated. Read-only registry metadata resolved the image; no image layers were downloaded locally.
