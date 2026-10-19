# Hugging Face Jobs runbook

The selected workflow runs all dependencies, tests, data generation and tokenizer work on HF hardware. Local activity is limited to source preparation and small control API requests. No expensive training is included.

## Reviewable first job

| Setting | Value |
|---|---|
| Account verified read-only | blackOrder |
| Hardware | t4-small, NVIDIA T4 16GB |
| Hard timeout | 600 seconds, including installation/checks |
| Current compute estimate | $0.06667 at $0.006667/minute, checked 2026-10-18 |
| Storage | New private blackOrder/liquid-mcdm-smoke-UTC_TIMESTAMP bucket |
| Uploads | source.zip, hf_job_bootstrap.py, prepared.json; excludes .env |
| Durable results | /job-store/results/JOB_ID in that bucket |
| W&B | Offline, logs and artifact staging in the bucket |
| Model access | Four pinned tokenizer/config/template files; no weights |
| Training | None; training_validated=false |

[configs/hf_jobs_smoke.json](../configs/hf_jobs_smoke.json) pins the image and job settings. The Python3.12 amd64 image digest was resolved from public registry metadata without downloading layers. Bucket storage survives completion and may incur charges beyond the account's allowance; the compute estimate does not include storage. See [HF billing](https://huggingface.co/docs/hub/jobs-pricing) and [bucket pricing](https://huggingface.co/docs/hub/storage-buckets#pricing). Credit balance and write permissions are not established by successful login.

## Prepare and inspect (no paid job)

The workstation already has huggingface-hub1.12.0 in /opt/miniconda3. No local installation is needed. Source preparation does not load credentials:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python scripts/prepare_hf_job.py
```

This creates dist/hf-job with the three upload files. The receipt records each source file's SHA256, archive SHA256/size, bootstrap SHA256, image and configuration. Source/config changes make the snapshot stale; submission refuses it. Re-prepare only after final source edits and review.

Read-only identity and hardware preflight uses a recognized HF token from environment or .env without printing it:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python scripts/prepare_hf_job.py --preflight
```

## Submit only after explicit approval

PROJECT_BRIEF.md requires permission before spending. The following command creates private storage, uploads the reviewed source and submits **one paid job**. It is not part of preparation:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python scripts/prepare_hf_job.py --submit
```

The script rechecks source identity and current compute price, creates a new private bucket without reusing an existing name, then submits the configured image/600-second job. It injects no API keys into the container. It writes dist/hf-job/submission.json with bucket, job ID, job URL and result location. A receipt is written before any mutation; an interrupted/ambiguous submission is not automatically retried. Reconcile the account before retrying. Never remove that receipt just to bypass the duplicate-attempt guard.

The remote bootstrap requires HF GPU markers, the mounted output bucket, archive integrity, Python3.12 and a successful NVIDIA query. It installs requirements-cpu.lock using --require-hashes/--only-binary in a fresh remote venv. Package source is loaded by PYTHONPATH. It saves setup logs before invoking scripts/verify_remote.py.

## Monitor and retrieve evidence

Open the job_url in the submission receipt to inspect status and logs. Existing hf CLI also supports `hf jobs logs JOB_ID --namespace blackOrder`, `hf jobs inspect JOB_ID --namespace blackOrder`, and `hf jobs cancel JOB_ID --namespace blackOrder` when authenticated. Do not put token values on the command line. The launcher reads .env internally; that does not log in a separate CLI process. Prefer the job page if the CLI is not already authenticated. Closing a log viewer does not stop billing; the explicit timeout or cancellation stops the job.

Use the bucket web page (https://huggingface.co/buckets/blackOrder/BUCKET_NAME) for result inspection without downloading datasets to the laptop. Results include:

- bootstrap.json and install logs;
- acceptance/acceptance.json, environment.log, freeze.log and pip-check.log;
- acceptance/pytest.xml and pytest.log;
- two generated datasets, tokenizer_preflight.json, baseline.json;
- offline W&B run and durable wandb-cache/wandb-data artifact files.

Require HF terminal success plus complete passed=true reports and persisted artifacts. Check zero skipped tests, matching dataset bytes/hashes and actual offline run files. A timeout can leave a running checkpoint or partial file; neither is success. Download only small reviewed reports if needed, keeping generated data/models remote. A successful smoke does not authorize a longer job or establish training compatibility. Next requires separately approved weights plus real TRL collator, LoRA, forward/backward and reload checks.

## Current verification status

Source and control API inspection only. New regression tests and the complete pipeline have **not** been executed in the new runtime. No bucket, source upload or job has been created during preparation. The 10-minute window may expire during initial image/package startup; inspect logs before proposing another bounded run.
