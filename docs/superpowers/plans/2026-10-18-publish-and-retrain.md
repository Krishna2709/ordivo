# Publish and Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the reproducible Liquid MCDM project, synthetic dataset, LoRA adapter, and experiment evidence to GitHub and Hugging Face, then run a better-instrumented second A40 training experiment that improves held-out accuracy.

**Architecture:** GitHub is the source and documentation record. Hugging Face has separate public dataset and PEFT adapter repositories so large artifacts never transit the laptop. Runpod reattaches the preserved volume, executes tests and training, uploads artifacts directly to Hugging Face, syncs W&B, then deletes compute. All runs retain immutable IDs, source/data hashes, manifests, metrics, and limitations.

**Tech Stack:** Python 3.12, NumPy, Transformers 5.19, TRL 1.15, PEFT 0.21, W&B 0.22, Runpod REST, GitHub CLI, Hugging Face Hub.

**Spec:** `PROJECT_BRIEF.md`

## Global Constraints

- No model download, dataset generation, tests, or training on the local workstation.
- Never commit, print, or upload `.env` or credentials.
- Publish code under Apache-2.0; synthetic data under CC-BY-4.0; the adapter remains subject to LiquidAI/d1-3B's LFM Open License v1.0.
- Keep exact-solver fidelity, synthetic imitation, and human preference claims separate.
- W&B must include step metrics, config, dataset identity, before/after metrics, prediction analysis, parity diagnostics, and artifact manifests.
- Delete paid Runpod compute after terminal evidence; preserve the network volume until remote publication is verified.

## Review Focus

- Repository bundles or histories must contain no secrets or generated model weights.
- Hugging Face model and dataset cards must state licenses, provenance, limitations, and exact base revision.
- Retraining acceptance must compare against run `20261019T002734Z` on the same held-out split.
- Reload diagnostics must distinguish decision agreement from exact probability equality.
- Publishing must verify remote file lists, sizes, and hashes rather than relying on upload success messages.

### Task 1: Publication-ready source repository

- [ ] Copy small run evidence into a tracked report directory and write publication/announcement documentation.
- [ ] Initialize git, create `Krishna2709/liquid-mcdm-lab`, commit, push, and verify the remote tree.

### Task 2: Hugging Face dataset and model cards

- [ ] Create source-controlled dataset/model cards and upload helpers.
- [ ] Create public `blackOrder/liquid-mcdm-synthetic` and `blackOrder/d1-3b-mcdm-lora` repositories.

### Task 3: Failure analysis and v2 training implementation

- [ ] Record every failed attempt and its root cause.
- [ ] Parameterize training for a larger sample/step budget, deterministic reload diagnostics, and complete W&B logging.
- [ ] Add remote tests for the new configuration and telemetry outputs.

### Task 4: Remote retrain and publication

- [ ] Reattach network volume `ngica7qkhd` to one A40 pod with a cost backstop.
- [ ] Run the remote suite, dataset validation, reload investigation, and v2 training.
- [ ] Sync W&B, upload dataset and adapter directly from Runpod to Hugging Face, and verify hashes/file lists.
- [ ] Delete the pod and decide whether the now-redundant volume can be deleted based on verified remote persistence.

### Task 5: Experiment report and final review

- [ ] Add the requested standalone HTML report with charts and prediction examples.
- [ ] Update README/model card/dataset card/announcement with actual v2 evidence.
- [ ] Push final GitHub changes and verify all public links.
