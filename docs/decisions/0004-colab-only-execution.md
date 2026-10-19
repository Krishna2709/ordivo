# ADR 0004: Colab-only execution after user constraint

Status: accepted, 2026-10-18.

The user stated that local RAM/disk are insufficient and required Colab GPUs only. This overrides earlier local CPU workflow assumptions. Local work is restricted to source/document editing and compact source packaging; all installations, tests, synthetic generation, tokenizer downloads, inference and training run on the managed Colab GPU VM. Exact numerical solvers still use that VM's CPU.

Notebook entry cells and CLI workload entry points check managed-Colab markers and NVIDIA GPU availability. No override to run locally is supplied. The previous temporary local environment/cache/data were removed. Previous results are retained only as historical evidence, and final source changes remain unverified until rerun remotely.

A connected VS Code kernel is user-reported. No remote executor is exposed to this agent, so notebook cells must be run through the connected UI until a supported execution tool is available. No public endpoint, tunnel, credential extraction or unsupported kernel attachment is introduced.
