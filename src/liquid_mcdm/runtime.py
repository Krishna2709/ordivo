"""Execution policy: workloads require a managed remote NVIDIA GPU runtime.

This standard-library-only module is also embedded in notebook preflight cells,
so hardware checks work before any package or project installation.
"""
import os
from pathlib import Path
import shutil
import subprocess


GPU_SETUP_HELP = (
    'In browser Colab select Runtime > Change runtime type > Hardware accelerator '
    '> an available NVIDIA GPU, then reconnect. In VS Code select a Colab server '
    'configured with an NVIDIA GPU and reconnect this notebook to that server. '
    'Run the preflight in the same runtime used for nbconvert. '
    'Do not try to fix GPU allocation with pip or install NVIDIA drivers in Colab.'
)


def find_nvidia_smi():
    """Resolve installed driver tooling, including common paths omitted from PATH."""
    executable = shutil.which('nvidia-smi')
    if executable:
        return executable
    for candidate in ('/usr/bin/nvidia-smi', '/usr/local/nvidia/bin/nvidia-smi'):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def require_colab_gpu():
    managed = bool(os.environ.get('COLAB_RELEASE_TAG') or
                   (Path('/content').is_dir() and os.environ.get('COLAB_BACKEND_VERSION')))
    if not managed:
        raise RuntimeError('Colab-only execution: connect a managed Colab GPU kernel; do not run locally.')
    return query_gpu('Colab', GPU_SETUP_HELP)


def require_remote_gpu():
    """Operational guard, not a security attestation: environment flags are spoofable."""
    if os.environ.get('RUNPOD_POD_ID'):
        pod_id = os.environ['RUNPOD_POD_ID'].strip()
        if not pod_id:
            raise RuntimeError('Runpod execution requires a nonempty pod identifier.')
        gpu = query_gpu('Runpod', 'Attach an NVIDIA GPU pod and retry.')
        return {'provider': 'runpod', 'pod_id': pod_id, 'gpu': gpu}
    if os.environ.get('JOB_ID'):
        accelerator = os.environ.get('ACCELERATOR', '').strip()
        if accelerator.lower() in ('', 'none'):
            raise RuntimeError('Hugging Face Jobs requires an NVIDIA GPU flavor, such as t4-small.')
        gpu = query_gpu('Hugging Face Jobs', 'Select an NVIDIA GPU Jobs flavor and retry.')
        return {'provider': 'huggingface_jobs', 'job_id': os.environ['JOB_ID'],
                'accelerator': accelerator, 'gpu': gpu}
    return {'provider': 'colab', 'gpu': require_colab_gpu()}


def query_gpu(provider, help_text):
    executable = find_nvidia_smi()
    if executable is None:
        raise RuntimeError(
            'NVIDIA GPU tooling is unavailable: nvidia-smi was not found in PATH or '
            'common driver locations. This session may be CPU/TPU or lack an attached '
            'NVIDIA GPU; no Python dependency installation has been attempted by this check. '
            + help_text
        )
    try:
        result = subprocess.run(
            [executable, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f'Could not query the {provider} NVIDIA GPU. ' + help_text) from exc
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('nvidia-smi did not report an available NVIDIA GPU. ' + help_text)
    return result.stdout.strip()
