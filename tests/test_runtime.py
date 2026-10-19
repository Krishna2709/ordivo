"""Runtime regression tests. Execute on approved remote GPUs, never locally."""
import json
from pathlib import Path
import subprocess

import pytest

from liquid_mcdm import runtime


@pytest.fixture(autouse=True)
def clear_runtime_flags(monkeypatch):
    for name in ('COLAB_RELEASE_TAG', 'COLAB_BACKEND_VERSION', 'JOB_ID', 'ACCELERATOR',
                 'RUNPOD_POD_ID'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def managed(monkeypatch):
    monkeypatch.setenv('COLAB_RELEASE_TAG', 'test-managed-runtime')


def no_gpu_tool(monkeypatch):
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: None)
    monkeypatch.setattr(runtime.os, 'access', lambda path, mode: False)


def test_missing_nvidia_tool_reports_runtime_selection(managed, monkeypatch):
    no_gpu_tool(monkeypatch)
    with pytest.raises(RuntimeError, match='NVIDIA GPU tooling is unavailable') as error:
        runtime.require_colab_gpu()
    assert 'Hardware accelerator' in str(error.value)
    assert 'pip' in str(error.value)


def test_gpu_inventory_returned_only_when_query_succeeds(managed, monkeypatch):
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, 'Tesla T4, 15360 MiB, 550\n', ''))
    assert runtime.require_colab_gpu() == 'Tesla T4, 15360 MiB, 550'


@pytest.mark.parametrize('code,stdout', [(1, ''), (0, '')])
def test_no_usable_gpu_is_not_success(managed, monkeypatch, code, stdout):
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], code, stdout, ''))
    with pytest.raises(RuntimeError, match='did not report an available NVIDIA GPU'):
        runtime.require_colab_gpu()


@pytest.mark.parametrize('failure', [FileNotFoundError('nvidia-smi'),
                                     subprocess.TimeoutExpired('nvidia-smi', 15)])
def test_tool_failure_has_actionable_error(managed, monkeypatch, failure):
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(runtime.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='Could not query the Colab NVIDIA GPU'):
        runtime.require_colab_gpu()


def test_common_driver_path_is_resolved_when_path_is_incomplete(monkeypatch):
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: None)
    monkeypatch.setattr(runtime.Path, 'is_file', lambda path: str(path) == '/usr/bin/nvidia-smi')
    monkeypatch.setattr(runtime.os, 'access', lambda path, mode: str(path) == '/usr/bin/nvidia-smi')
    assert runtime.find_nvidia_smi() == '/usr/bin/nvidia-smi'


def test_local_runtime_remains_rejected(monkeypatch):
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    monkeypatch.delenv('COLAB_BACKEND_VERSION', raising=False)
    with pytest.raises(RuntimeError, match='do not run locally'):
        runtime.require_colab_gpu()
    with pytest.raises(RuntimeError, match='do not run locally'):
        runtime.require_remote_gpu()


@pytest.mark.parametrize('accelerator', ['', 'none'])
def test_hf_cpu_or_missing_accelerator_rejected(monkeypatch, accelerator):
    monkeypatch.setenv('JOB_ID', 'test-job')
    monkeypatch.setenv('ACCELERATOR', accelerator)
    with pytest.raises(RuntimeError, match='requires an NVIDIA GPU flavor'):
        runtime.require_remote_gpu()


def test_accelerator_alone_is_not_managed_runtime(monkeypatch):
    monkeypatch.setenv('ACCELERATOR', 't4-small')
    with pytest.raises(RuntimeError, match='do not run locally'):
        runtime.require_remote_gpu()


def test_hf_requires_actual_gpu_tool(monkeypatch):
    monkeypatch.setenv('JOB_ID', 'test-job')
    monkeypatch.setenv('ACCELERATOR', 't4-small')
    no_gpu_tool(monkeypatch)
    with pytest.raises(RuntimeError, match='NVIDIA GPU tooling is unavailable'):
        runtime.require_remote_gpu()


def test_hf_gpu_reports_provider_and_job(monkeypatch):
    monkeypatch.setenv('JOB_ID', 'test-job')
    monkeypatch.setenv('ACCELERATOR', 't4-small')
    monkeypatch.setattr(runtime, 'find_nvidia_smi', lambda: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 0, 'T4, 15360 MiB, 550\n', ''))
    assert runtime.require_remote_gpu() == {
        'provider': 'huggingface_jobs', 'job_id': 'test-job',
        'accelerator': 't4-small', 'gpu': 'T4, 15360 MiB, 550'}


def test_runpod_requires_actual_gpu_tool(monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod-1')
    no_gpu_tool(monkeypatch)
    with pytest.raises(RuntimeError, match='NVIDIA GPU tooling is unavailable'):
        runtime.require_remote_gpu()


def test_runpod_gpu_reports_provider_and_pod(monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID', 'pod-1')
    monkeypatch.setattr(runtime, 'find_nvidia_smi', lambda: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 0, 'A40, 46068 MiB, 570\n', ''))
    assert runtime.require_remote_gpu() == {
        'provider': 'runpod', 'pod_id': 'pod-1', 'gpu': 'A40, 46068 MiB, 570'}


@pytest.mark.parametrize('notebook', ['de01.ipynb',
    'notebooks/00_environment_check.ipynb', 'notebooks/01_mcdm_data_and_baselines.ipynb',
    'notebooks/02_d1_3b_lora_colab.ipynb', 'notebooks/03_evaluate_and_compare.ipynb'])
def test_actual_notebook_first_cell_reports_missing_gpu(notebook, managed, monkeypatch):
    # Reproduce the user's failing cell, before any pip/install/data work.
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / notebook).read_text())
    source = ''.join(next(c for c in nb['cells'] if c['cell_type'] == 'code')['source'])
    no_gpu_tool(monkeypatch)
    with pytest.raises(RuntimeError, match='NVIDIA GPU tooling is unavailable'):
        exec(compile(source, notebook, 'exec'), {})


def test_inventory_uses_resolved_driver_path(monkeypatch):
    from liquid_mcdm import environment
    executable = '/usr/local/nvidia/bin/nvidia-smi'
    monkeypatch.setattr(environment, 'find_nvidia_smi', lambda: executable)

    def absent_distribution(name):
        raise environment.importlib.metadata.PackageNotFoundError(name)

    def query(args, **kwargs):
        if args[0] != executable:
            raise FileNotFoundError(args[0])
        return subprocess.CompletedProcess(args, 0, 'Tesla T4, 15360 MiB, 550\n', '')

    monkeypatch.setattr(environment.importlib.metadata, 'version', absent_distribution)
    monkeypatch.setattr(environment.subprocess, 'run', query)
    assert environment.environment_report()['gpu'] == 'Tesla T4, 15360 MiB, 550'
