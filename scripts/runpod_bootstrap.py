"""Runpod-side bootstrap: immutable source, CPU acceptance, then gated GPU training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify_archive(archive, expected):
    if not re.fullmatch(r'[0-9a-f]{64}', expected or ''):
        raise RuntimeError('A valid source SHA256 is required')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Source archive SHA256 differs from the prepared snapshot')
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] != 'decision-engines'
                    or any(part.startswith('.') for part in path.parts)
                    or mode & 0o170000 == 0o120000):
                raise RuntimeError('Unsafe source archive member')


def run_logged(command, log, *, cwd=None, env=None):
    print(f'Running {command[0]}; log: {log}', flush=True)
    with Path(log).open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        print(Path(log).read_text(errors='replace')[-8000:], flush=True)
        raise RuntimeError(f'Remote command failed with exit code {result.returncode}; see {log}')


def reject_torch_replacement(report_path):
    report = json.loads(Path(report_path).read_text())
    forbidden = []
    for item in report.get('install', []):
        name = item.get('metadata', {}).get('name', '').lower().replace('_', '-')
        if name in {'torch', 'torchvision', 'torchaudio'} or name.startswith('nvidia-'):
            forbidden.append(name)
    if forbidden:
        raise RuntimeError(f'Candidate dependencies would replace the image CUDA stack: {sorted(set(forbidden))}')


def main():
    pod_id = os.environ.get('RUNPOD_POD_ID', '')
    run_id = os.environ.get('RUNPOD_RUN_ID', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', pod_id) or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise RuntimeError('Runpod identifiers are required; do not run this bootstrap locally')
    workspace = Path('/workspace')
    source_archive = workspace / 'incoming' / run_id / 'source.zip'
    result = workspace / 'runs' / run_id
    result.mkdir(parents=True, exist_ok=False)
    state = {'run_id': run_id, 'pod_id': pod_id, 'status': 'starting', 'passed': False,
             'training_started': False, 'source_sha256': os.environ.get('MCDM_SOURCE_SHA256'),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version}

    def checkpoint():
        write_json(result / 'bootstrap.json', state)
        print(f"Bootstrap: {state['status']}", flush=True)

    checkpoint()
    try:
        verify_archive(source_archive, state['source_sha256'])
        source_dir = workspace / 'source' / run_id
        source_dir.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(source_archive) as archive:
            archive.extractall(source_dir)
        root = source_dir / 'decision-engines'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PIP_NO_INPUT='1',
                   PIP_DISABLE_PIP_VERSION_CHECK='1', HF_HUB_DISABLE_TELEMETRY='1',
                   PYTHONPATH=str(root / 'src'),
                   WANDB_MODE='offline', WANDB_DIR=str(result / 'wandb'),
                   WANDB_PROJECT='liquid-mcdm-lab',
                   WANDB_CACHE_DIR=str(result / 'wandb-cache'),
                   WANDB_DATA_DIR=str(result / 'wandb-data'))
        for name in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_DATA_DIR'):
            Path(env[name]).mkdir(parents=True, exist_ok=True)

        state['status'] = 'cpu_environment'
        checkpoint()
        cpu_python = workspace / 'venvs' / run_id / 'cpu' / 'bin' / 'python'
        cpu_python.parent.parent.mkdir(parents=True, exist_ok=False)
        run_logged([sys.executable, '-m', 'venv', str(cpu_python.parent.parent)],
                   result / 'cpu-venv.log')
        run_logged([str(cpu_python), '-m', 'pip', 'install', '--require-hashes',
                    '--only-binary=:all:', '--no-cache-dir', '-r',
                    str(root / 'requirements-cpu.lock')], result / 'cpu-install.log', env=env)
        state['status'] = 'cpu_acceptance'
        checkpoint()
        run_logged([str(cpu_python), str(root / 'scripts' / 'verify_remote.py'),
                    '--output-dir', str(result / 'acceptance')],
                   result / 'cpu-acceptance.log', cwd=root, env=env)
        acceptance = json.loads((result / 'acceptance' / 'acceptance.json').read_text())
        if not acceptance.get('passed'):
            raise RuntimeError('CPU acceptance did not produce a passing receipt')
        state['cpu_acceptance_passed'] = True

        state['status'] = 'training_environment'
        checkpoint()
        train_python = workspace / 'venvs' / run_id / 'train' / 'bin' / 'python'
        run_logged([sys.executable, '-m', 'venv', '--system-site-packages',
                    str(train_python.parent.parent)], result / 'train-venv.log', env=env)
        dry_run = result / 'candidate-install-report.json'
        run_logged([str(train_python), '-m', 'pip', 'install', '--dry-run', '--report', str(dry_run),
                    '-r', str(root / 'requirements-colab-candidate.txt')],
                   result / 'candidate-dry-run.log', env=env)
        reject_torch_replacement(dry_run)
        run_logged([str(train_python), '-m', 'pip', 'install', '--no-cache-dir', '-r',
                    str(root / 'requirements-colab-candidate.txt')],
                   result / 'candidate-install.log', env=env)
        run_logged([str(train_python), '-m', 'pip', 'check'], result / 'train-pip-check.log', env=env)

        state['status'] = 'gpu_compatibility_and_training'
        checkpoint()
        run_logged([str(train_python), str(root / 'scripts' / 'train_d1_overfit.py'),
                    '--config', str(root / 'configs' / 'd1_3b_lora.yaml'),
                    '--data', str(result / 'acceptance' / 'data'),
                    '--output', str(result / 'training')],
                   result / 'training.log', cwd=root, env=env)
        training = json.loads((result / 'training' / 'run.json').read_text())
        if not training.get('passed') or not training.get('training_validated'):
            raise RuntimeError('Training did not produce a passing compatibility/reload receipt')
        state['training_started'] = bool(training.get('training_started'))
        state.update(status='passed', passed=True,
                     finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        checkpoint()


if __name__ == '__main__':
    main()
