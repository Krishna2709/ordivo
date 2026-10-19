"""Standard-library bootstrap. Executed by HF Jobs, never on the workstation."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import zipfile


def verify_archive(archive, expected):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Source archive SHA256 differs from the submitted snapshot')
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] != 'decision-engines'
                    or any(part.startswith('.') for part in path.parts)
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError('Unsafe source archive member')


def main():
    job_id = os.environ.get('JOB_ID', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', job_id):
        raise RuntimeError('HF Jobs only: valid JOB_ID required; do not run locally')
    if os.environ.get('ACCELERATOR', '').lower() in ('', 'none'):
        raise RuntimeError('An NVIDIA GPU Jobs flavor is required')
    # Fail closed if the persistent bucket was omitted from the launch request.
    store = Path('/job-store')
    mounts = {line.split()[4] for line in Path('/proc/self/mountinfo').read_text().splitlines()}
    if str(store) not in mounts:
        raise RuntimeError('The private bucket must be mounted at /job-store')
    out = store/'results'/job_id
    out.mkdir(parents=True, exist_ok=False)
    state = {'job_id': job_id, 'status': 'starting', 'passed': False,
             'image': os.environ.get('MCDM_JOB_IMAGE'),
             'source_sha256': os.environ.get('MCDM_SOURCE_SHA256'),
             'python': sys.version, 'model_loaded': False, 'training_validated': False}

    def checkpoint():
        (out/'bootstrap.json').write_text(json.dumps(state, indent=2)+'\n')
        print('Bootstrap:', state['status'], flush=True)

    checkpoint()
    try:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('This acceptance image must supply Python 3.12')
        archive = store/'source.zip'
        verify_archive(archive, state['source_sha256'])
        with tempfile.TemporaryDirectory(prefix='liquid-mcdm-') as work:
            work = Path(work)
            with zipfile.ZipFile(archive) as source:
                source.extractall(work)
            root = work/'decision-engines'
            sys.path.insert(0, str(root/'src'))
            from liquid_mcdm.runtime import require_remote_gpu
            state['runtime'] = require_remote_gpu()
            state['status'] = 'installing'
            checkpoint()
            env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONDONTWRITEBYTECODE='1',
                       PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1', WANDB_MODE='offline')
            python = work/'venv/bin/python'
            commands = [
                [sys.executable, '-m', 'venv', str(work/'venv')],
                [str(python), '-m', 'pip', 'install', '--require-hashes', '--only-binary=:all:',
                 '--no-cache-dir', '-r', str(root/'requirements-cpu.lock')],
            ]
            for index, command in enumerate(commands):
                log = out/f'install-{index}.log'
                with log.open('w') as stream:
                    result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT)
                if result.returncode:
                    print(log.read_text()[-6000:], flush=True)
                    raise RuntimeError(f'Installation step {index} failed; see {log}')
            state['status'] = 'checking'
            checkpoint()
            result = subprocess.run([str(python), str(root/'scripts/verify_remote.py'),
                                     '--output-dir', str(out/'acceptance')], cwd=root, env=env)
            if result.returncode:
                raise RuntimeError('Acceptance failed; inspect persisted step logs')
            acceptance = json.loads((out/'acceptance/acceptance.json').read_text())
            if not acceptance.get('passed'):
                raise RuntimeError('Acceptance result is absent or incomplete')
            state.update(status='passed', passed=True)
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        checkpoint()


if __name__ == '__main__':
    main()
