"""Acceptance pipeline. Execute on supported remote GPU VMs only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from liquid_mcdm.runtime import require_remote_gpu


def require_complete_tests(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    if not cases or any(case.find(tag) is not None for case in cases
                        for tag in ('skipped', 'failure', 'error')):
        raise RuntimeError('Acceptance requires a nonempty test suite with zero skips/failures/errors')
    return len(cases)


def require_identical_datasets(first, second):
    for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
        if (first/name).read_bytes() != (second/name).read_bytes():
            raise RuntimeError(f'Reproducibility failed: {name} differs')


def run_acceptance(out):
    runtime = require_remote_gpu()
    root = Path(__file__).resolve().parents[1]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'runtime': runtime, 'passed': False, 'status': 'running',
             'model_loaded': False, 'training_validated': False, 'completed_steps': [],
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'lock_sha256': hashlib.sha256((root/'requirements-cpu.lock').read_bytes()).hexdigest()}

    def checkpoint():
        (out/'acceptance.json').write_text(json.dumps(state, indent=2)+'\n')

    env = dict(os.environ, WANDB_MODE='offline',
               WANDB_CACHE_DIR=str(out/'wandb-cache'), WANDB_DATA_DIR=str(out/'wandb-data'))
    for name in ('WANDB_CACHE_DIR', 'WANDB_DATA_DIR'):
        Path(env[name]).mkdir(parents=True)
    commands = [
        ('pip-check', [sys.executable, '-m', 'pip', 'check']),
        ('freeze', [sys.executable, '-m', 'pip', 'freeze', '--all']),
        ('environment', [sys.executable, '-m', 'liquid_mcdm.cli', 'environment']),
        ('generate', [sys.executable, '-m', 'liquid_mcdm.cli', 'generate', '--output', str(out/'data'), '--count', '1000', '--seed', '42']),
        ('repeat', [sys.executable, '-m', 'liquid_mcdm.cli', 'generate', '--output', str(out/'repeat'), '--count', '1000', '--seed', '42']),
        ('tokenizer', [sys.executable, '-m', 'liquid_mcdm.cli', 'tokenizer-check', '--data', str(out/'data'), '--output', str(out/'tokenizer_preflight.json')]),
        ('pytest', [sys.executable, '-m', 'pytest', '--junitxml', str(out/'pytest.xml')]),
        ('baseline', [sys.executable, '-m', 'liquid_mcdm.cli', 'baseline', '--data', str(out/'data'), '--split', 'test', '--output', str(out/'baseline.json'), '--wandb', 'offline']),
    ]
    checkpoint()
    try:
        for name, command in commands:
            state['active_step'] = name
            checkpoint()
            log = out/f'{name}.log'
            print(f'Running {name}; log: {log}', flush=True)
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-6000:], flush=True)
                raise RuntimeError(f'{name} failed with exit code {result.returncode}')
            if name == 'pytest':
                state['test_count'] = require_complete_tests(out/'pytest.xml')
            if name == 'repeat':
                require_identical_datasets(out/'data', out/'repeat')
                state['two_run_byte_identity'] = True
            if name == 'tokenizer':
                report = json.loads((out/'tokenizer_preflight.json').read_text())
                if not report['alias_eos_boundary_passed'] or report['checked_examples'] < 1:
                    raise RuntimeError('Tokenizer preflight did not check eligible examples')
            state['completed_steps'].append(name)
            checkpoint()
        state.update(passed=True, status='passed', active_step=None)
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        checkpoint()
    print(json.dumps(state, indent=2), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run_acceptance(args.output_dir)


if __name__ == '__main__':
    main()
