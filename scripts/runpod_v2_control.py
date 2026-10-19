"""One-pod Ordivo v2 controller; local work is control and source packaging only.

Usage: --status, --launch, --setup, --run TASK --repo PATH --commit SHA
       --command SHELL, --fetch TASK, --cleanup. Every run has an immutable receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from runpod_control import request, ssh_target

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ENV = Path('/Users/krishna/Desktop/apex-7/decision-engines/.env')
CONFIG = json.loads((ROOT / 'configs/runpod_v2.json').read_text())
RUNS = ROOT / 'reports/runs'
PREFIX = 'v2-swarm-'


def utcnow():
    return datetime.now(timezone.utc)


def write_once(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def write_state(path: Path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def load_secret(name):
    if os.getenv(name):
        return os.environ[name]
    for line in CANONICAL_ENV.read_text().splitlines():
        key, sep, value = line.removeprefix('export ').partition('=')
        if sep and key.strip() == name:
            return value.strip().strip('"\'')
    raise RuntimeError(f'{name} unavailable')


def api(method, path, *, body=None):
    return request(method, path, body=body, token=load_secret('RUNPOD_API_KEY'))


def active_run():
    matches = sorted(RUNS.glob(PREFIX + '*/pod.json'))
    active = []
    for path in matches:
        state = json.loads(path.read_text())
        if state['status'] not in {'deleted', 'absent'}:
            active.append((path, state))
    if len(active) > 1:
        raise RuntimeError('Multiple active v2 receipts require reconciliation')
    return active[0] if active else None


def pods():
    result = api('GET', '/pods')
    return result if isinstance(result, list) else result.get('data', [])


def preflight():
    current = pods()
    if current:
        raise RuntimeError('RunPod account already has a pod; reconcile before allocating')
    catalog = request('GET',
                      'https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD',
                      absolute=True, token=load_secret('RUNPOD_API_KEY'))
    items = catalog if isinstance(catalog, list) else catalog.get('data', [])
    gpu = next((x for x in items if x.get('id') == CONFIG['gpu_type_id']), None)
    if gpu is None or gpu.get('availability') == 'NONE':
        raise RuntimeError('A40 is unavailable')
    price = gpu.get('price', {}).get('secure')
    if price is None or float(price) > CONFIG['max_hourly_usd']:
        raise RuntimeError('A40 secure catalog price exceeds cap')
    if CONFIG['max_runtime_seconds'] > 7200 or CONFIG['container_disk_gb'] != 50:
        raise RuntimeError('Runtime or disk policy mismatch')
    return price


def launch():
    if active_run():
        raise RuntimeError('An active v2 launch receipt already exists')
    price = preflight()
    run_id = PREFIX + utcnow().strftime('%Y%m%dT%H%M%SZ')
    path = RUNS / run_id / 'pod.json'
    started = utcnow()
    state = {'run_id': run_id, 'status': 'creating', 'created_utc': started.isoformat(),
             'deadline_utc': (started + timedelta(seconds=CONFIG['max_runtime_seconds'])).isoformat(),
             'catalog_hourly_usd': price, 'pod_id': None, 'network_volume_id': None,
             'config_sha256': hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()}
    write_once(path, state)
    try:
        pod = api('POST', '/pods', body={
            'name': run_id, 'cloudType': 'SECURE', 'computeType': 'GPU',
            'gpuTypeIds': [CONFIG['gpu_type_id']], 'gpuTypePriority': 'custom',
            'gpuCount': 1, 'dataCenterIds': [CONFIG['data_center_id']],
            'dataCenterPriority': 'custom', 'imageName': CONFIG['image_name'],
            'containerDiskInGb': CONFIG['container_disk_gb'], 'interruptible': False,
            'ports': ['22/tcp'], 'supportPublicIp': True,
            'env': {'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_MODE': 'online'}})
        state['pod_id'] = pod['id']
        state['hourly_usd'] = pod.get('costPerHr')
        state['status'] = 'created'
        write_state(path, state)
        if state['hourly_usd'] is None or float(state['hourly_usd']) > CONFIG['max_hourly_usd']:
            cleanup()
            raise RuntimeError('Created pod exceeds price cap')
        reaper = subprocess.Popen([sys.executable, __file__, '--reap', str(path)],
                                  stdin=subprocess.DEVNULL,
                                  stdout=(path.parent / 'reaper.log').open('w'),
                                  stderr=subprocess.STDOUT, start_new_session=True)
        state['reaper_pid'] = reaper.pid
        write_state(path, state)
        time.sleep(1)
        if reaper.poll() is not None:
            cleanup()
            raise RuntimeError('Cost reaper failed to arm')
        return {'run_id': run_id, 'pod_id': state['pod_id'],
                'hourly_usd': state['hourly_usd'], 'deadline_utc': state['deadline_utc']}
    except BaseException:
        if state.get('pod_id') and state['status'] != 'deleted':
            cleanup()
        raise


def reap(path):
    while True:
        state = json.loads(path.read_text())
        if state['status'] in {'deleted', 'absent'}:
            return
        remaining = (datetime.fromisoformat(state['deadline_utc']) - utcnow()).total_seconds()
        if remaining <= 0:
            try:
                api('DELETE', f"/pods/{state['pod_id']}")
                state['status'] = 'deleted'
                state['deleted_utc'] = utcnow().isoformat()
                write_state(path, state)
                return
            except Exception:
                time.sleep(15)
        else:
            time.sleep(min(30, remaining))


def live_pod():
    pair = active_run()
    if not pair:
        raise RuntimeError('No active v2 pod receipt')
    path, state = pair
    pod = api('GET', f"/pods/{state['pod_id']}")
    if float(pod.get('costPerHr') or 999) > CONFIG['max_hourly_usd']:
        raise RuntimeError('Live pod price exceeds cap')
    return path, state, pod


def ssh_args(pod):
    host, port = ssh_target(pod)
    key = str(Path(CONFIG['ssh_key']).expanduser())
    known = str((RUNS / active_run()[1]['run_id'] / 'known_hosts'))
    common = ['-i', key, '-o', f'UserKnownHostsFile={known}',
              '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15']
    return ['ssh', *common, '-p', str(port), f'root@{host}'], ['scp', *common, '-P', str(port)], f'root@{host}'


def ssh(pod, command, *, timeout=3600):
    base, _, _ = ssh_args(pod)
    return subprocess.run([*base, command], capture_output=True, text=True, timeout=timeout)


def wait_ssh(timeout=600):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        _, _, pod = live_pod()
        try:
            result = ssh(pod, 'true', timeout=25)
            if result.returncode == 0:
                return pod
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        time.sleep(10)
    raise RuntimeError('Pod SSH unavailable before timeout')


def setup():
    pod = wait_ssh()
    root = CONFIG['remote_root']
    packages = shlex.join(CONFIG['cpu_packages'])
    tokenizers = shlex.join(CONFIG['tokenizer_packages'])
    script = (f'set -eu; mkdir -p {shlex.quote(root)}; '
              f'python -m venv --system-site-packages {shlex.quote(root)}/cpu-venv; '
              f'{shlex.quote(root)}/cpu-venv/bin/python -m pip install --disable-pip-version-check '
              f'{packages} {tokenizers}; '
              f'{shlex.quote(root)}/cpu-venv/bin/python -c "import torch; print(torch.__version__)"; '
              f'{shlex.quote(root)}/cpu-venv/bin/python -m pip freeze')
    result = ssh(pod, script, timeout=1800)
    path, state, _ = live_pod()
    (path.parent / 'setup.log').write_text(result.stdout[-30000:] + result.stderr[-10000:])
    if result.returncode:
        raise RuntimeError('Remote CPU environment setup failed; inspect setup.log')
    state['setup_utc'] = utcnow().isoformat()
    state['setup_freeze_sha256'] = hashlib.sha256(result.stdout.encode()).hexdigest()
    write_state(path, state)
    return {'setup': 'complete', 'freeze_sha256': state['setup_freeze_sha256']}


def run_task(task, repo, commit, command, secret_names=()):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('Invalid task identifier')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Full 40-character commit required')
    path, state, pod = live_pod()
    if 'setup_utc' not in state:
        raise RuntimeError('Run --setup before tasks')
    if utcnow() >= datetime.fromisoformat(state['deadline_utc']) - timedelta(minutes=5):
        raise RuntimeError('Too close to pod deadline for another task')
    task_dir = path.parent / 'tasks' / task
    if task_dir.exists():
        raise RuntimeError('Task ID already used; receipts are immutable')
    task_dir.mkdir(parents=True)
    archive = task_dir / 'source.tar'
    with archive.open('wb') as stream:
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', commit],
                       check=True, stdout=stream)
    source_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    remote = f"{CONFIG['remote_root']}/tasks/{task}"
    base, scp, target = ssh_args(pod)
    subprocess.run([*base, f'mkdir -p {shlex.quote(remote)}/src'], check=True)
    subprocess.run([*scp, str(archive), f'{target}:{remote}/source.tar'], check=True)
    allowed = {'wandb': ('WANDB_API_KEY', 'WEIGHTS_AND_BIASES_API_KEY'),
               'hf': ('HF_TOKEN', 'HUGGINGFACE_API_KEY')}
    if any(name not in allowed for name in secret_names):
        raise ValueError('Unknown secret alias')
    if secret_names:
        env_text = ''.join(f'export {allowed[name][0]}={shlex.quote(load_secret(allowed[name][1]))}\n'
                           for name in secret_names)
        secret_path = f'{remote}/secrets.sh'
        transfer = subprocess.run([*base, f'umask 077; cat > {shlex.quote(secret_path)}'],
                                  input=env_text, capture_output=True, text=True, timeout=30)
        if transfer.returncode:
            raise RuntimeError('Scoped remote secret transfer failed')
    wrapped = (f'set -o pipefail; cd {shlex.quote(remote)}/src; '
               f'tar -xf {shlex.quote(remote)}/source.tar; '
               f'export PATH={shlex.quote(CONFIG["remote_root"] + "/cpu-venv/bin")}:$PATH; '
               f'export RUNPOD_POD_ID={shlex.quote(state["pod_id"])}; '
               + (f'. {shlex.quote(remote)}/secrets.sh; '
                  f'trap "rm -f {shlex.quote(remote)}/secrets.sh" EXIT; '
                  if secret_names else '') +
               f'({command}) > {shlex.quote(remote)}/stdout.log 2>&1; '
               f'code=$?; echo "$code" > {shlex.quote(remote)}/exit_code; exit "$code"')
    receipt = {'task': task, 'commit': commit, 'command': command,
               'source_sha256': source_sha, 'started_utc': utcnow().isoformat(),
               'remote_dir': remote, 'pod_id': state['pod_id'],
               'secret_aliases': list(secret_names)}
    write_once(task_dir / 'receipt.json', receipt)
    result = subprocess.run([*base, 'bash -lc ' + shlex.quote(wrapped)],
                            capture_output=True, text=True, timeout=3600)
    receipt['finished_utc'] = utcnow().isoformat()
    receipt['returncode'] = result.returncode
    write_state(task_dir / 'receipt.json', receipt)
    (task_dir / 'ssh.log').write_text(result.stdout[-10000:] + result.stderr[-10000:])
    fetch_task(task)
    return {key: receipt[key] for key in ('task', 'commit', 'returncode', 'source_sha256')}


def fetch_task(task):
    path, _, pod = live_pod()
    task_dir = path.parent / 'tasks' / task
    if not task_dir.exists():
        raise RuntimeError('Unknown task')
    remote = f"{CONFIG['remote_root']}/tasks/{task}"
    _, scp, target = ssh_args(pod)
    for filename in ('stdout.log', 'exit_code'):
        subprocess.run([*scp, f'{target}:{remote}/{filename}', str(task_dir / filename)],
                       check=True, capture_output=True)
    return {'task': task, 'log_sha256': hashlib.sha256((task_dir / 'stdout.log').read_bytes()).hexdigest()}


def cleanup():
    pair = active_run()
    if not pair:
        return {'status': 'no-active-pod'}
    path, state = pair
    pod_id = state.get('pod_id')
    if pod_id:
        try:
            api('DELETE', f'/pods/{pod_id}')
        except RuntimeError as exc:
            if '404' not in str(exc):
                raise
    state['status'] = 'deleted'
    state['deleted_utc'] = utcnow().isoformat()
    if state.get('hourly_usd'):
        elapsed = (utcnow() - datetime.fromisoformat(state['created_utc'])).total_seconds()
        state['estimated_compute_usd'] = round(elapsed / 3600 * float(state['hourly_usd']), 4)
    write_state(path, state)
    return {'status': state['status'], 'pod_id': pod_id,
            'estimated_compute_usd': state.get('estimated_compute_usd')}


def status():
    pair = active_run()
    if not pair:
        return {'active_receipt': None, 'account_pods': len(pods())}
    _, state = pair
    live = api('GET', f"/pods/{state['pod_id']}")
    return {'run_id': state['run_id'], 'pod_id': state['pod_id'],
            'status': live.get('desiredStatus'), 'hourly_usd': live.get('costPerHr'),
            'deadline_utc': state['deadline_utc'], 'setup': 'setup_utc' in state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ('status', 'launch', 'setup', 'cleanup'):
        group.add_argument('--' + action, action='store_true')
    group.add_argument('--run', metavar='TASK')
    group.add_argument('--fetch', metavar='TASK')
    group.add_argument('--reap', metavar='RECEIPT')
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--command')
    parser.add_argument('--with-secret', action='append', choices=('wandb', 'hf'), default=[])
    args = parser.parse_args()
    if args.reap:
        reap(Path(args.reap))
        return
    if args.run:
        if not all((args.repo, args.commit, args.command)):
            parser.error('--run requires --repo, --commit and --command')
        result = run_task(args.run, args.repo, args.commit, args.command, args.with_secret)
    elif args.fetch:
        result = fetch_task(args.fetch)
    elif args.launch:
        result = launch()
    elif args.setup:
        result = setup()
    elif args.cleanup:
        result = cleanup()
    else:
        result = status()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'v2 control failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
