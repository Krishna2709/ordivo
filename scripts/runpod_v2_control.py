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
import tarfile
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
    items = catalog if isinstance(catalog, list) else next(
        (catalog[key] for key in ('data', 'items', 'gpus')
         if isinstance(catalog.get(key), list)), [])
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
                if any(p.get('id') == state['pod_id'] for p in pods()):
                    raise RuntimeError('Pod still present after delete')
                state['status'] = 'deleted'
                state['deleted_utc'] = utcnow().isoformat()
                state['provider_absence_confirmed'] = True
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
    gpu = ssh(pod, 'nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader',
              timeout=30)
    if gpu.returncode or len(gpu.stdout.strip().splitlines()) != 1 or 'A40' not in gpu.stdout:
        raise RuntimeError('Expected exactly one NVIDIA A40 on remote pod')
    path, state, _ = live_pod()
    if 'setup_utc' in state:
        state['gpu_identity'] = gpu.stdout.strip()
        write_state(path, state)
        return {'setup': 'complete', 'freeze_sha256': state['setup_freeze_sha256'],
                'gpu_identity': state['gpu_identity']}
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
    (path.parent / 'setup.log').write_text(result.stdout[-30000:] + result.stderr[-10000:])
    if result.returncode:
        raise RuntimeError('Remote CPU environment setup failed; inspect setup.log')
    state['setup_utc'] = utcnow().isoformat()
    state['setup_freeze_sha256'] = hashlib.sha256(result.stdout.encode()).hexdigest()
    state['gpu_identity'] = gpu.stdout.strip()
    write_state(path, state)
    return {'setup': 'complete', 'freeze_sha256': state['setup_freeze_sha256']}


def run_task(task, repo, commit, command, secret_names=(), outputs=()):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('Invalid task identifier')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Full 40-character commit required')
    allowed = {'wandb': ('WANDB_API_KEY', 'WEIGHTS_AND_BIASES_API_KEY'),
               'hf': ('HF_TOKEN', 'HUGGINGFACE_API_KEY'),
               'kaggle': ('KAGGLE_API_TOKEN', 'KAGGLE_API_TOKEN')}
    if any(name not in allowed for name in secret_names):
        raise ValueError('Unknown secret alias')
    for output in outputs:
        value = Path(output)
        if value.is_absolute() or '..' in value.parts or not value.parts:
            raise ValueError('Outputs must be relative to the task source checkout')
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
    source_allow = {'ARCHITECTURE.md', 'CITATION.cff', 'CONTRIBUTING.md', 'LICENSE',
                    'NOTICE', 'PROJECT_BRIEF.md', 'README.md', 'configs', 'docs',
                    'notebooks', 'de01.ipynb', 'pyproject.toml', 'research', 'scripts', 'src',
                    'tests', 'uv.lock', 'requirements-cpu.lock',
                    'requirements-colab-candidate.txt', 'requirements-behavioral.txt'}
    tracked = subprocess.run(['git', '-C', str(repo), 'ls-tree', '--name-only', commit],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    paths = sorted(source_allow.intersection(tracked))
    with archive.open('wb') as stream:
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', commit, *paths],
                       check=True, stdout=stream)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            name = Path(member.name)
            lowered = [part.lower() for part in name.parts]
            if (name.is_absolute() or '..' in name.parts or member.issym() or member.islnk()
                    or any(part.startswith('.env') or
                           any(term in part for term in ('credential', 'secret', 'private_key',
                                                          'kaggle.json', 'token.json')) or
                           part in {'.kaggle', '.wandb',
                                    'reports', 'dist', 'checkpoints', 'models'}
                           for part in lowered)
                    or name.suffix.lower() in {'.safetensors', '.pt', '.pth', '.ckpt',
                                               '.parquet', '.arrow'}):
                raise RuntimeError('Archive contains forbidden source member')
            if not (member.isfile() or member.isdir()):
                raise RuntimeError('Archive contains unsupported member type')
    member_count = len(members)
    source_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    remote = f"{CONFIG['remote_root']}/tasks/{task}"
    base, scp, target = ssh_args(pod)
    subprocess.run([*base, f'mkdir -p {shlex.quote(remote)}/src'], check=True)
    subprocess.run([*scp, str(archive), f'{target}:{remote}/source.tar'], check=True)
    remote_hash = ssh(pod, f'sha256sum {shlex.quote(remote)}/source.tar', timeout=30)
    if remote_hash.returncode or remote_hash.stdout.split()[0] != source_sha:
        raise RuntimeError('Remote source archive hash mismatch')
    wrapped = (f'set -o pipefail; cd {shlex.quote(remote)}/src; '
               f'tar -xf {shlex.quote(remote)}/source.tar || exit 90; '
               f'export PATH={shlex.quote(CONFIG["remote_root"] + "/cpu-venv/bin")}:$PATH; '
               f'export RUNPOD_POD_ID={shlex.quote(state["pod_id"])}; '
               f'export ORDIVO_SOURCE_COMMIT={commit}; '
               f'export ORDIVO_SOURCE_SHA256={source_sha}; '
               f'({command}) > {shlex.quote(remote)}/stdout.log 2>&1; '
               f'code=$?; echo "$code" > {shlex.quote(remote)}/exit_code; exit "$code"')
    # Secret values travel only over SSH stdin, never command argv or a remote file.
    secret_values = {allowed[name][0]: load_secret(allowed[name][1]) for name in secret_names}
    exports = ''.join(f'export {key}={shlex.quote(value)}\n'
                      for key, value in secret_values.items())
    remote_script = exports + wrapped + '\n'
    receipt = {'task': task, 'commit': commit, 'command': command,
               'source_sha256': source_sha, 'remote_source_sha256': remote_hash.stdout.split()[0],
               'source_member_count': member_count, 'started_utc': utcnow().isoformat(),
               'remote_dir': remote, 'pod_id': state['pod_id'],
               'gpu_identity': state.get('gpu_identity'), 'secret_aliases': list(secret_names),
               'outputs': list(outputs), 'status': 'running'}
    write_once(task_dir / 'receipt.json', receipt)
    try:
        result = subprocess.run([*base, 'bash -s'], input=remote_script,
                                capture_output=True, text=True, timeout=3600)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # The SSH process may have died before an EXIT trap; no secret file exists.
        write_once(task_dir / 'interrupted.json', {'status': 'incomplete',
                    'finished_utc': utcnow().isoformat()})
        raise
    receipt['finished_utc'] = utcnow().isoformat()
    receipt['returncode'] = result.returncode
    receipt['status'] = 'command_finished'
    write_once(task_dir / 'finished.json', receipt)
    ssh_log = result.stdout[-10000:] + result.stderr[-10000:]
    for value in secret_values.values():
        ssh_log = ssh_log.replace(value, '[REDACTED]')
    (task_dir / 'ssh.log').write_text(ssh_log)
    fetch_task(task, secret_values=secret_values)
    if outputs:
        output_root = task_dir / 'outputs'
        output_root.mkdir(mode=0o700)
        pack_code = '''import hashlib,os,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
archive=pathlib.Path(sys.argv[2])
paths=sys.argv[3:]
with tarfile.open(archive,'w') as out:
 for name in paths:
  p=root/name
  if not p.exists(): raise RuntimeError('requested output missing')
  entries=[p] if p.is_file() else [p,*p.rglob('*')]
  for item in entries:
   if item.is_symlink() or not (item.is_file() or item.is_dir()): raise RuntimeError('unsafe output type')
   resolved=item.resolve(strict=True)
   if not resolved.is_relative_to(root): raise RuntimeError('output escapes checkout')
   out.add(item,arcname=str(item.relative_to(root)),recursive=False)
'''
        pack_cmd = shlex.join(['python', '-c', pack_code, f'{remote}/src',
                               f'{remote}/outputs.tar', *outputs])
        packed = ssh(pod, pack_cmd, timeout=120)
        if packed.returncode:
            raise RuntimeError('Remote output rejected or packaging failed')
        local_tar = task_dir / 'outputs.tar'
        subprocess.run([*scp, f'{target}:{remote}/outputs.tar', str(local_tar)],
                       check=True, capture_output=True)
        hashes = {}
        with tarfile.open(local_tar) as archive_out:
            for member in archive_out.getmembers():
                name = Path(member.name)
                if (name.is_absolute() or '..' in name.parts or
                        not (member.isfile() or member.isdir())):
                    raise RuntimeError('Unsafe output archive member')
                local = output_root / name
                if any(parent.is_symlink() for parent in [local, *local.parents]):
                    raise RuntimeError('Local output symlink detected')
                if member.isdir():
                    local.mkdir(parents=True, exist_ok=True)
                else:
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with archive_out.extractfile(member) as source, local.open('xb') as dest:
                        dest.write(source.read())
                    hashes[str(name)] = hashlib.sha256(local.read_bytes()).hexdigest()
        write_once(task_dir / 'outputs.json', {'sha256': hashes,
                   'archive_sha256': hashlib.sha256(local_tar.read_bytes()).hexdigest()})
    return {key: receipt[key] for key in ('task', 'commit', 'returncode', 'source_sha256')}


def fetch_task(task, *, secret_values=None):
    path, _, pod = live_pod()
    task_dir = path.parent / 'tasks' / task
    if not task_dir.exists():
        raise RuntimeError('Unknown task')
    remote = f"{CONFIG['remote_root']}/tasks/{task}"
    _, scp, target = ssh_args(pod)
    if secret_values is None:
        aliases = json.loads((task_dir / 'receipt.json').read_text()).get('secret_aliases', [])
        mapping = {'wandb': 'WEIGHTS_AND_BIASES_API_KEY', 'hf': 'HUGGINGFACE_API_KEY',
                   'kaggle': 'KAGGLE_API_TOKEN'}
        secret_values = {x: load_secret(mapping[x]) for x in aliases}
    base, _, _ = ssh_args(pod)
    received = {}
    prior_path = task_dir / 'fetched.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    for filename in ('stdout.log', 'exit_code'):
        fetched = subprocess.run([*base, f'cat {shlex.quote(remote)}/{filename}'],
                                 check=True, capture_output=True)
        content = fetched.stdout
        for value in secret_values.values():
            content = content.replace(value.encode(), b'[REDACTED]')
        digest = hashlib.sha256(content).hexdigest()
        if prior:
            expected = prior['log_sha256' if filename == 'stdout.log' else 'exit_code_sha256']
            if digest != expected or hashlib.sha256((task_dir / filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Fetched evidence changed locally or remotely')
        else:
            with (task_dir / filename).open('xb') as dest:
                dest.write(content)
        received[filename] = digest
    result = {'task': task, 'log_sha256': received['stdout.log'],
              'exit_code_sha256': received['exit_code']}
    if not prior:
        write_once(prior_path, result)
    return result


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
        for attempt in range(12):
            if not any(p.get('id') == pod_id for p in pods()):
                break
            time.sleep(5)
        else:
            raise RuntimeError('Provider still lists pod after delete; receipt remains active')
    state['status'] = 'deleted'
    state['deleted_utc'] = utcnow().isoformat()
    state['provider_absence_confirmed'] = True
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
    parser.add_argument('--with-secret', action='append', choices=('wandb', 'hf', 'kaggle'),
                        default=[])
    parser.add_argument('--output', action='append', default=[])
    args = parser.parse_args()
    if args.reap:
        reap(Path(args.reap))
        return
    if args.run:
        if not all((args.repo, args.commit, args.command)):
            parser.error('--run requires --repo, --commit and --command')
        result = run_task(args.run, args.repo, args.commit, args.command,
                          args.with_secret, args.output)
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
