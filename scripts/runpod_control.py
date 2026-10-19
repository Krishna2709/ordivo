"""Runpod control plane. Source packaging is local; every workload stays remote."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

from package_colab import ROOT, package, source_paths

API = 'https://rest.runpod.io/v1'
CATALOG = 'https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD'
STAGE = ROOT / 'dist' / 'runpod'
CONFIG = ROOT / 'configs' / 'runpod_d1_overfit.json'
USER_AGENT = 'runpodctl/1.0 liquid-mcdm-control/1'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}


def load_api_key():
    if os.environ.get('RUNPOD_API_KEY'):
        return os.environ['RUNPOD_API_KEY']
    path = ROOT / '.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            key, separator, value = line.removeprefix('export ').partition('=')
            if separator and key.strip() == 'RUNPOD_API_KEY':
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                if value:
                    return value
    raise RuntimeError('RUNPOD_API_KEY is absent; it is read only for control-plane calls')


def request(method, path, *, body=None, absolute=False, token=None):
    url = path if absolute else API + path
    headers = {'Authorization': f'Bearer {token or load_api_key()}',
               'User-Agent': USER_AGENT, 'Accept': 'application/json'}
    data = None
    if body is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            payload = response.read()
            return json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        # Provider response bodies can echo request context. Do not print them.
        raise RuntimeError(f'Runpod API {method} {path} failed with HTTP {exc.code}') from None


def prepare():
    STAGE.mkdir(parents=True, exist_ok=True)
    if (STAGE / 'launch.json').exists():
        raise RuntimeError('A launch receipt exists; preserve/reconcile it before another launch')
    archive = package(STAGE / 'source.zip')
    shutil.copyfile(ROOT / 'scripts' / 'runpod_bootstrap.py', STAGE / 'runpod_bootstrap.py')
    with zipfile.ZipFile(archive) as source:
        files = {name.removeprefix('decision-engines/'):
                 hashlib.sha256(source.read(name)).hexdigest() for name in source.namelist()}
    if files != source_identity() or files['scripts/runpod_bootstrap.py'] != digest(
            STAGE / 'runpod_bootstrap.py'):
        raise RuntimeError('Source changed during packaging; prepare a consistent snapshot again')
    config = json.loads(CONFIG.read_text())
    receipt = {'config': config, 'source_sha256': digest(archive),
               'bootstrap_sha256': digest(STAGE / 'runpod_bootstrap.py'),
               'source_files': files, 'archive_bytes': archive.stat().st_size,
               'prepared_utc': datetime.now(timezone.utc).isoformat()}
    write_json(STAGE / 'prepared.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'source_files'}, indent=2))
    print('Prepared only: no API key loaded, no upload, and no provider resource created.')


def _items(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ('data', 'items', 'gpus'):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def _a40_catalog_summary(catalog, config):
    target = config['gpu_type_id'].lower()
    gpu = next((item for item in _items(catalog)
                if str(item.get('id') or item.get('gpuTypeId') or item.get('displayName', '')).lower()
                == target), None)
    if gpu is None:
        raise RuntimeError('The selected A40 is absent from the current Runpod catalog')
    secure_price = gpu.get('securePrice')
    if secure_price is None:
        secure_price = gpu.get('secureCloudPrice')
    if secure_price is not None and float(secure_price) > config['max_hourly_usd']:
        raise RuntimeError('Current secure A40 price exceeds the reviewed hourly ceiling')
    return {'gpu': config['gpu_type_id'], 'secure_price': secure_price,
            'availability': gpu.get('availability') or gpu.get('available')}


def preflight():
    token = load_api_key()
    config = json.loads(CONFIG.read_text())
    pods = _items(request('GET', '/pods', token=token))
    volumes = _items(request('GET', '/networkvolumes', token=token))
    catalog = request('GET', CATALOG, absolute=True, token=token)
    prefix = config['name_prefix']
    collisions = [item.get('id') for item in [*pods, *volumes]
                  if str(item.get('name', '')).startswith(prefix)]
    if collisions:
        raise RuntimeError(f'Existing project resources require reconciliation: {collisions}')
    selected = _a40_catalog_summary(catalog, config)
    summary = {'existing_pods': len(pods), 'existing_network_volumes': len(volumes),
               'selected': selected, 'max_runtime_seconds': config['max_runtime_seconds'],
               'reviewed_compute_ceiling_usd':
                   config['max_hourly_usd'] * config['max_runtime_seconds'] / 3600,
               'network_volume_gb': config['network_volume_gb']}
    print(json.dumps(summary, indent=2))
    return summary


def start_reaper(config, state):
    reaper_state = STAGE / 'reaper.json'
    reaper_state.unlink(missing_ok=True)
    reaper_log = (STAGE / 'reaper.log').open('w')
    try:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / 'scripts' / 'runpod_reaper.py'),
             '--deadline-seconds', str(config['max_runtime_seconds'])],
            cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
            stdout=reaper_log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        reaper_log.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Runpod reaper exited before arming')
        if reaper_state.exists():
            armed = json.loads(reaper_state.read_text())
            if armed.get('status') == 'armed' and armed.get('pod_id') == state['pod_id']:
                return process.pid
        time.sleep(.1)
    process.terminate()
    raise RuntimeError('Runpod reaper did not acknowledge its deadline')


def launch():
    receipt_path = STAGE / 'prepared.json'
    if not receipt_path.exists():
        raise RuntimeError('Prepare the immutable source snapshot before launch')
    receipt = json.loads(receipt_path.read_text())
    if (STAGE / 'launch.json').exists():
        raise RuntimeError('Launch was already attempted; inspect the receipt before retrying')
    if (receipt['source_sha256'] != digest(STAGE / 'source.zip')
            or receipt['bootstrap_sha256'] != digest(STAGE / 'runpod_bootstrap.py')
            or receipt['source_files'] != source_identity()
            or receipt['config'] != json.loads(CONFIG.read_text())):
        raise RuntimeError('Prepared source is stale or changed')
    preflight()
    token = load_api_key()
    config = receipt['config']
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = f"{config['name_prefix']}-{run_id.lower()}"
    state = {'status': 'creating_network_volume', 'run_id': run_id, 'name': name,
             'source_sha256': receipt['source_sha256'], 'pod_id': None, 'network_volume_id': None,
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'max_runtime_seconds': config['max_runtime_seconds']}
    with (STAGE / 'launch.json').open('x') as stream:
        stream.write(json.dumps(state, indent=2, sort_keys=True) + '\n')

    volume = request('POST', '/networkvolumes', token=token, body={
        'name': name, 'size': config['network_volume_gb'],
        'dataCenterId': config['data_center_ids'][0]})
    state['network_volume_id'] = volume['id']
    state['status'] = 'creating_pod'
    write_json(STAGE / 'launch.json', state)
    pod = request('POST', '/pods', token=token, body={
        'name': name, 'cloudType': config['cloud_type'], 'computeType': 'GPU',
        'gpuTypeIds': [config['gpu_type_id']], 'gpuTypePriority': 'custom',
        'gpuCount': config['gpu_count'], 'dataCenterIds': config['data_center_ids'],
        'dataCenterPriority': 'custom', 'imageName': config['image_name'],
        'containerDiskInGb': config['container_disk_gb'], 'interruptible': False,
        'networkVolumeId': state['network_volume_id'],
        'volumeMountPath': config['volume_mount_path'], 'ports': config['ports'],
        'supportPublicIp': True,
        'env': {'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_MODE': config['wandb_mode']}})
    state['pod_id'] = pod['id']
    state['status'] = 'pod_created'
    state['cost_per_hour'] = pod.get('costPerHr')
    write_json(STAGE / 'launch.json', state)
    if state['cost_per_hour'] is None or float(state['cost_per_hour']) > config['max_hourly_usd']:
        state['status'] = 'price_guard_cleanup'
        write_json(STAGE / 'launch.json', state)
        try:
            request('POST', f"/pods/{state['pod_id']}/stop", token=token)
        finally:
            request('DELETE', f"/pods/{state['pod_id']}", token=token)
        state.update(status='pod_deleted_price_guard',
                     pod_deleted_utc=datetime.now(timezone.utc).isoformat(),
                     network_volume_preserved=True)
        write_json(STAGE / 'launch.json', state)
        raise RuntimeError('Created pod price was absent or above the reviewed hourly ceiling')
    try:
        reaper_pid = start_reaper(config, state)
    except BaseException:
        try:
            request('POST', f"/pods/{state['pod_id']}/stop", token=token)
        finally:
            request('DELETE', f"/pods/{state['pod_id']}", token=token)
        state.update(status='pod_deleted_reaper_start_failure',
                     pod_deleted_utc=datetime.now(timezone.utc).isoformat(),
                     network_volume_preserved=True)
        write_json(STAGE / 'launch.json', state)
        raise
    state.update(reaper_pid=reaper_pid, reaper_deadline_seconds=config['max_runtime_seconds'])
    write_json(STAGE / 'launch.json', state)
    print(json.dumps(state, indent=2))


def receipt():
    path = STAGE / 'launch.json'
    if not path.exists():
        raise RuntimeError('No Runpod launch receipt exists')
    return json.loads(path.read_text())


def get_pod(state=None):
    state = state or receipt()
    if not state.get('pod_id'):
        raise RuntimeError('Launch receipt has no pod ID; reconcile the volume before retrying')
    return request('GET', f"/pods/{state['pod_id']}")


def ssh_target(pod):
    public_ip = pod.get('publicIp')
    mappings = pod.get('portMappings') or {}
    port = mappings.get('22') or mappings.get(22)
    if not public_ip or not port:
        raise RuntimeError('Pod SSH endpoint is not ready')
    if not re.fullmatch(r'[0-9.]+', public_ip) or not isinstance(port, int):
        raise RuntimeError('Unexpected SSH endpoint format')
    return public_ip, port


def ssh_base(pod):
    host, port = ssh_target(pod)
    known_hosts = STAGE / 'known_hosts'
    return ['ssh', '-i', str(Path.home() / '.ssh' / 'id_ed25519'), '-p', str(port),
            '-o', f'UserKnownHostsFile={known_hosts}', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ConnectTimeout=15', f'root@{host}']


def wait_ssh(timeout=900):
    state = receipt()
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        pod = get_pod(state)
        try:
            base = ssh_base(pod)
        except RuntimeError as exc:
            last = exc
            time.sleep(10)
            continue
        result = subprocess.run([*base, 'true'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return pod
        last = RuntimeError('SSH endpoint did not accept the configured account key')
        time.sleep(10)
    raise RuntimeError(f'Pod did not become SSH-ready within {timeout}s: {last}')


def start_remote():
    state = receipt()
    if state['status'] not in {'pod_created', 'ssh_ready'}:
        raise RuntimeError(f"Remote start is invalid from receipt state {state['status']}")
    pod = wait_ssh()
    prepared = json.loads((STAGE / 'prepared.json').read_text())
    if (prepared['source_sha256'] != digest(STAGE / 'source.zip')
            or prepared['bootstrap_sha256'] != digest(STAGE / 'runpod_bootstrap.py')
            or prepared['source_files'] != source_identity()
            or prepared['source_sha256'] != state['source_sha256']):
        raise RuntimeError('Staged source/bootstrap changed after the reviewed launch')
    state['status'] = 'ssh_ready'
    write_json(STAGE / 'launch.json', state)
    host, port = ssh_target(pod)
    incoming = f"/workspace/incoming/{state['run_id']}"
    base = ssh_base(pod)
    subprocess.run([*base, 'mkdir', '-p', incoming], check=True)
    common = ['scp', '-i', str(Path.home() / '.ssh' / 'id_ed25519'), '-P', str(port),
              '-o', f"UserKnownHostsFile={STAGE / 'known_hosts'}",
              '-o', 'StrictHostKeyChecking=accept-new']
    subprocess.run([*common, str(STAGE / 'source.zip'),
                    f'root@{host}:{incoming}/source.zip'], check=True)
    subprocess.run([*common, str(STAGE / 'runpod_bootstrap.py'),
                    f'root@{host}:{incoming}/runpod_bootstrap.py'], check=True)
    remote = ['env', f"RUNPOD_POD_ID={state['pod_id']}",
              f"RUNPOD_RUN_ID={state['run_id']}",
              f"MCDM_SOURCE_SHA256={state['source_sha256']}",
              'PYTHONUNBUFFERED=1', 'nohup', 'python', f'{incoming}/runpod_bootstrap.py']
    command = (f"{shlex.join(remote)} > {shlex.quote(incoming + '/driver.log')} 2>&1 "
               f"< /dev/null & echo $!")
    started = subprocess.run([*base, command], capture_output=True, text=True, check=True, timeout=30)
    pid = started.stdout.strip().splitlines()[-1]
    if not re.fullmatch(r'[0-9]+', pid):
        raise RuntimeError('Remote bootstrap did not return a process ID')
    state.update(status='remote_started', remote_pid=int(pid),
                 remote_started_utc=datetime.now(timezone.utc).isoformat())
    write_json(STAGE / 'launch.json', state)
    print(json.dumps(state, indent=2))


def retry_failed_remote():
    """Start a new immutable attempt on the same paid pod after a pre-training gate failure."""
    state = receipt()
    pod = wait_ssh()
    base = ssh_base(pod)
    old_result = f"/workspace/runs/{state['run_id']}/bootstrap.json"
    checked = subprocess.run([*base, f"cat {shlex.quote(old_result)}"],
                             capture_output=True, text=True, check=True, timeout=30)
    prior = json.loads(checked.stdout)
    if prior.get('status') != 'failed' or prior.get('training_started'):
        raise RuntimeError('Retry is allowed only after a terminal pre-training failure')
    if prior.get('cpu_acceptance_passed'):
        training_receipt = f"/workspace/runs/{state['run_id']}/training/run.json"
        inspected = subprocess.run(
            [*base, (f"if test -f {shlex.quote(training_receipt)}; then "
                     f"cat {shlex.quote(training_receipt)}; else echo null; fi")],
            capture_output=True, text=True, check=True, timeout=30)
        training_state = json.loads(inspected.stdout)
        if training_state is not None and training_state.get('training_started'):
            raise RuntimeError('Retry refused because optimizer training already started')

    archive = package(STAGE / 'source.zip')
    shutil.copyfile(ROOT / 'scripts' / 'runpod_bootstrap.py', STAGE / 'runpod_bootstrap.py')
    with zipfile.ZipFile(archive) as source:
        files = {name.removeprefix('decision-engines/'):
                 hashlib.sha256(source.read(name)).hexdigest() for name in source.namelist()}
    if files != source_identity() or files['scripts/runpod_bootstrap.py'] != digest(
            STAGE / 'runpod_bootstrap.py'):
        raise RuntimeError('Retry source changed during packaging')
    prepared = {'config': json.loads(CONFIG.read_text()), 'source_sha256': digest(archive),
                'bootstrap_sha256': digest(STAGE / 'runpod_bootstrap.py'),
                'source_files': files, 'archive_bytes': archive.stat().st_size,
                'prepared_utc': datetime.now(timezone.utc).isoformat(),
                'retry_of_run_id': state['run_id']}
    write_json(STAGE / 'prepared.json', prepared)
    previous = {'run_id': state['run_id'], 'source_sha256': state['source_sha256'],
                'terminal_status': prior['status'], 'error_type': prior.get('error_type'),
                'training_started': prior.get('training_started', False),
                'cpu_acceptance_passed': prior.get('cpu_acceptance_passed', False)}
    state.setdefault('attempt_history', []).append(previous)
    state.update(status='retry_prepared', run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                 source_sha256=prepared['source_sha256'], remote_pid=None)
    write_json(STAGE / 'launch.json', state)

    incoming = f"/workspace/incoming/{state['run_id']}"
    host, port = ssh_target(pod)
    subprocess.run([*base, 'mkdir', '-p', incoming], check=True)
    common = ['scp', '-i', str(Path.home() / '.ssh' / 'id_ed25519'), '-P', str(port),
              '-o', f"UserKnownHostsFile={STAGE / 'known_hosts'}",
              '-o', 'StrictHostKeyChecking=accept-new']
    subprocess.run([*common, str(STAGE / 'source.zip'),
                    f'root@{host}:{incoming}/source.zip'], check=True)
    subprocess.run([*common, str(STAGE / 'runpod_bootstrap.py'),
                    f'root@{host}:{incoming}/runpod_bootstrap.py'], check=True)
    remote = ['env', f"RUNPOD_POD_ID={state['pod_id']}", f"RUNPOD_RUN_ID={state['run_id']}",
              f"MCDM_SOURCE_SHA256={state['source_sha256']}", 'PYTHONUNBUFFERED=1',
              'nohup', 'python', f'{incoming}/runpod_bootstrap.py']
    command = (f"{shlex.join(remote)} > {shlex.quote(incoming + '/driver.log')} 2>&1 "
               f"< /dev/null & echo $!")
    started = subprocess.run([*base, command], capture_output=True, text=True, check=True, timeout=30)
    pid = started.stdout.strip().splitlines()[-1]
    if not re.fullmatch(r'[0-9]+', pid):
        raise RuntimeError('Retry bootstrap did not return a process ID')
    state.update(status='remote_started', remote_pid=int(pid),
                 remote_started_utc=datetime.now(timezone.utc).isoformat())
    write_json(STAGE / 'launch.json', state)
    print(json.dumps(state, indent=2))


def remote_status(*, tail=40):
    state = receipt()
    pod = get_pod(state)
    summary = {'receipt_status': state['status'], 'pod_id': state['pod_id'],
               'pod_status': pod.get('desiredStatus'), 'cost_per_hour': pod.get('costPerHr')}
    try:
        base = ssh_base(pod)
        result_path = f"/workspace/runs/{state['run_id']}/bootstrap.json"
        driver_path = f"/workspace/incoming/{state['run_id']}/driver.log"
        command = f"if test -f {shlex.quote(result_path)}; then cat {shlex.quote(result_path)}; fi"
        result = subprocess.run([*base, command], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            summary['bootstrap'] = json.loads(result.stdout)
        log = subprocess.run([*base, f"tail -n {int(tail)} {shlex.quote(driver_path)}"],
                             capture_output=True, text=True, timeout=30)
        summary['driver_log_tail'] = log.stdout[-12000:]
    except (RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        summary['remote_status_error'] = type(exc).__name__
    print(json.dumps(summary, indent=2))
    return summary


def cleanup_pod():
    state = receipt()
    if not state.get('pod_id'):
        raise RuntimeError('No recorded pod to delete')
    try:
        request('POST', f"/pods/{state['pod_id']}/stop")
    except RuntimeError:
        pass
    request('DELETE', f"/pods/{state['pod_id']}")
    state.update(status='pod_deleted', pod_deleted_utc=datetime.now(timezone.utc).isoformat(),
                 network_volume_preserved=True)
    write_json(STAGE / 'launch.json', state)
    if state.get('reaper_pid'):
        try:
            os.kill(int(state['reaper_pid']), 15)
        except ProcessLookupError:
            pass
    print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--launch', action='store_true')
    group.add_argument('--start-remote', action='store_true')
    group.add_argument('--retry-failed', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--cleanup-pod', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    elif args.launch:
        launch()
    elif args.start_remote:
        start_remote()
    elif args.retry_failed:
        retry_failed_remote()
    elif args.status:
        remote_status()
    else:
        cleanup_pod()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Runpod control failed ({type(exc).__name__}). No automatic retry.', file=sys.stderr)
        if isinstance(exc, (RuntimeError, FileNotFoundError)):
            print(str(exc), file=sys.stderr)
        sys.exit(1)
