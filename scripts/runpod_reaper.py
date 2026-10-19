"""Local lightweight cost backstop for a Runpod launched by runpod_control.py."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'dist' / 'runpod'
API = 'https://rest.runpod.io/v1'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


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
    raise RuntimeError('RUNPOD_API_KEY is absent from the project .env')


def request(method, path, token):
    req = urllib.request.Request(API + path, method=method, headers={
        'Authorization': f'Bearer {token}', 'User-Agent': 'runpodctl/1.0 liquid-mcdm-reaper/1'})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return 404
        raise RuntimeError(f'Runpod reaper {method} failed with HTTP {exc.code}') from None


def run(deadline_seconds):
    launch = json.loads((STAGE / 'launch.json').read_text())
    pod_id = launch.get('pod_id')
    if not pod_id:
        raise RuntimeError('Launch receipt has no pod ID')
    state = {'pod_id': pod_id, 'status': 'armed', 'deadline_seconds': deadline_seconds,
             'armed_utc': datetime.now(timezone.utc).isoformat()}
    write_json(STAGE / 'reaper.json', state)
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        time.sleep(min(30, max(1, deadline - time.monotonic())))
        try:
            current = json.loads((STAGE / 'launch.json').read_text())
            if current.get('status') in {'pod_deleted', 'pod_deleted_price_guard',
                                         'pod_deleted_reaper_start_failure'}:
                state.update(status='disarmed', finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(STAGE / 'reaper.json', state)
                return
        except (OSError, json.JSONDecodeError):
            # A concurrent receipt rewrite or transient filesystem error must
            # not disarm the cost backstop.
            continue
    token = load_api_key()
    status = None
    for attempt in range(20):
        try:
            try:
                request('POST', f'/pods/{pod_id}/stop', token)
            finally:
                status = request('DELETE', f'/pods/{pod_id}', token)
            break
        except (OSError, RuntimeError):
            if attempt == 19:
                raise
            time.sleep(15)
    state.update(status='deadline_deleted' if status != 404 else 'already_absent',
                 finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(STAGE / 'reaper.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-seconds', type=int, required=True)
    args = parser.parse_args()
    if not 60 <= args.deadline_seconds <= 7200:
        raise RuntimeError('Reaper deadline must be between 60 and 7200 seconds')
    run(args.deadline_seconds)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_json(STAGE / 'reaper.json', {
            'status': 'failed', 'error_type': type(exc).__name__,
            'finished_utc': datetime.now(timezone.utc).isoformat()})
        sys.exit(1)
