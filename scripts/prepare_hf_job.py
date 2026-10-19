"""Small control-plane tool: package locally; all installations/workloads stay remote.

Default prepares source only. --preflight performs read-only account/hardware checks.
--submit creates private storage, uploads source and spends money: explicit approval required.
Uses the already installed HF client for API operations; never installs a client.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from package_colab import ROOT, package, source_paths

STAGE = ROOT/'dist/hf-job'
CONFIG = ROOT/'configs/hf_jobs_smoke.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {str(path.relative_to(ROOT)): digest(path) for path in source_paths()}


def prepare():
    STAGE.mkdir(parents=True, exist_ok=True)
    if (STAGE/'submission.json').exists():
        raise RuntimeError('A submission receipt exists. Preserve it before preparing another run.')
    archive = package(STAGE/'source.zip')
    shutil.copyfile(ROOT/'scripts/hf_job_bootstrap.py', STAGE/'hf_job_bootstrap.py')
    with zipfile.ZipFile(archive) as source:
        files = {name.removeprefix('decision-engines/'):
                 hashlib.sha256(source.read(name)).hexdigest() for name in source.namelist()}
        config = json.loads(source.read('decision-engines/configs/hf_jobs_smoke.json'))
    if (files != source_identity() or config != json.loads(CONFIG.read_text())
            or files['scripts/hf_job_bootstrap.py'] != digest(STAGE/'hf_job_bootstrap.py')):
        raise RuntimeError('Source changed during packaging; re-prepare a consistent snapshot')
    receipt = {'config': config, 'source_sha256': digest(archive),
               'bootstrap_sha256': digest(STAGE/'hf_job_bootstrap.py'),
               'source_files': files, 'archive_bytes': archive.stat().st_size,
               'prepared_utc': datetime.now(timezone.utc).isoformat()}
    (STAGE/'prepared.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'source_files'}, indent=2))
    print('Prepared only: no credentials loaded, no upload, no job submitted.')


def load_token():
    names = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN',
             'HF_API_KEY', 'HUGGINGFACE_API_KEY', 'HUGGING_FACE_API_KEY')
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    path = ROOT/'.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            key, separator, value = line.removeprefix('export ').partition('=')
            if separator and key.strip() in names:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                if value:
                    return value
    raise RuntimeError('No recognized HF token. Set HF_TOKEN securely; never place it in source.')


def preflight():
    from huggingface_hub import HfApi
    api = HfApi(token=load_token())
    account = api.whoami()
    config = json.loads(CONFIG.read_text())
    if account['name'] != config['namespace']:
        raise RuntimeError('Authenticated account differs from the reviewed job namespace')
    hardware = next((item for item in api.list_jobs_hardware()
                     if item.name == config['flavor']), None)
    if hardware is None or hardware.unit_label != 'minute':
        raise RuntimeError('Selected hardware or expected billing unit is unavailable')
    estimated = hardware.unit_cost_usd * ((config['timeout_seconds'] + 59) // 60)
    if estimated > 0.07:
        raise RuntimeError('Current compute estimate exceeds the reviewed $0.07 limit')
    summary = {'namespace': account['name'], 'flavor': hardware.name,
               'compute_usd_per_minute': hardware.unit_cost_usd,
               'timeout_seconds': config['timeout_seconds'], 'compute_estimate_usd': estimated,
               'credits_and_job_write_permission': 'not established by identity lookup',
               'storage_charges': 'separate from compute'}
    print(json.dumps(summary, indent=2))
    return api, account['name'], summary


def submit():
    from huggingface_hub import Volume
    receipt = json.loads((STAGE/'prepared.json').read_text())
    if (STAGE/'submission.json').exists():
        raise RuntimeError('Submission already attempted; inspect the receipt and account before retrying')
    if (receipt['source_sha256'] != digest(STAGE/'source.zip')
            or receipt['bootstrap_sha256'] != digest(STAGE/'hf_job_bootstrap.py')
            or receipt['source_files'] != source_identity()
            or receipt['config'] != json.loads(CONFIG.read_text())):
        raise RuntimeError('Prepared source is stale or changed; re-prepare before approval')
    api, namespace, price = preflight()
    config = receipt['config']
    bucket = namespace+'/liquid-mcdm-smoke-'+datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    state = {'status': 'creating_private_bucket', 'bucket': bucket,
             'source_sha256': receipt['source_sha256'], 'price': price}

    def checkpoint():
        (STAGE/'submission.json').write_text(json.dumps(state, indent=2)+'\n')

    # Atomically claim the attempt: two simultaneous submitters cannot both spend.
    with (STAGE/'submission.json').open('x') as stream:
        stream.write(json.dumps(state, indent=2)+'\n')
    # No exist_ok: never overwrite an existing bucket or its visibility.
    api.create_bucket(bucket, private=True, exist_ok=False)
    state['status'] = 'uploading_source'
    checkpoint()
    api.batch_bucket_files(bucket, add=[
        (STAGE/'source.zip', 'source.zip'),
        (STAGE/'hf_job_bootstrap.py', 'hf_job_bootstrap.py'),
        (STAGE/'prepared.json', 'prepared.json'),
    ])
    state['status'] = 'submitting_job'
    checkpoint()
    # No automatic submission retries. An ambiguous API failure needs account reconciliation.
    job = api.run_job(
        image=config['image'], flavor=config['flavor'], timeout=config['timeout_seconds'],
        command=['python', '/job-store/hf_job_bootstrap.py'], namespace=namespace,
        volumes=[Volume(type='bucket', source=bucket, mount_path='/job-store')],
        env={'MCDM_SOURCE_SHA256': receipt['source_sha256'], 'MCDM_JOB_IMAGE': config['image'],
             'WANDB_MODE': 'offline', 'HF_HUB_DISABLE_TELEMETRY': '1'},
        labels={'project': 'liquid-mcdm', 'stage': 'acceptance'},
    )
    state.update(status='submitted', job_id=job.id, job_url=job.url,
                 result_path=f'hf://buckets/{bucket}/results/{job.id}')
    checkpoint()
    print(json.dumps(state, indent=2))
    print('Closing the log stream does not cancel a job. Cancel explicitly if needed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--submit', action='store_true', help='PAID: requires prior user approval')
    args = parser.parse_args()
    if args.submit:
        submit()
    elif args.preflight:
        preflight()
    else:
        prepare()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Provider exceptions can include request details: never print their raw payloads.
        print(f'Control step failed ({type(exc).__name__}). No automatic retry.', file=sys.stderr)
        if isinstance(exc, (RuntimeError, FileNotFoundError)):
            print(str(exc), file=sys.stderr)
        sys.exit(1)
