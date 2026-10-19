"""Publish sanitized Ordivo v2 operation evidence and verify W&B cloud readback.

This script runs on the approved remote pod. It reads aggregate/manifest JSON only;
raw rows, per-query files, model weights and credentials never become artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time


def load_metadata(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError('Expected JSON object metadata')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swiss-aggregate', type=Path, required=True)
    parser.add_argument('--swiss-manifest', type=Path, required=True)
    parser.add_argument('--esci-manifest', type=Path, required=True)
    parser.add_argument('--hourly-usd', type=float, required=True)
    parser.add_argument('--baseline-tests', type=int, required=True)
    parser.add_argument('--integrated-tests', type=int, required=True)
    parser.add_argument('--data-tests', type=int, required=True)
    parser.add_argument('--release-tests', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = os.environ.get('ORDIVO_SOURCE_COMMIT', '')
    archive = os.environ.get('ORDIVO_SOURCE_SHA256', '')
    if not re.fullmatch('[0-9a-f]{40}', commit) or not re.fullmatch('[0-9a-f]{64}', archive):
        raise RuntimeError('Controller-verified source identity is required')
    if not os.environ.get('RUNPOD_POD_ID') or not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Scoped remote pod and W&B credentials are required')
    if args.hourly_usd > 0.59:
        raise ValueError('Hourly cost cap exceeded')
    if args.output.exists():
        raise FileExistsError(args.output)
    swiss = load_metadata(args.swiss_aggregate)
    swiss_manifest = load_metadata(args.swiss_manifest)
    esci_manifest = load_metadata(args.esci_manifest)
    if swiss['source'] != 'swissmetro' or swiss_manifest['episode_count'] != 10719:
        raise ValueError('Swissmetro evidence mismatch')
    if esci_manifest['episode_count'] != 48300:
        raise ValueError('ESCI evidence mismatch')
    policy = next(iter(swiss['metrics']['conditional_logit']['by_source_task_policy'].values()))
    conditional_accuracy = policy['macro']['choice_accuracy@1']['value']
    conditional_ndcg = policy['macro']['ndcg@10']['value']
    if swiss['metrics']['conditional_logit']['scored_queries'] != 1044:
        raise ValueError('Swissmetro scored-query count mismatch')
    import wandb

    config = {'experiment_family': 'ordivo-v2-behavioral',
              'telemetry_scope': 'sanitized_aggregate_only',
              'controller_commit': commit, 'controller_archive_sha256': archive,
              'swiss_code_revision': swiss['code_revision'],
              'swiss_manifest_sha256': swiss['manifest_sha256'],
              'swiss_source_sha256': swiss['source_sha256'],
              'swiss_split_sha256': swiss_manifest['split_sha256'],
              'esci_source_sha256': esci_manifest['source_sha256'],
              'esci_split_sha256': esci_manifest['split_sha256'],
              'pod_gpu': 'NVIDIA A40', 'pod_hourly_usd': args.hourly_usd,
              'expedia_status': 'official_download_http_403',
              'expedia_access_scope': 'file_inventory_only'}
    run = wandb.init(entity='black-order', project='liquid-mcdm-lab',
                     job_type='ordivo-v2-evidence',
                     tags=['ordivo-v2-behavioral', 'remote-source-bound'],
                     config=config, mode='online')
    metrics = {'tests/base_passed': args.baseline_tests,
               'tests/integrated_passed': args.integrated_tests,
               'tests/data_passed': args.data_tests,
               'tests/release_passed': args.release_tests,
               'tests/skipped': 0,
               'data/swissmetro_groups': swiss_manifest['episode_count'],
               'data/esci_groups': esci_manifest['episode_count'],
               'baseline/swissmetro_test_queries': 1044,
               'baseline/swissmetro_conditional_logit_accuracy_at_1': conditional_accuracy,
               'baseline/swissmetro_conditional_logit_ndcg_at_10': conditional_ndcg,
               'failures/expedia_download_http_403': 1}
    run.log(metrics)
    run_id = run.id
    run_path = f'black-order/liquid-mcdm-lab/{run_id}'
    run.finish()
    api = wandb.Api()
    for _ in range(12):
        cloud = api.run(run_path)
        if cloud.summary.get('baseline/swissmetro_test_queries') == 1044:
            break
        time.sleep(5)
    else:
        raise RuntimeError('W&B cloud summary did not reflect logged evidence')
    if cloud.summary.get('failures/expedia_download_http_403') != 1:
        raise RuntimeError('W&B cloud failure record missing')
    receipt = {'run_path': run_path, 'run_url': cloud.url,
               'cloud_readback_confirmed': True, 'source_commit': commit,
               'source_archive_sha256': archive,
               'cloud_summary': {key: cloud.summary.get(key) for key in metrics}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'run_path': run_path, 'run_url': cloud.url,
                      'cloud_readback_confirmed': True}, sort_keys=True))


if __name__ == '__main__':
    main()
