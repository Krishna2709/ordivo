"""Attach the final immutable evaluation receipt to an already-synced W&B run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def finalize(training_dir: Path, run_id: str, project: str, entity: str | None = None,
             data_dir: Path | None = None, adapter_dir: Path | None = None):
    import wandb

    training_dir = Path(training_dir).resolve()
    receipt = training_dir / 'run.json'
    state = json.loads(receipt.read_text())
    if not state.get('passed') or state.get('status') != 'passed':
        raise RuntimeError('Only a passed training receipt can finalize a W&B run')

    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume='allow',
        mode='online',
        job_type='evaluation-finalization',
    )
    run.summary['compatibility'] = state['compatibility']
    run.summary['overfit_outcome'] = state['overfit_outcome']
    run.summary['held_out_outcome'] = state['held_out_outcome']
    run.summary['reload_diagnostics'] = state['reload_diagnostics']
    run.summary['final_receipt_status'] = state['status']
    run.summary['final_receipt_passed'] = state['passed']
    run.summary['finished_utc'] = state['finished_utc']
    run.config.update({
        'model_revision': state['config']['revision'],
        'dataset_manifest_sha256': state['dataset_manifest_sha256'],
        'prompt_version': state['config']['prompt_version'],
        'runtime': state['runtime'],
    }, allow_val_change=True)

    predictions = training_dir / 'prediction_rows.json'
    if predictions.is_file():
        rows = json.loads(predictions.read_text())
        columns = list(rows[0])
        table = wandb.Table(
            columns=columns,
            data=[[row[column] for column in columns] for row in rows],
        )
        run.log({'evaluation/predictions': table})

    if data_dir is not None:
        data_dir = Path(data_dir).resolve()
        dataset_artifact = wandb.Artifact(
            f"liquid-mcdm-synthetic-{state['dataset_manifest_sha256'][:12]}",
            type='dataset',
            metadata={'manifest_sha256': state['dataset_manifest_sha256']},
        )
        for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
            path = data_dir / name
            if path.is_file():
                dataset_artifact.add_file(str(path), name=name)
        run.log_artifact(dataset_artifact, aliases=['latest', 'release'])

    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir).resolve()
        model_artifact = wandb.Artifact(
            f"d1-3b-mcdm-lora-{state['config']['experiment_name']}",
            type='model',
            metadata={
                'base_model': state['config']['model_id'],
                'revision': state['config']['revision'],
            },
        )
        model_artifact.add_dir(str(adapter_dir))
        run.log_artifact(model_artifact, aliases=['latest', 'release'])

    artifact = wandb.Artifact(
        f'd1-mcdm-final-evaluation-{run_id}',
        type='evaluation',
        metadata={
            'run_id': run_id,
            'dataset_manifest_sha256': state['dataset_manifest_sha256'],
            'model_revision': state['config']['revision'],
        },
    )
    for path in sorted(training_dir.glob('*.json')):
        artifact.add_file(str(path), name=path.name)
    freeze = training_dir / 'pip-freeze.txt'
    if freeze.is_file():
        artifact.add_file(str(freeze), name=freeze.name)
    run.log_artifact(artifact, aliases=['latest', 'passed'])
    run.log({'evaluation/finalized': 1})
    url = run.url
    run.finish()
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--project', default='liquid-mcdm-lab')
    parser.add_argument('--entity')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--adapter', type=Path)
    args = parser.parse_args()
    print(finalize(args.training, args.run_id, args.project, args.entity,
                   args.data, args.adapter))


if __name__ == '__main__':
    main()
