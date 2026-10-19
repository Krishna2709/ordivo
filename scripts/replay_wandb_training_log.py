"""Replay recorded TRL step metrics into an existing W&B release run."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


METRIC_KEYS = {
    'loss': 'train/loss',
    'grad_norm': 'train/grad_norm',
    'learning_rate': 'train/learning_rate',
    'entropy': 'train/entropy',
    'mean_token_accuracy': 'train/mean_token_accuracy',
    'num_tokens': 'train/num_tokens',
    'epoch': 'train/epoch',
}


def parse_training_records(path: Path):
    records = []
    for line in Path(path).read_text(errors='replace').splitlines():
        start = line.find("{'loss':")
        end = line.rfind('}')
        if start < 0 or end < start:
            continue
        try:
            raw = ast.literal_eval(line[start:end + 1])
        except (SyntaxError, ValueError):
            continue
        record = {}
        for source, target in METRIC_KEYS.items():
            if source in raw:
                record[target] = float(raw[source])
        if 'train/loss' in record and 'train/learning_rate' in record:
            record['train/global_step'] = len(records) + 1
            records.append(record)
    return records


def replay(log_path: Path, run_id: str, project: str, entity: str, expected_steps: int):
    import wandb

    records = parse_training_records(log_path)
    if len(records) != expected_steps:
        raise RuntimeError(
            f'Expected {expected_steps} trainer steps, parsed {len(records)}'
        )
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume='allow',
        mode='online',
        job_type='training-telemetry-replay',
    )
    for record in records:
        run.log(record)
    run.summary['replayed_training_steps'] = len(records)
    run.summary['telemetry_source'] = 'remote training.log'
    url = run.url
    run.finish()
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--project', default='liquid-mcdm-lab')
    parser.add_argument('--entity', required=True)
    parser.add_argument('--expected-steps', type=int, required=True)
    args = parser.parse_args()
    print(replay(args.log, args.run_id, args.project, args.entity,
                 args.expected_steps))


if __name__ == '__main__':
    main()
