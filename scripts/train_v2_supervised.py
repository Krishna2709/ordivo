"""Bounded shared d1 adapter pilot; remote execution after data/baseline review only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import yaml

from liquid_mcdm.runtime import require_remote_gpu
from liquid_mcdm.training.behavioral_rendering import render_behavioral
from liquid_mcdm.training.option_policy import option_logits

MODEL = 'LiquidAI/d1-3B'
REVISION = '051bcc464b01b9f92942b364d9586b0ef5912432'


def _load_source(spec):
    directory = Path(spec['directory'])
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if set(manifest['split_sha256']) != {'train', 'validation', 'calibration', 'test'}:
        raise ValueError('four-way frozen split manifest required')
    rows = {}
    for split in ('train', 'validation'):
        content = (directory / f'{split}.jsonl').read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['split_sha256'][split]:
            raise ValueError(f'{split} hash differs from source manifest')
        rows[split] = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not rows[split]:
            raise ValueError(f'empty {split} split')
    return rows, hashlib.sha256(manifest_bytes).hexdigest()


def _label(episode, ids, eligible, feature_policy):
    """One query-normalized target; all candidate identities stay outside prompts."""
    import torch

    task = episode['task_type']
    event = feature_policy.get('prediction_event')
    supervision = episode['supervision']
    if task == 'observed_choice' and event == 'conditional_booking_choice':
        positives = [i for i, id_ in enumerate(ids) if eligible[i] and
                     supervision['outcomes_by_id'][id_]['booking_bool']]
        if len(positives) != 1:
            return None
        return 'choice', torch.tensor(positives[0], dtype=torch.long), torch.tensor(eligible)
    if task == 'stated_choice':
        return 'choice', torch.tensor(ids.index(supervision['selected_id']), dtype=torch.long), torch.tensor(eligible)
    if task == 'observed_choice' and event == 'combined_click_booking_relevance':
        gains = [5. if supervision['outcomes_by_id'][id_]['booking_bool'] else
                 1. if supervision['outcomes_by_id'][id_]['click_bool'] else 0. for id_ in ids]
    elif task == 'graded_relevance':
        mapping = {'E': 1., 'C': .1, 'S': .01, 'I': 0.}
        gains = [mapping[supervision['grades_by_id'][id_]] if id_ in supervision['grades_by_id']
                 else 0. for id_ in ids]
    else:
        raise ValueError(f'unsupported supervised task/event: {task}/{event}')
    values = torch.tensor(gains, dtype=torch.float32)
    values[~torch.tensor(eligible)] = 0
    if values.sum() <= 0:
        return None
    judged = [id_ in supervision['grades_by_id'] for id_ in ids] if task == 'graded_relevance' else eligible
    return 'ranking', values / values.sum(), torch.tensor(judged)


def _examples(rows, tokenizer, policy, max_length):
    examples = []
    for episode in rows:
        rendered = render_behavioral(episode, tokenizer, policy, max_length=max_length)
        labeled = _label(episode, rendered['candidate_ids'], rendered['eligible'], policy)
        if labeled is None:
            continue
        examples.append((rendered, labeled))
    if not examples:
        raise ValueError('no eligible supervised groups')
    return examples


def _loss(model, tokenizer, item, device):
    import torch
    import torch.nn.functional as F

    row, (kind, target, loss_mask) = item
    inputs = tokenizer(row['prompt'], add_special_tokens=False, return_tensors='pt').to(device)
    output = model(**inputs, logits_to_keep=1).logits[0, -1]
    eligible = torch.tensor(row['eligible'], dtype=torch.bool, device=device)
    scores = option_logits(output, row['readout_groups'], eligible).float()
    scores = scores.masked_fill(~loss_mask.to(device), -torch.inf)
    if kind == 'choice':
        return F.cross_entropy(scores[None], target.to(device)[None])
    logp = F.log_softmax(scores, dim=-1)
    weights = target.to(device)
    safe_logp = torch.where(weights > 0, logp, torch.zeros_like(logp))
    return -(weights * safe_logp).sum()


def _validate(model, tokenizer, validation, device):
    import torch

    model.eval()
    totals = {}
    with torch.no_grad():
        for source, items in validation.items():
            totals[source] = sum(float(_loss(model, tokenizer, item, device)) for item in items) / len(items)
    model.train()
    return totals, sum(totals.values()) / len(totals)


def run(config_path, output):
    import torch
    import wandb
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    runtime = require_remote_gpu()
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    if config['model_id'] != MODEL or config['revision'] != REVISION:
        raise ValueError('unexpected base model identity')
    if not (1 <= config['max_steps'] <= 100 and 60 <= config['max_runtime_seconds'] <= 7200):
        raise ValueError('pilot step/runtime limits exceeded')
    if not (1 <= config['validation_interval'] <= config['max_steps']):
        raise ValueError('invalid validation interval')
    specs = config['sources']
    if len(specs) < 2 or len({spec['name'] for spec in specs}) != len(specs):
        raise ValueError('shared pilot needs at least two named domains')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    training, validation, provenance = {}, {}, {}
    for spec in specs:
        rows, manifest_hash = _load_source(spec)
        if (spec['feature_policy'].get('prediction_event') == 'conditional_booking_choice' and
                json.loads((Path(spec['directory']) / 'manifest.json').read_text()).get('candidate_pool_scope')
                != 'source_exposed_candidates'):
            raise ValueError('conditional booking requires exposed candidate pool')
        training[spec['name']] = _examples(rows['train'], tokenizer, spec['feature_policy'], config['max_length'])
        validation[spec['name']] = _examples(rows['validation'], tokenizer, spec['feature_policy'], config['max_length'])
        provenance[spec['name']] = {'manifest_sha256': manifest_hash,
                                    'feature_policy': spec['feature_policy'],
                                    'train_groups': len(rows['train']),
                                    'validation_groups': len(rows['validation']),
                                    'eligible_train_groups': len(training[spec['name']]),
                                    'eligible_validation_groups': len(validation[spec['name']])}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, device_map='auto')
    lora = config['lora']
    model = get_peft_model(base, LoraConfig(r=lora['r'], lora_alpha=lora['alpha'],
                                            lora_dropout=lora['dropout'], bias='none',
                                            task_type='CAUSAL_LM',
                                            target_modules=lora['target_modules'],
                                            exclude_modules='.*vision_tower.*'))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config['learning_rate'])
    probe_parameter = next(p for p in model.parameters() if p.requires_grad)
    observed_update = False
    run = wandb.init(project=config['wandb_project'], group='ordivo-v2-behavioral',
                     mode=config.get('wandb_mode', 'offline'),
                     config={'model_id': MODEL, 'revision': REVISION,
                             'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                             'sources': provenance, 'seed': config['seed']})
    cursors = {source: 0 for source in training}
    for items in training.values():
        random.shuffle(items)
    best = float('inf')
    best_step = 0
    best_probe_loss = None
    try:
        for step in range(1, config['max_steps'] + 1):
            if time.monotonic() - started > config['max_runtime_seconds']:
                break
            source = list(training)[(step - 1) % len(training)]
            items = training[source]
            position = cursors[source] % len(items)
            if position == 0 and cursors[source] > 0:
                random.shuffle(items)
            cursors[source] += 1
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, tokenizer, items[position], 'cuda')
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite supervised loss')
            loss.backward()
            before_update = probe_parameter.detach().clone()
            optimizer.step()
            observed_update |= bool(torch.count_nonzero(probe_parameter.detach() - before_update).item())
            if step % config['validation_interval'] == 0 or step == config['max_steps']:
                by_source, macro = _validate(model, tokenizer, validation, 'cuda')
                run.log({'train/query_loss': float(loss.detach()),
                         'validation/macro_source_loss': macro,
                         **{f'validation/{key}_query_loss': value for key, value in by_source.items()}},
                        step=step)
                if macro < best:
                    best, best_step = macro, step
                    model.eval()
                    with torch.no_grad():
                        best_probe_loss = float(_loss(model, tokenizer, validation[list(validation)[0]][0], 'cuda'))
                    model.train()
                    model.save_pretrained(output / 'best_adapter')
                    tokenizer.save_pretrained(output / 'best_adapter')
        if best_step == 0:
            raise RuntimeError('no validation checkpoint before runtime cap')
        if not observed_update:
            raise RuntimeError('no nonzero LoRA optimizer update')
        # Compare the saved best checkpoint, not the final step, with a fresh base.
        del model, base
        torch.cuda.empty_cache()
        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, revision=REVISION, dtype=dtype, device_map='auto')
        restored = PeftModel.from_pretrained(fresh, output / 'best_adapter')
        restored.eval()
        with torch.no_grad():
            after = float(_loss(restored, tokenizer, validation[list(validation)[0]][0], 'cuda'))
        reload_delta = abs(best_probe_loss - after)
        if reload_delta > 0.02:
            raise RuntimeError(f'best adapter reload changed probe loss: {reload_delta}')
        receipt = {'status': 'pilot_completed', 'claim_scope': 'preliminary bounded two-domain pilot',
                   'model_id': MODEL, 'revision': REVISION, 'runtime': runtime,
                   'seed': config['seed'], 'best_step': best_step,
                   'nonzero_lora_update': observed_update,
                   'best_validation_macro_source_loss': best,
                   'source_provenance': provenance,
                   'wall_seconds': time.monotonic() - started,
                   'reload_probe_loss': after,
                   'best_checkpoint_reload_probe_loss_delta': reload_delta}
        (output / 'pilot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        run.log({'selection/best_validation_macro_source_loss': best,
                 'selection/best_step': best_step})
        artifact = wandb.Artifact(f'ordivo-v2-supervised-{run.id}', type='model',
                                  metadata={'base_revision': REVISION, 'best_step': best_step})
        artifact.add_dir(str(output / 'best_adapter'))
        artifact.add_file(str(output / 'pilot_receipt.json'))
        run.log_artifact(artifact)
        return receipt
    finally:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
