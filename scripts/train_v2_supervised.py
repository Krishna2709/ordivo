"""Bounded shared d1 adapter pilot; remote execution after data/baseline review only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import time

import yaml

from liquid_mcdm.runtime import require_remote_gpu
from liquid_mcdm.training.behavioral_rendering import PROMPT_VERSION, render_behavioral
from liquid_mcdm.training.option_policy import option_logits

MODEL = 'LiquidAI/d1-3B'
REVISION = '051bcc464b01b9f92942b364d9586b0ef5912432'
SOURCE_CONTRACTS = {
    'expedia_personalized_sort_2013': ('expedia-2013-split-v2-calibration',
                                       'source_exposed_candidates', 'expedia-click-booking-v1'),
    'amazon_esci': ('esci-official-train-test-v2-calibration',
                    'source_task1_reduced_candidate_list', 'esci-task1-official-gain-v1'),
    'swissmetro': ('swissmetro-respondent-v2-calibration',
                   'three_survey_modes', 'swissmetro-choice-v1'),
}


def _source_identity():
    commit = os.environ.get('ORDIVO_SOURCE_COMMIT', '')
    digest = os.environ.get('ORDIVO_SOURCE_SHA256', '')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('ORDIVO_SOURCE_COMMIT requires controller-verified 40-hex archive commit')
    if not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('ORDIVO_SOURCE_SHA256 requires controller-verified 64-hex archive digest')
    return {'commit': commit, 'archive_sha256': digest}


def _select_unit_ids(units, *, limit, seed, source, split):
    if not isinstance(limit, int) or limit < 1:
        raise ValueError('positive group subset limit required')
    return frozenset(sorted(units, key=lambda unit: (
        hashlib.sha256(f'{seed}|{source}|{split}|{unit}'.encode()).digest(), unit))[:limit])


def _unit(row):
    return f"{row['source']}:{row['supervision'].get('respondent_id', row['group_id'])}"


def _validate_source_specs(specs):
    names = [spec.get('name') for spec in specs]
    sources = [spec.get('expected_source') for spec in specs]
    directories = [str(Path(spec['directory']).resolve()) for spec in specs]
    if len(specs) < 2 or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError('shared pilot needs distinct named domains')
    if len(set(sources)) != len(sources) or any(source not in SOURCE_CONTRACTS for source in sources):
        raise ValueError('shared pilot needs distinct canonical domains')
    if len(set(directories)) != len(directories):
        raise ValueError('duplicate source manifest directory')


def _load_source(spec, *, seed, max_train_groups, max_validation_groups, deadline,
                 max_train_episodes=10000, max_validation_episodes=1000):
    if (not isinstance(max_train_groups, int) or not 1 <= max_train_groups <= 1000 or
            not isinstance(max_validation_groups, int) or not 1 <= max_validation_groups <= 100):
        raise ValueError('pilot group caps are 1000 train and 100 validation per source')
    if (not isinstance(max_train_episodes, int) or not 1 <= max_train_episodes <= 10000 or
            not isinstance(max_validation_episodes, int) or not 1 <= max_validation_episodes <= 1000):
        raise ValueError('pilot episode caps are 10000 train and 1000 validation per source')
    directory = Path(spec['directory'])
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_source = spec['expected_source']
    expected = SOURCE_CONTRACTS.get(expected_source)
    if expected is None or (manifest.get('source'), manifest.get('split_policy_id'),
                            manifest.get('candidate_pool_scope'), manifest.get('target_policy_id')) != (
                                expected_source, *expected):
        raise ValueError('manifest source contract differs from pilot spec')
    if set(manifest['split_sha256']) != {'train', 'validation', 'calibration', 'test'}:
        raise ValueError('four-way frozen split manifest required')
    counts = {}
    with tempfile.TemporaryDirectory() as scratch:
        connection = sqlite3.connect(str(Path(scratch) / 'lineage.sqlite'))
        connection.execute('CREATE TABLE episodes (episode_id TEXT PRIMARY KEY)')
        connection.execute('CREATE TABLE units (unit TEXT PRIMARY KEY, split TEXT, selector BLOB)')
        try:
            for split in ('train', 'validation', 'calibration', 'test'):
                digest = hashlib.sha256()
                counts[split] = 0
                with (directory / f'{split}.jsonl').open('rb') as handle:
                    for line in handle:
                        if time.monotonic() > deadline:
                            raise TimeoutError('pilot time cap reached during source integrity pass')
                        digest.update(line)
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get('source') != expected_source:
                            raise ValueError('row source contract differs from pilot spec')
                        unit = _unit(row)
                        selector = hashlib.sha256(
                            f"{seed}|{spec['name']}|{split}|{unit}".encode()).digest()
                        try:
                            connection.execute('INSERT INTO episodes VALUES (?)', (row['episode_id'],))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError('duplicate episode across source splits') from exc
                        connection.execute('INSERT OR IGNORE INTO units VALUES (?,?,?)',
                                           (unit, split, selector))
                        other = connection.execute('SELECT split FROM units WHERE unit=?',
                                                   (unit,)).fetchone()[0]
                        if other != split:
                            raise ValueError(f'group overlap between {other} and {split}')
                        counts[split] += 1
                if digest.hexdigest() != manifest['split_sha256'][split]:
                    raise ValueError(f'{split} hash differs from source manifest')
                if split in ('train', 'validation') and not counts[split]:
                    raise ValueError(f'empty {split} split')
            selected = {}
            unit_counts = {}
            for split, limit in (('train', max_train_groups),
                                 ('validation', max_validation_groups)):
                if not isinstance(limit, int) or limit < 1:
                    raise ValueError('positive group subset limit required')
                selected[split] = frozenset(row[0] for row in connection.execute(
                    'SELECT unit FROM units WHERE split=? ORDER BY selector, unit LIMIT ?',
                    (split, limit)))
            for split in ('train', 'validation', 'calibration', 'test'):
                unit_counts[split] = connection.execute(
                    'SELECT COUNT(*) FROM units WHERE split=?', (split,)).fetchone()[0]
        finally:
            connection.close()
    rows = {}
    for split in ('train', 'validation'):
        rows[split] = []
        with (directory / f'{split}.jsonl').open('rb') as handle:
            for line in handle:
                if time.monotonic() > deadline:
                    raise TimeoutError('pilot time cap reached during subset loading')
                if line.strip():
                    row = json.loads(line)
                    if row.get('source') != expected_source:
                        raise ValueError('row source contract differs from pilot spec')
                    if _unit(row) in selected[split]:
                        rows[split].append(row)
                        budget = max_train_episodes if split == 'train' else max_validation_episodes
                        if len(rows[split]) > budget:
                            raise ValueError('selected complete panels exceed absolute episode budget')
    info = {'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'complete_episode_counts': counts,
            'complete_split_unit_counts': unit_counts,
            'selected_split_unit_counts': {key: len(value) for key, value in selected.items()},
            'selected_split_unit_sha256': {
                key: hashlib.sha256(json.dumps(sorted(value)).encode()).hexdigest()
                for key, value in selected.items()},
            'selected_unit_ids': {key: sorted(value) for key, value in selected.items()}}
    return rows, info


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


def _examples(rows, tokenizer, policy, max_length, deadline):
    examples = []
    excluded = {'no_supervised_event_or_positive_gain': 0,
                'fewer_than_two_eligible': 0, 'prompt_overflow': 0}
    for episode in rows:
        if time.monotonic() > deadline:
            raise TimeoutError('pilot time cap reached during rendering')
        ids = [candidate['id'] for candidate in episode['alternatives']]
        eligible = [candidate.get('available', True) for candidate in episode['alternatives']]
        if sum(eligible) < 2:
            excluded['fewer_than_two_eligible'] += 1
            continue
        labeled = _label(episode, ids, eligible, policy)
        if labeled is None:
            excluded['no_supervised_event_or_positive_gain'] += 1
            continue
        try:
            rendered = render_behavioral(episode, tokenizer, policy, max_length=max_length)
        except ValueError as exc:
            if 'exceeds max_length' not in str(exc):
                raise
            excluded['prompt_overflow'] += 1
            continue
        if rendered['candidate_ids'] != ids or rendered['eligible'] != eligible:
            raise ValueError('rendered candidate coverage differs from episode')
        examples.append((rendered, labeled))
    if not examples:
        raise ValueError('no eligible supervised groups')
    return examples, excluded


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


def _validate(model, tokenizer, validation, device, deadline):
    import torch

    model.eval()
    totals = {}
    with torch.no_grad():
        for source, items in validation.items():
            total = 0.0
            for item in items:
                if time.monotonic() > deadline:
                    raise TimeoutError('pilot time cap reached during validation')
                total += float(_loss(model, tokenizer, item, device))
            totals[source] = total / len(items)
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
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    source_identity = _source_identity()
    if config.get('publish_model_artifacts', False):
        raise ValueError('model artifact publication requires separate rights and destination review')
    if config['model_id'] != MODEL or config['revision'] != REVISION:
        raise ValueError('unexpected base model identity')
    if not (1 <= config['max_steps'] <= 100 and 60 <= config['max_runtime_seconds'] <= 7200):
        raise ValueError('pilot step/runtime limits exceeded')
    if not (1 <= config['validation_interval'] <= config['max_steps']):
        raise ValueError('invalid validation interval')
    effective_hyperparameters = {
        **{key: config[key] for key in ('max_steps', 'max_runtime_seconds',
            'validation_interval', 'max_length', 'learning_rate', 'seed', 'lora')},
        'max_train_groups': config.get('max_train_groups', 1000),
        'max_validation_groups': config.get('max_validation_groups', 100),
        'max_train_episodes': config.get('max_train_episodes', 10000),
        'max_validation_episodes': config.get('max_validation_episodes', 1000)}
    specs = config['sources']
    _validate_source_specs(specs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    training, validation, provenance = {}, {}, {}
    seen_manifest_hashes = set()
    for spec in specs:
        rows, source_info = _load_source(
            spec, seed=config['seed'], max_train_groups=config.get('max_train_groups', 1000),
            max_validation_groups=config.get('max_validation_groups', 100),
            max_train_episodes=config.get('max_train_episodes', 10000),
            max_validation_episodes=config.get('max_validation_episodes', 1000),
            deadline=started + config['max_runtime_seconds'])
        if source_info['manifest_sha256'] in seen_manifest_hashes:
            raise ValueError('duplicate source manifest across domains')
        seen_manifest_hashes.add(source_info['manifest_sha256'])
        (output / f"selected_ids_{spec['name']}.json").write_text(
            json.dumps(source_info.pop('selected_unit_ids'), indent=2) + '\n')
        if (spec['feature_policy'].get('prediction_event') == 'conditional_booking_choice' and
                json.loads((Path(spec['directory']) / 'manifest.json').read_text()).get('candidate_pool_scope')
                != 'source_exposed_candidates'):
            raise ValueError('conditional booking requires exposed candidate pool')
        training[spec['name']], train_excluded = _examples(
            rows['train'], tokenizer, spec['feature_policy'], config['max_length'],
            started + config['max_runtime_seconds'])
        validation[spec['name']], validation_excluded = _examples(
            rows['validation'], tokenizer, spec['feature_policy'], config['max_length'],
            started + config['max_runtime_seconds'])
        if time.monotonic() > started + config['max_runtime_seconds']:
            raise TimeoutError('pilot time cap reached before model load')
        provenance[spec['name']] = {**source_info,
                                    'canonical_source': spec['expected_source'],
                                    'feature_policy': spec['feature_policy'],
                                    'selected_train_episodes': len(rows['train']),
                                    'selected_validation_episodes': len(rows['validation']),
                                    'eligible_train_groups': len(training[spec['name']]),
                                    'eligible_validation_groups': len(validation[spec['name']]),
                                    'train_loss_exclusions': train_excluded,
                                    'validation_loss_exclusions': validation_excluded}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, device_map='auto')
    lora = config['lora']
    model = get_peft_model(base, LoraConfig(r=lora['r'], lora_alpha=lora['alpha'],
                                            lora_dropout=lora['dropout'], bias='none',
                                            task_type='CAUSAL_LM',
                                            target_modules=lora['target_modules'],
                                            exclude_modules='.*vision_tower.*'))
    if time.monotonic() > started + config['max_runtime_seconds']:
        raise TimeoutError('pilot time cap reached during model load')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config['learning_rate'])
    probe_parameter = next(p for p in model.parameters() if p.requires_grad)
    observed_update = False
    run = wandb.init(project=config['wandb_project'], group='ordivo-v2-behavioral',
                     mode=config.get('wandb_mode', 'offline'),
                     config={'model_id': MODEL, 'revision': REVISION,
                             'source_identity': source_identity,
                             'source_archive_sha256': source_identity['archive_sha256'],
                             'config_sha256': config_sha256,
                             'prompt_version': PROMPT_VERSION,
                             'effective_hyperparameters': effective_hyperparameters,
                             'sources': provenance, 'seed': config['seed']})
    cursors = {source: 0 for source in training}
    for items in training.values():
        random.shuffle(items)
    try:
        base_by_source, base_macro = _validate(
            model, tokenizer, validation, 'cuda', started + config['max_runtime_seconds'])
        run.log({'validation/base_macro_source_loss': base_macro,
                 **{f'validation/base_{key}_query_loss': value
                    for key, value in base_by_source.items()}}, step=0)
    except BaseException:
        run.finish()
        raise
    best = float('inf')
    best_step = 0
    best_probe_loss = None
    best_by_source = None
    realized_steps = 0
    updates_by_source = {source: 0 for source in training}
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
            realized_steps = step
            updates_by_source[source] += 1
            observed_update |= bool(torch.count_nonzero(probe_parameter.detach() - before_update).item())
            run.log({'train/query_loss': float(loss.detach()),
                     f'train/{source}_query_loss': float(loss.detach()),
                     f'train/{source}_updates': updates_by_source[source]}, step=step)
            if step % config['validation_interval'] == 0 or step == config['max_steps']:
                by_source, macro = _validate(model, tokenizer, validation, 'cuda',
                                             started + config['max_runtime_seconds'])
                run.log({'validation/macro_source_loss': macro,
                         **{f'validation/{key}_query_loss': value for key, value in by_source.items()}},
                        step=step)
                if macro < best:
                    best, best_step = macro, step
                    best_by_source = dict(by_source)
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
        if time.monotonic() > started + config['max_runtime_seconds']:
            raise TimeoutError('pilot time cap reached during adapter reload')
        restored = PeftModel.from_pretrained(fresh, output / 'best_adapter')
        restored.eval()
        with torch.no_grad():
            after = float(_loss(restored, tokenizer, validation[list(validation)[0]][0], 'cuda'))
        reload_delta = abs(best_probe_loss - after)
        if reload_delta > 0.02:
            raise RuntimeError(f'best adapter reload changed probe loss: {reload_delta}')
        receipt = {'status': 'pilot_completed',
                   'claim_scope': f'preliminary bounded {len(specs)}-domain pilot',
                   'model_id': MODEL, 'revision': REVISION, 'runtime': runtime,
                   'source_identity': source_identity,
                   'config_sha256': config_sha256,
                   'prompt_version': PROMPT_VERSION,
                   'effective_hyperparameters': effective_hyperparameters,
                   'seed': config['seed'], 'best_step': best_step,
                   'realized_optimizer_steps': realized_steps,
                   'optimizer_updates_by_source': updates_by_source,
                   'nonzero_lora_update': observed_update,
                   'base_validation_macro_source_loss': base_macro,
                   'base_validation_by_source_loss': base_by_source,
                   'best_validation_macro_source_loss': best,
                   'best_validation_by_source_loss': best_by_source,
                   'source_provenance': provenance,
                   'wall_seconds': time.monotonic() - started,
                   'reload_probe_loss': after,
                   'best_checkpoint_reload_probe_loss_delta': reload_delta}
        (output / 'pilot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        run.log({'selection/best_validation_macro_source_loss': best,
                 'selection/best_step': best_step})
        return receipt
    finally:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
