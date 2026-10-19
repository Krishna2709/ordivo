"""Matched, bounded direct-Brier C1 versus experimental report-policy R0 continuation.

Both arms start from the same verified supervised adapter and use the same
complete-group train schedule and validation cohort. This is an exploratory
two-domain experiment, not GRPO, a TypeSafe/Laya reproduction, or a release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import yaml

from liquid_mcdm.runtime import require_remote_gpu
from liquid_mcdm.training.behavioral_rendering import PROMPT_VERSION
from liquid_mcdm.training.option_policy import option_logits
from liquid_mcdm.training.report_policy import report_policy_loss
if __package__:
    from .train_v2_supervised import (
        MODEL, REVISION, _examples, _load_source, _loss, _source_identity,
        _validate_source_specs,
    )
else:
    from train_v2_supervised import (
        MODEL, REVISION, _examples, _load_source, _loss, _source_identity,
        _validate_source_specs,
    )

EVENT = 'stated_choice'
POPULATION = 'eligible_stated_choice_group'
DOMAINS = {'swissmetro': 'swissmetro', 'esci': 'amazon_esci'}


def _adapter_digest(directory):
    """Hash sorted relative paths and per-file hashes; same scheme as ops."""
    directory = Path(directory)
    if not (directory / 'adapter_config.json').is_file() or not list(directory.glob('adapter_model.*')):
        raise ValueError('complete frozen PEFT adapter required')
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    if not files:
        raise ValueError('empty adapter directory')
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode() + b'\0' + file_hash.encode() + b'\n')
    return digest.hexdigest()


def _verify_frozen_inputs(config, pilot, receipt_hash, adapter_hash):
    """Reject drift from the completed pilot before reading training labels."""
    if pilot.get('status') != 'pilot_completed':
        raise ValueError('a completed pilot receipt is required')
    if config.get('event') != EVENT or config.get('candidate_population') != POPULATION:
        raise ValueError('continuation requires stated choice event/population')
    if config['pilot_receipt_sha256'] != receipt_hash or config['starting_adapter_sha256'] != adapter_hash:
        raise ValueError('pilot receipt or starting adapter digest differs')
    if (config['pilot_source_commit'], config['pilot_source_archive_sha256'],
            config['pilot_config_sha256']) != (
                pilot['source_identity']['commit'], pilot['source_identity']['archive_sha256'],
                pilot['config_sha256']):
        raise ValueError('pilot source/config identity differs')
    if (pilot['model_id'], pilot['revision'], pilot['prompt_version']) != (MODEL, REVISION, PROMPT_VERSION):
        raise ValueError('pilot base or prompt protocol differs')
    specs = config['sources']
    if {spec['name']: spec['expected_source'] for spec in specs} != DOMAINS or len(specs) != 2:
        raise ValueError('matched arms require Swissmetro and ESCI only')
    for spec in specs:
        previous = pilot['source_provenance'][spec['name']]
        if (spec['expected_source'] != previous['canonical_source'] or
                spec['manifest_sha256'] != previous['manifest_sha256'] or
                spec['selected_split_unit_sha256'] != previous['selected_split_unit_sha256'] or
                spec['feature_policy'] != previous['feature_policy']):
            raise ValueError('manifest, cohort, source or feature policy differs from pilot')
        if spec['name'] == 'swissmetro' and spec['feature_policy'].get('prediction_event') not in (None, EVENT):
            raise ValueError('Swissmetro event differs from stated choice')


def _matched_schedule(lengths, *, seed, steps):
    if set(lengths) != set(DOMAINS) or any(not isinstance(n, int) or n < 1 for n in lengths.values()):
        raise ValueError('two nonempty matched domains required')
    if not isinstance(steps, int) or not 1 <= steps <= 40:
        raise ValueError('continuation is capped at 40 updates per arm')
    rng = random.Random(seed)
    order = list(DOMAINS)
    pools = {name: list(range(lengths[name])) for name in order}
    for values in pools.values():
        rng.shuffle(values)
    cursors = {name: 0 for name in order}
    schedule = []
    for step in range(steps):
        source = order[step % len(order)]
        if cursors[source] >= len(pools[source]):
            rng.shuffle(pools[source])
            cursors[source] = 0
        schedule.append((source, pools[source][cursors[source]]))
        cursors[source] += 1
    return schedule


def _choice_brier(scores, eligible, target_index):
    import torch

    if (scores.ndim != 1 or eligible.dtype != torch.bool or eligible.shape != scores.shape or
            int(eligible.sum()) < 2 or not isinstance(target_index, int) or
            target_index < 0 or target_index >= scores.numel() or not eligible[target_index]):
        raise ValueError('invalid complete categorical group')
    selected = scores[eligible].float()
    if not torch.isfinite(selected).all():
        raise ValueError('nonfinite eligible option score')
    probability = torch.softmax(selected, dim=-1)
    target = torch.zeros_like(probability)
    target[int(eligible[:target_index].sum())] = 1
    return (probability - target).square().sum()


def _scores(model, tokenizer, item, device):
    import torch

    rendered, (kind, _, loss_mask) = item
    if kind != 'choice':
        raise ValueError('Swissmetro must use categorical choice target')
    inputs = tokenizer(rendered['prompt'], add_special_tokens=False, return_tensors='pt').to(device)
    output = model(**inputs, logits_to_keep=1).logits[0, -1]
    eligible = torch.tensor(rendered['eligible'], dtype=torch.bool, device=device)
    if not torch.equal(eligible, loss_mask.to(device)):
        raise ValueError('choice mask differs from available candidates')
    return option_logits(output, rendered['readout_groups'], eligible).float(), eligible


def _metric(model, tokenizer, item, source, device):
    if source == 'esci':
        return _loss(model, tokenizer, item, device)
    scores, eligible = _scores(model, tokenizer, item, device)
    return _choice_brier(scores, eligible, int(item[1][1]))


def _validate(model, tokenizer, validation, deadline):
    import torch

    model.eval()
    totals = {}
    with torch.no_grad():
        for source, items in validation.items():
            total = 0.
            for item in items:
                if time.monotonic() > deadline:
                    raise TimeoutError('continuation time cap reached during validation')
                total += float(_metric(model, tokenizer, item, source, 'cuda'))
            totals[source] = total / len(items)
    model.train()
    return totals, sum(totals.values()) / len(totals)


def _load_adapter(path, dtype, *, trainable):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    base = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION,
                                                       dtype=dtype, device_map='auto')
    return PeftModel.from_pretrained(base, path, is_trainable=trainable)


def _reference_scores(start_adapter, tokenizer, training, schedule, dtype, deadline):
    import torch

    model = _load_adapter(start_adapter, dtype, trainable=False)
    model.eval()
    scores = {}
    try:
        with torch.no_grad():
            for source, index in sorted(set(schedule)):
                if source != 'swissmetro':
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError('continuation time cap reached during reference scoring')
                vector, _ = _scores(model, tokenizer, training[source][index], 'cuda')
                scores[index] = vector.detach().cpu()
    finally:
        del model
        torch.cuda.empty_cache()
    return scores


def run(config_path, output):
    import torch
    import wandb
    from transformers import AutoTokenizer

    runtime = require_remote_gpu()
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    source_identity = _source_identity()
    if config.get('publish_model_artifacts', False):
        raise ValueError('model artifact publication requires separate rights review')
    if (config['model_id'], config['revision'], config['prompt_version']) != (MODEL, REVISION, PROMPT_VERSION):
        raise ValueError('base or prompt protocol differs')
    if (not isinstance(config['seed'], int) or config['max_steps'] != 40 or
            not 60 <= config['max_runtime_seconds'] <= 7200 or
            not 1 <= config['validation_interval'] <= 40 or
            config['max_length'] != 2048 or
            not math.isfinite(config['learning_rate']) or config['learning_rate'] <= 0):
        raise ValueError('invalid frozen continuation budget')
    if (not isinstance(config['report_count'], int) or config['report_count'] < 4 or
            not math.isfinite(config['concentration']) or config['concentration'] <= 0 or
            not math.isfinite(config['kl_weight']) or config['kl_weight'] < 0 or
            not math.isfinite(config['gradient_clip_norm']) or config['gradient_clip_norm'] <= 0):
        raise ValueError('invalid report-policy parameters')
    pilot_bytes = Path(config['pilot_receipt']).read_bytes()
    pilot = json.loads(pilot_bytes)
    start_adapter = Path(config['starting_adapter'])
    adapter_hash = _adapter_digest(start_adapter)
    _verify_frozen_inputs(config, pilot, hashlib.sha256(pilot_bytes).hexdigest(), adapter_hash)
    _validate_source_specs(config['sources'])
    pilot_hyper = pilot['effective_hyperparameters']
    if any(config[key] != pilot_hyper[key] for key in (
            'seed', 'max_steps', 'max_length', 'learning_rate', 'validation_interval')):
        raise ValueError('matched continuation differs from pilot seed, steps or schedule-relevant parameters')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config['max_runtime_seconds']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    training, validation, provenance = {}, {}, {}
    for spec in config['sources']:
        rows, info = _load_source(
            spec, seed=pilot_hyper['seed'], max_train_groups=pilot_hyper['max_train_groups'],
            max_validation_groups=pilot_hyper['max_validation_groups'],
            max_train_episodes=pilot_hyper['max_train_episodes'],
            max_validation_episodes=pilot_hyper['max_validation_episodes'], deadline=deadline)
        prior = pilot['source_provenance'][spec['name']]
        if (info['manifest_sha256'] != prior['manifest_sha256'] or
                info['selected_split_unit_sha256'] != prior['selected_split_unit_sha256'] or
                info['complete_episode_counts'] != prior['complete_episode_counts'] or
                info['complete_split_unit_counts'] != prior['complete_split_unit_counts']):
            raise ValueError('all-four-split source or selected cohort differs from pilot')
        if spec['name'] == 'swissmetro' and any(row['task_type'] != EVENT for part in rows.values() for row in part):
            raise ValueError('Swissmetro episode event differs from frozen stated choice')
        training[spec['name']], train_excluded = _examples(
            rows['train'], tokenizer, spec['feature_policy'], config['max_length'], deadline)
        validation[spec['name']], val_excluded = _examples(
            rows['validation'], tokenizer, spec['feature_policy'], config['max_length'], deadline)
        if (len(training[spec['name']]), len(validation[spec['name']]),
                train_excluded, val_excluded) != (
                    prior['eligible_train_groups'], prior['eligible_validation_groups'],
                    prior['train_loss_exclusions'], prior['validation_loss_exclusions']):
            raise ValueError('rendered training/validation cohort differs from pilot')
        provenance[spec['name']] = {key: info[key] for key in (
            'manifest_sha256', 'complete_episode_counts', 'complete_split_unit_counts',
            'selected_split_unit_counts', 'selected_split_unit_sha256')}
        provenance[spec['name']].update({'eligible_train_groups': len(training[spec['name']]),
                                        'eligible_validation_groups': len(validation[spec['name']]),
                                        'train_loss_exclusions': train_excluded,
                                        'validation_loss_exclusions': val_excluded})
    schedule = _matched_schedule({name: len(items) for name, items in training.items()},
                                 seed=config['seed'], steps=config['max_steps'])
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    reference = _reference_scores(start_adapter, tokenizer, training, schedule, dtype, deadline)
    arm_receipts = {}
    common_base = None
    for arm in ('c1_direct_brier', 'r0_dirichlet_reinforce'):
        if time.monotonic() > deadline:
            raise TimeoutError('continuation time cap reached before second arm')
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        model = _load_adapter(start_adapter, dtype, trainable=True)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError('no trainable LoRA parameters')
        optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'])
        (output / arm).mkdir()
        run_wandb = wandb.init(project=config['wandb_project'], mode=config.get('wandb_mode', 'offline'),
                               group='ordivo-v2-matched-continuation', name=arm,
                               config={'arm': arm, 'source_identity': source_identity,
                                       'config_sha256': config_hash, 'pilot_receipt_sha256': config['pilot_receipt_sha256'],
                                       'starting_adapter_sha256': adapter_hash, 'schedule_sha256': schedule_hash,
                                       'event': EVENT, 'population': POPULATION,
                                       'hyperparameters': {key: config[key] for key in (
                                           'seed', 'max_steps', 'max_runtime_seconds', 'validation_interval',
                                           'max_length', 'learning_rate', 'report_count', 'concentration',
                                           'kl_weight')}, 'sources': provenance})
        try:
            base_by_source, base_macro = _validate(model, tokenizer, validation, deadline)
            if common_base is None:
                common_base = (base_by_source, base_macro)
            elif (abs(base_macro - common_base[1]) > 1e-4 or
                  any(abs(base_by_source[name] - common_base[0][name]) > 1e-4 for name in DOMAINS)):
                raise RuntimeError('matched arms did not start at the same validation state')
            run_wandb.log({'validation/base_macro_loss': base_macro,
                           **{f'validation/base_{source}_loss': value
                              for source, value in base_by_source.items()}}, step=0)
            best, best_step, best_by_source = float('inf'), 0, None
            probe_at_save = None
            realized = 0
            nonzero_update = False
            updates_by_source = {source: 0 for source in DOMAINS}
            zero_advantage_steps = 0
            kl_values, reward_variances, advantage_variances = [], [], []
            for step, (source, index) in enumerate(schedule, 1):
                if time.monotonic() > deadline:
                    break
                item = training[source][index]
                optimizer.zero_grad(set_to_none=True)
                diagnostics = {}
                if source == 'swissmetro':
                    scores, eligible = _scores(model, tokenizer, item, 'cuda')
                    target = int(item[1][1])
                    if arm == 'c1_direct_brier':
                        loss = _choice_brier(scores, eligible, target)
                    else:
                        result = report_policy_loss(
                            scores, reference[index].to('cuda'), eligible, target,
                            event=EVENT, candidate_population=POPULATION,
                            concentration=config['concentration'], report_count=config['report_count'],
                            kl_weight=config['kl_weight'])
                        loss = result['loss']
                        zero_advantage_steps += int(result['zero_variance_group'])
                        kl_values.append(float(result['kl'].detach()))
                        reward_variances.append(float(result['reward_variance']))
                        advantage_variances.append(float(result['advantage_variance']))
                        diagnostics = {'train/reference_kl': float(result['kl'].detach()),
                                       'train/reward_mean': float(result['mean_reward']),
                                       'train/reward_variance': float(result['reward_variance']),
                                       'train/advantage_variance': float(result['advantage_variance']),
                                       'train/zero_advantage_steps': zero_advantage_steps}
                else:
                    loss = _loss(model, tokenizer, item, 'cuda')
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite continuation loss')
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip_norm'])
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError('nonfinite continuation gradient')
                before = trainable[0].detach().clone()
                optimizer.step()
                nonzero_update |= bool(torch.count_nonzero(trainable[0].detach() - before).item())
                realized = step
                updates_by_source[source] += 1
                run_wandb.log({'train/loss': float(loss.detach()),
                               f'train/{source}_loss': float(loss.detach()),
                               f'train/{source}_updates': updates_by_source[source],
                               'train/gradient_norm': float(gradient_norm), **diagnostics}, step=step)
                if step % config['validation_interval'] == 0 or step == config['max_steps']:
                    by_source, macro = _validate(model, tokenizer, validation, deadline)
                    run_wandb.log({'validation/macro_loss': macro,
                                   **{f'validation/{name}_loss': value for name, value in by_source.items()}},
                                  step=step)
                    if macro < best:
                        best, best_step, best_by_source = macro, step, dict(by_source)
                        model.eval()
                        with torch.no_grad():
                            probe_at_save = float(_metric(model, tokenizer, validation['swissmetro'][0],
                                                          'swissmetro', 'cuda'))
                        model.train()
                        model.save_pretrained(output / arm / 'best_adapter')
                        tokenizer.save_pretrained(output / arm / 'best_adapter')
            if best_step == 0 or not nonzero_update or realized != config['max_steps']:
                raise RuntimeError('continuation did not finish finite nonzero updates and validation')
            del model
            torch.cuda.empty_cache()
            restored = _load_adapter(output / arm / 'best_adapter', dtype, trainable=False)
            restored.eval()
            with torch.no_grad():
                probe_after = float(_metric(restored, tokenizer, validation['swissmetro'][0],
                                            'swissmetro', 'cuda'))
            reload_delta = abs(probe_at_save - probe_after)
            if reload_delta > 0.02:
                raise RuntimeError('best continuation adapter changed on reload')
            del restored
            torch.cuda.empty_cache()
            arm_receipts[arm] = {'base_validation_macro_loss': base_macro,
                                 'base_validation_by_source_loss': base_by_source,
                                 'best_validation_macro_loss': best,
                                 'best_validation_by_source_loss': best_by_source,
                                 'best_step': best_step, 'realized_optimizer_steps': realized,
                                 'updates_by_source': updates_by_source,
                                 'nonzero_lora_update': nonzero_update,
                                 'zero_advantage_steps': zero_advantage_steps,
                                 'mean_reference_kl': sum(kl_values) / len(kl_values) if kl_values else None,
                                 'mean_reward_variance': sum(reward_variances) / len(reward_variances) if reward_variances else None,
                                 'mean_advantage_variance': sum(advantage_variances) / len(advantage_variances) if advantage_variances else None,
                                 'reload_probe_loss_delta': reload_delta,
                                 'wandb_run_id': run_wandb.id}
            run_wandb.log({'selection/best_validation_macro_loss': best,
                           'selection/best_step': best_step})
        finally:
            run_wandb.finish()
    receipt = {'status': 'matched_continuation_completed', 'claim_scope': 'exploratory C1 versus R0',
               'source_identity': source_identity, 'runtime': runtime,
               'model_id': MODEL, 'revision': REVISION, 'prompt_version': PROMPT_VERSION,
               'config_sha256': config_hash, 'pilot_receipt_sha256': config['pilot_receipt_sha256'],
               'pilot_config_sha256': config['pilot_config_sha256'],
               'starting_adapter_sha256': adapter_hash, 'schedule_sha256': schedule_hash,
               'event': EVENT, 'candidate_population': POPULATION,
               'finite_concentration_caveat': 'R0 expected Brier is not strictly proper for mean probabilities',
               'hyperparameters': {key: config[key] for key in (
                   'seed', 'max_steps', 'max_runtime_seconds', 'validation_interval', 'max_length',
                   'learning_rate', 'gradient_clip_norm', 'report_count', 'concentration', 'kl_weight')},
               'source_provenance': provenance, 'arms': arm_receipts,
               'wall_seconds': time.monotonic() - started}
    (output / 'continuation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
