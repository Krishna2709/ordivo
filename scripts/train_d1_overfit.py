"""Gated d1-3B LoRA overfit run. Remote NVIDIA GPU execution only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import yaml

from liquid_mcdm.data import load_dataset
from liquid_mcdm.evaluation import evaluate_predictions
from liquid_mcdm.runtime import require_remote_gpu
from liquid_mcdm.training.rendering import decision_input, option_probabilities, render_example


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def finite_probabilities(probabilities, option_ids):
    if set(probabilities) != set(option_ids):
        return False
    values = [float(probabilities[key]) for key in option_ids]
    return (all(math.isfinite(v) and v >= 0 for v in values)
            and math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5))


def eligible(episodes, count):
    rows = [episode for episode in episodes if episode['target']['winner'] is not None]
    if len(rows) < count:
        raise RuntimeError(f'Need {count} unique-winner episodes; found {len(rows)}')
    return rows[:count]


def model_inputs(batch):
    allowed = {'input_ids', 'attention_mask', 'position_ids', 'labels', 'pixel_values',
               'pixel_attention_mask', 'image_sizes'}
    return {key: value for key, value in batch.items() if key in allowed}


def predict_standard(model, tokenizer, episodes):
    import torch

    predictions = {}
    model.eval()
    device = next(model.parameters()).device
    for episode in episodes:
        row = render_example(episode, tokenizer)
        inputs = tokenizer(row['prompt'], add_special_tokens=False, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[0, -1].float().cpu()
        ids = [a['id'] for a in episode['alternatives']]
        probabilities = option_probabilities(tokenizer, ids, logits)
        if not finite_probabilities(probabilities, ids):
            raise RuntimeError('Standard model emitted an invalid option distribution')
        predictions[episode['episode_id']] = probabilities
    return predictions


def predict_custom(model, episodes):
    predictions = {}
    requests = []
    for episode in episodes:
        state, question = decision_input(episode)
        # The loaded repository has its own prompt module/class identity. A
        # plain schema mapping is converted by that pinned module's as_question.
        question_schema = {'type': question.type, 'instructions': question.instructions,
                           'criteria': dict(question.criteria)}
        requests.append((state, {'winner': question_schema}))
    responses = model.system_one_batch(requests)
    for episode, response in zip(episodes, responses):
        probabilities = response['answers']['winner']['probabilities']
        ids = [a['id'] for a in episode['alternatives']]
        if not finite_probabilities(probabilities, ids):
            raise RuntimeError('Custom System One model emitted an invalid option distribution')
        predictions[episode['episode_id']] = probabilities
    return predictions


def compare_predictions(left, right):
    absolute = []
    agreements = 0
    for episode_id, probabilities in left.items():
        other = right[episode_id]
        keys = list(probabilities)
        absolute.extend(abs(probabilities[key] - other[key]) for key in keys)
        agreements += max(keys, key=probabilities.get) == max(keys, key=other.get)
    return {'episodes': len(left), 'argmax_agreement': agreements / len(left),
            'max_absolute_probability_delta': max(absolute),
            'mean_absolute_probability_delta': sum(absolute) / len(absolute)}


def winner_nll(episodes, predictions):
    losses = []
    for episode in episodes:
        winner = episode['target']['winner']
        if winner is not None:
            losses.append(-math.log(max(float(predictions[episode['episode_id']][winner]), 1e-12)))
    if not losses:
        raise RuntimeError('No unique-winner episodes for winner NLL')
    return sum(losses) / len(losses)


def unload(model=None):
    if model is not None:
        del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def build_sft_config(config, output, dtype):
    import torch
    from trl import SFTConfig

    values = {
        'output_dir': str(output / 'trainer'),
        'max_steps': config['max_steps'],
        'per_device_train_batch_size': config['per_device_train_batch_size'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'gradient_checkpointing': config['gradient_checkpointing'],
        'learning_rate': config['learning_rate'],
        'lr_scheduler_type': config['lr_scheduler_type'],
        # Transformers 5 accepts a fractional warmup through warmup_steps;
        # values below one are interpreted as a ratio of total training steps.
        'warmup_steps': config['warmup_ratio'],
        'logging_steps': 1,
        'save_strategy': 'steps',
        'save_steps': config['save_steps'],
        'save_total_limit': config['save_total_limit'],
        'eval_strategy': 'no',
        'bf16': dtype == torch.bfloat16,
        'fp16': dtype == torch.float16,
        'report_to': ['wandb'],
        'run_name': 'd1-3b-mcdm-overfit-64x25',
        'seed': config['seed'],
        'data_seed': config['seed'],
        'completion_only_loss': config['completion_only_loss'],
        'packing': config['packing'],
        'max_length': config['max_length'],
        'remove_unused_columns': True,
    }
    parameters = inspect.signature(SFTConfig).parameters
    unsupported = sorted(set(values) - set(parameters))
    if unsupported:
        raise RuntimeError(f'TRL SFTConfig compatibility failure: unsupported fields {unsupported}')
    return SFTConfig(**values)


def inspect_batch_and_gradients(trainer, expected_alias_ids, eos_id):
    import torch

    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch.get('labels')
    if labels is None:
        raise RuntimeError('TRL collator did not create labels')
    supervised = labels[0][labels[0] != -100].detach().cpu().tolist()
    if len(supervised) != 2 or supervised[-1] != eos_id or supervised[0] not in expected_alias_ids:
        raise RuntimeError(f'TRL completion mask mismatch: got {supervised!r}, expected alias plus EOS')
    trainer.model.train()
    trainer.model.zero_grad(set_to_none=True)
    outputs = trainer.model(**model_inputs(batch))
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError('Compatibility forward pass produced a missing or non-finite loss')
    loss.backward()
    checked = 0
    nonzero = 0
    for name, parameter in trainer.model.named_parameters():
        if parameter.requires_grad:
            checked += 1
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f'Non-finite LoRA gradient: {name}')
                nonzero += int(torch.count_nonzero(parameter.grad).item() > 0)
    if checked == 0 or nonzero == 0:
        raise RuntimeError('Compatibility backward pass produced no nonzero LoRA gradients')
    trainer.model.zero_grad(set_to_none=True)
    return {'supervised_token_ids': supervised, 'supervised_tokens_per_example': len(supervised),
            'finite_loss': float(loss.detach().cpu()), 'trainable_tensors': checked,
            'trainable_tensors_with_nonzero_gradient': nonzero,
            'gpu_memory_allocated_bytes': torch.cuda.memory_allocated(),
            'gpu_memory_reserved_bytes': torch.cuda.memory_reserved()}


def run(config_path, data_dir, output):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer, set_seed
    from trl import SFTTrainer

    runtime = require_remote_gpu()
    config = yaml.safe_load(Path(config_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'starting', 'passed': False, 'training_started': False,
             'model_loaded': False, 'training_validated': False,
             'started_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime,
             'config': config, 'python': sys.version, 'platform': platform.platform()}

    def checkpoint():
        write_json(output / 'run.json', state)

    checkpoint()
    try:
        if config['model_id'] != 'LiquidAI/d1-3B' or len(config['revision']) != 40:
            raise RuntimeError('Unexpected model identity')
        if config['max_steps'] != 25 or config['overfit_examples'] != 64:
            raise RuntimeError('This entry point is bounded to the reviewed 64-example, 25-step run')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable after runtime guard')
        set_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        state['dtype'] = str(dtype)
        state['torch'] = torch.__version__
        state['cuda'] = torch.version.cuda
        state['status'] = 'loading_data'
        state['dataset_manifest_sha256'] = hashlib.sha256(
            (Path(data_dir) / 'manifest.json').read_bytes()).hexdigest()
        checkpoint()

        train_episodes = eligible(load_dataset(data_dir, 'train'), config['overfit_examples'])
        test_episodes = eligible(load_dataset(data_dir, 'test'), 16)
        parity_episodes = test_episodes[:8]
        tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        rows = [render_example(episode, tokenizer, max_length=config['max_length'])
                for episode in train_episodes]
        dataset = Dataset.from_list([{'prompt': row['prompt'], 'completion': row['completion']}
                                     for row in rows])
        expected_alias_ids = {row['alias_token_id'] for row in rows}

        state['status'] = 'base_custom_evaluation'
        checkpoint()
        custom = AutoModel.from_pretrained(config['model_id'], revision=config['revision'],
                                            trust_remote_code=True, dtype=dtype).to('cuda')
        # Bypass D1Model.engine because its lazy getter reloads a tokenizer
        # without forwarding revision. Build that pinned module's engine with
        # the tokenizer already loaded at the reviewed checkpoint revision.
        remote_module = importlib.import_module(custom.__class__.__module__)
        custom.__dict__['engine'] = remote_module.SystemOne(model=custom, tokenizer=tokenizer)
        state['model_loaded'] = True
        custom_predictions = predict_custom(custom, parity_episodes)
        write_json(output / 'base_custom_predictions.json', custom_predictions)
        del custom
        unload()

        state['status'] = 'standard_model_compatibility'
        checkpoint()
        model = AutoModelForImageTextToText.from_pretrained(
            config['model_id'], revision=config['revision'], dtype=dtype, device_map='auto')
        base_predictions = predict_standard(model, tokenizer, test_episodes)
        base_train_predictions = predict_standard(model, tokenizer, train_episodes)
        write_json(output / 'base_predictions.json', base_predictions)
        write_json(output / 'base_train_predictions.json', base_train_predictions)
        write_json(output / 'base_metrics.json', evaluate_predictions(test_episodes, base_predictions))
        base_train_metrics = evaluate_predictions(train_episodes, base_train_predictions)
        base_train_metrics['winner_nll'] = winner_nll(train_episodes, base_train_predictions)
        write_json(output / 'base_train_metrics.json', base_train_metrics)
        parity = compare_predictions(custom_predictions,
                                     {key: base_predictions[key] for key in custom_predictions})
        write_json(output / 'custom_standard_parity.json', parity)
        if parity['argmax_agreement'] < 0.875 or parity['max_absolute_probability_delta'] > 0.05:
            raise RuntimeError(f'Custom/standard readout parity failed: {parity}')

        lora_values = config['lora']
        peft_config = LoraConfig(
            r=lora_values['r'], lora_alpha=lora_values['lora_alpha'],
            lora_dropout=lora_values['lora_dropout'], bias=lora_values['bias'],
            task_type=lora_values['task_type'], target_modules=lora_values['target_modules'],
            exclude_modules=lora_values['exclude_modules'])
        model = get_peft_model(model, peft_config)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any('vision_tower' in name for name in trainable):
            raise RuntimeError('LoRA trainable scope is empty or includes the vision tower')
        state['lora'] = {'trainable_parameter_tensors': len(trainable),
                         'vision_trainable_tensors': 0}

        sft_config = build_sft_config(config, output, dtype)
        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset,
                             processing_class=tokenizer)
        state['compatibility'] = inspect_batch_and_gradients(
            trainer, expected_alias_ids, tokenizer.eos_token_id)
        state['training_validated'] = True
        state['status'] = 'training'
        state['training_started'] = True
        checkpoint()
        result = trainer.train()
        state['train_metrics'] = {key: float(value) if isinstance(value, (int, float)) else value
                                  for key, value in result.metrics.items()}
        adapter = output / 'adapter'
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(adapter)

        state['status'] = 'post_train_evaluation'
        checkpoint()
        post_predictions = predict_standard(trainer.model, tokenizer, test_episodes)
        post_train_predictions = predict_standard(trainer.model, tokenizer, train_episodes)
        write_json(output / 'post_predictions.json', post_predictions)
        write_json(output / 'post_train_predictions.json', post_train_predictions)
        write_json(output / 'post_metrics.json', evaluate_predictions(test_episodes, post_predictions))
        post_train_metrics = evaluate_predictions(train_episodes, post_train_predictions)
        post_train_metrics['winner_nll'] = winner_nll(train_episodes, post_train_predictions)
        write_json(output / 'post_train_metrics.json', post_train_metrics)
        state['overfit_outcome'] = {
            'base_train_accuracy': base_train_metrics['accuracy'],
            'post_train_accuracy': post_train_metrics['accuracy'],
            'accuracy_delta': post_train_metrics['accuracy'] - base_train_metrics['accuracy'],
            'base_train_winner_nll': base_train_metrics['winner_nll'],
            'post_train_winner_nll': post_train_metrics['winner_nll'],
            'winner_nll_delta': post_train_metrics['winner_nll'] - base_train_metrics['winner_nll'],
            'improved': (post_train_metrics['accuracy'] > base_train_metrics['accuracy']
                         or post_train_metrics['winner_nll'] < base_train_metrics['winner_nll']),
            'claim_scope': '64-example training subset only; not generalization evidence',
        }
        reference = {key: post_predictions[key] for key in list(post_predictions)[:4]}
        del trainer, model
        unload()

        state['status'] = 'adapter_reload'
        checkpoint()
        base = AutoModelForImageTextToText.from_pretrained(
            config['model_id'], revision=config['revision'], dtype=dtype, device_map='auto')
        reloaded = PeftModel.from_pretrained(base, adapter)
        reload_episodes = test_episodes[:4]
        reloaded_predictions = predict_standard(reloaded, tokenizer, reload_episodes)
        reload_parity = compare_predictions(reference, reloaded_predictions)
        write_json(output / 'adapter_reload_parity.json', reload_parity)
        if reload_parity['argmax_agreement'] != 1 or reload_parity['max_absolute_probability_delta'] > 1e-4:
            raise RuntimeError(f'Saved adapter reload parity failed: {reload_parity}')
        state.update(status='passed', passed=True,
                     finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        checkpoint()
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], check=False,
                           stdout=(output / 'pip-freeze.txt').open('w'), stderr=subprocess.STDOUT)
        except Exception:
            pass
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.data, args.output)


if __name__ == '__main__':
    main()
