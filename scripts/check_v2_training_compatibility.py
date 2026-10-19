"""Bounded RunPod A40 d1 option-policy forward/backward and adapter reload check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquid_mcdm._vendor import d1_prompt
from liquid_mcdm.runtime import require_remote_gpu
from liquid_mcdm.training.option_policy import choice_loss, option_probabilities

MODEL = 'LiquidAI/d1-3B'
REVISION = '051bcc464b01b9f92942b364d9586b0ef5912432'


def run(output: Path):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    runtime = require_remote_gpu()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    question = d1_prompt.Choice('Select one eligible option.',
                                {'opt0': 'First option', 'opt1': 'Second option',
                                 'opt2': 'Unavailable option'})
    prompt = d1_prompt.render(tokenizer, {'task_type': 'stated_choice',
                                           'alternatives': ['First', 'Second', 'Third']},
                              question, bos=tokenizer.bos_token)
    groups = d1_prompt.readout_ids(tokenizer, question)
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
    eligible = torch.tensor([True, True, False], device='cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, device_map='auto')
    model.eval()
    with torch.no_grad():
        base_logits = model(**encoded, logits_to_keep=1).logits[0, -1].float()
    tensor_probs = option_probabilities(base_logits, groups).detach().cpu()
    pinned_probs = torch.tensor(d1_prompt.readout(tokenizer, question, base_logits.detach().cpu()))
    parity = float((tensor_probs - pinned_probs).abs().max())
    if parity > 1e-6:
        raise RuntimeError(f'pinned inference readout parity failed: {parity}')
    if option_probabilities(base_logits, groups, eligible)[-1].item() != 0:
        raise RuntimeError('unavailable option was not masked')

    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0., bias='none',
                        task_type='CAUSAL_LM',
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj',
                                        'in_proj', 'w1', 'w2', 'w3'],
                        exclude_modules='.*vision_tower.*')
    model = get_peft_model(model, config)
    model.train()
    model.zero_grad(set_to_none=True)
    answer = model(**encoded, logits_to_keep=1).logits[0, -1]
    loss = choice_loss(answer, groups, torch.tensor(0, device='cuda'),
                       eligible=eligible, objective='nll')
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite direct policy loss')
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError('missing or nonfinite LoRA gradients')
    nonzero = sum(int(torch.count_nonzero(gradient).item() > 0) for gradient in gradients)
    if nonzero == 0:
        raise RuntimeError('all LoRA gradients are zero')
    model.eval()
    with torch.no_grad():
        before = option_probabilities(model(**encoded, logits_to_keep=1).logits[0, -1],
                                      groups, eligible).detach().float().cpu()
    adapter = output / 'adapter'
    model.save_pretrained(adapter)
    del model
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, device_map='auto')
    reloaded = PeftModel.from_pretrained(base, adapter)
    reloaded.eval()
    with torch.no_grad():
        after = option_probabilities(reloaded(**encoded, logits_to_keep=1).logits[0, -1],
                                     groups, eligible).detach().float().cpu()
    reload_delta = float((before - after).abs().max())
    if reload_delta > 0.01 or before.argmax() != after.argmax():
        raise RuntimeError(f'adapter reload parity failed: {reload_delta}')
    receipt = {'passed': True, 'model': MODEL, 'revision': REVISION, 'runtime': runtime,
               'dtype': str(dtype), 'readout_parity_max_delta': parity,
               'finite_loss': float(loss.detach().cpu()),
               'trainable_tensors_with_nonzero_gradient': nonzero,
               'unavailable_probability': 0.0, 'reload_max_delta': reload_delta,
               'claim_scope': 'one fixed prompt, one backward pass, no optimization'}
    (output / 'compatibility.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
