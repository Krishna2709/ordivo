"""Leakage-checked d1 rendering for complete behavioral candidate groups."""
from __future__ import annotations

from liquid_mcdm._vendor import d1_prompt
from liquid_mcdm.data.behavioral import inference_input

PROMPT_VERSION = 'behavioral-choice-v1'


def render_behavioral(episode, tokenizer, feature_policy, *, max_length=2048):
    """Render the data owner's inference projection, preserving every candidate."""
    projected = inference_input(episode, feature_policy)
    expected = {'task_type', 'context', 'criteria', 'constraints', 'preferences',
                'alternatives'}
    if set(projected) != expected:
        raise ValueError('inference projection has missing or unapproved fields')
    alternatives = projected['alternatives']
    if not alternatives or len({a['id'] for a in alternatives}) != len(alternatives):
        raise ValueError('candidate IDs must be present and unique')
    if any(set(a) != {'id', 'features', 'available'} for a in alternatives):
        raise ValueError('candidate projection has missing or unapproved fields')
    if not any(a['available'] for a in alternatives):
        raise ValueError('no eligible candidates')
    # IDs remain output keys only. The model sees positional aliases and features.
    ids = [a['id'] for a in alternatives]
    internal = [f'opt{i}' for i in range(len(ids))]
    state = {key: projected[key] for key in ('task_type', 'context', 'criteria',
                                             'constraints', 'preferences')}
    state['alternatives'] = [
        {'option': internal[i], 'features': a['features'], 'available': a['available']}
        for i, a in enumerate(alternatives)]
    instructions = {
        'observed_choice': 'Select the eligible option most likely to be chosen for the declared event.',
        'stated_choice': 'Select the eligible option most likely to be selected.',
        'graded_relevance': 'Select the most relevant eligible option.',
        'explicit_ranking': 'Select the most preferred eligible option as a single-choice probe.',
    }
    if projected['task_type'] not in instructions:
        raise ValueError('unsupported task type')
    question = d1_prompt.Choice(
        instructions=instructions[projected['task_type']],
        criteria={key: None for key in internal})
    prompt = d1_prompt.render(tokenizer, state, question, bos=tokenizer.bos_token)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_length:
        raise ValueError('complete candidate evidence exceeds max_length; truncation forbidden')
    groups = d1_prompt.readout_ids(tokenizer, question)
    if len(groups) != len(ids):
        raise ValueError('readout failed candidate coverage')
    return {'prompt': prompt, 'input_ids': input_ids, 'candidate_ids': ids,
            'eligible': [bool(a['available']) for a in alternatives],
            'readout_groups': groups, 'prompt_version': PROMPT_VERSION}
