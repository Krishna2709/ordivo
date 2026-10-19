"""Leakage-checked d1 rendering for complete behavioral candidate groups."""
from __future__ import annotations

from liquid_mcdm._vendor import d1_prompt
from liquid_mcdm.data.behavioral import FORBIDDEN_FEATURES, inference_input

PROMPT_VERSION = 'behavioral-choice-v1'


def render_behavioral(episode, tokenizer, feature_policy, *, max_length=2048):
    """Render the data owner's inference projection, preserving every candidate."""
    projected = inference_input(episode, feature_policy)
    return _render_projection(projected, tokenizer, feature_policy, max_length=max_length)


def _reject_forbidden_keys(value):
    if isinstance(value, dict):
        if set(value) & FORBIDDEN_FEATURES:
            raise ValueError('predictor input contains forbidden feature or target key')
        for item in value.values():
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def render_predictor_input(request, tokenizer, feature_policy, *, max_length=2048):
    """Render a label-free, already projected behavioral-v1 predictor request.

    The request uses exactly the inference projection plus schema_version.
    This boundary never accepts a labeled episode or silently drops features.
    It shares the frozen pilot's rendering/readout body byte for byte.
    """
    if not isinstance(request, dict) or request.get('schema_version') != 'ordivo-behavioral-v1':
        raise ValueError('ordivo-behavioral-v1 predictor input required')
    projected = {key: value for key, value in request.items() if key != 'schema_version'}
    expected = {'task_type', 'context', 'criteria', 'constraints', 'preferences', 'alternatives'}
    observed = projected.get('task_type') == 'observed_choice'
    if observed:
        expected.add('prediction_event')
    if set(projected) != expected:
        raise ValueError('predictor request has missing or forbidden fields')
    policy_keys = {'context_features', 'alternative_features'}
    if observed:
        policy_keys.add('prediction_event')
    if (not isinstance(feature_policy, dict) or set(feature_policy) != policy_keys or
            any(not isinstance(feature_policy.get(key), list) or
                any(not isinstance(name, str) for name in feature_policy[key]) or
                len(feature_policy[key]) != len(set(feature_policy[key])) or
                set(feature_policy[key]) & FORBIDDEN_FEATURES
                for key in ('context_features', 'alternative_features'))):
        raise ValueError('explicit safe feature allowlists required')
    if not isinstance(projected['context'], dict) or set(projected['context']) != set(feature_policy['context_features']):
        raise ValueError('context features differ from frozen feature policy')
    if (not isinstance(projected['criteria'], list) or
            not isinstance(projected['constraints'], list) or
            not isinstance(projected['preferences'], dict) or
            not isinstance(projected['alternatives'], list)):
        raise ValueError('predictor input has malformed decision fields')
    _reject_forbidden_keys(projected['context'])
    _reject_forbidden_keys(projected['criteria'])
    _reject_forbidden_keys(projected['constraints'])
    _reject_forbidden_keys(projected['preferences'])
    for candidate in projected['alternatives']:
        if (not isinstance(candidate, dict) or set(candidate) != {'id', 'features', 'available'} or
                not isinstance(candidate['id'], str) or not candidate['id'] or
                not isinstance(candidate['features'], dict) or
                set(candidate['features']) != set(feature_policy['alternative_features']) or
                type(candidate['available']) is not bool):
            raise ValueError('candidate differs from frozen feature policy')
        _reject_forbidden_keys(candidate['features'])
    if observed:
        if (projected['prediction_event'] not in {
                'combined_click_booking_relevance', 'conditional_booking_choice'} or
                feature_policy.get('prediction_event') != projected['prediction_event']):
            raise ValueError('observed choice event differs from policy')
    elif feature_policy.get('prediction_event') is not None:
        raise ValueError('prediction event applies only to observed choice')
    return _render_projection(projected, tokenizer, feature_policy, max_length=max_length)


def _render_projection(projected, tokenizer, feature_policy, *, max_length):
    """Unchanged prompt/readout body shared by episode and predictor paths."""
    expected = {'task_type', 'context', 'criteria', 'constraints', 'preferences',
                'alternatives'}
    if projected.get('task_type') == 'observed_choice':
        expected.add('prediction_event')
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
    if projected['task_type'] == 'observed_choice':
        event = projected['prediction_event']
        if event not in {'combined_click_booking_relevance', 'conditional_booking_choice'}:
            raise ValueError('observed choice needs a declared prediction event')
        if feature_policy.get('prediction_event') != event:
            raise ValueError('prediction event differs from feature policy')
        state['prediction_event'] = event
    state['alternatives'] = [
        {'option': internal[i], 'features': a['features'], 'available': a['available']}
        for i, a in enumerate(alternatives)]
    instructions = {
        'observed_choice': {
            'conditional_booking_choice': 'Conditional on exactly one booking within this exposed candidate group, select the booked eligible option.',
            'combined_click_booking_relevance': 'Select the most relevant eligible option under the declared click and booking gain policy. This is a ranking probe.',
        },
        'stated_choice': 'Select the eligible option most likely to be selected.',
        'graded_relevance': 'Select the most relevant eligible option.',
        'explicit_ranking': 'Select the most preferred eligible option as a single-choice probe.',
    }
    if projected['task_type'] not in instructions:
        raise ValueError('unsupported task type')
    instruction = instructions[projected['task_type']]
    if isinstance(instruction, dict):
        instruction = instruction[event]
    question = d1_prompt.Choice(
        instructions=instruction,
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
