from liquid_mcdm._vendor import d1_prompt
from liquid_mcdm.data import validate_episode

PROMPT_VERSION = 'mcdm-choice-v1-full-precision'
NORMALIZATIONS = {'COPRAS': 'column_sum', 'WASPAS': 'max_benefit_min_cost_ratio',
                  'VIKOR': 'criterion_range_regret', 'TOPSIS': 'vector'}


def decision_input(episode):
    """Whitelist input fields so targets/provenance cannot leak into the prompt."""
    state = {k: episode[k] for k in ('method','method_params','criteria','alternatives')}
    state['normalization'] = NORMALIZATIONS[episode['method']]
    question = d1_prompt.Choice(
        instructions='Select the unique winning alternative under the specified MCDM method, '
                     'parameters, normalization, weights and criterion directions. '
                     'For VIKOR select its unique acceptable compromise (lower Q is better).',
        criteria={a['id']: f"Alternative {a['id']}" for a in episode['alternatives']})
    return state, question


def render_example(episode, tokenizer, *, max_length=4096):
    validate_episode(episode)
    winner = episode['target']['winner']
    if winner is None:
        raise ValueError('single-choice training excludes tied/nonunique compromise episodes')
    state, question = decision_input(episode)
    options = list(question.criteria)
    aliases = d1_prompt.aliases(tokenizer, options)
    alias, alias_id = aliases[options.index(winner)]
    prompt = d1_prompt.render(tokenizer, state, question, bos=tokenizer.bos_token)
    completion = alias + tokenizer.eos_token
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    combined = tokenizer.encode(prompt+completion, add_special_tokens=False)
    expected = [alias_id, tokenizer.eos_token_id]
    if combined != prefix_ids+expected:
        raise ValueError('tokenizer merges prompt/completion boundary or answer is not alias+EOS')
    if len(combined) > max_length:
        raise ValueError('example exceeds max_length; do not truncate decision alternatives or answer')
    return {'episode_id': episode['episode_id'], 'prompt': prompt, 'completion': completion,
            'winner': winner, 'option_ids': options, 'alias_token_id': alias_id,
            'readout_groups': d1_prompt.readout_ids(tokenizer, question),
            'input_ids': combined, 'labels': [-100]*len(prefix_ids)+expected,
            'supervised_ids': expected, 'prompt_version': PROMPT_VERSION}


def option_probabilities(tokenizer, option_ids, logits):
    q = d1_prompt.Choice(instructions='Select one.',criteria=dict.fromkeys(option_ids))
    return dict(zip(option_ids,d1_prompt.readout(tokenizer,q,logits)))
