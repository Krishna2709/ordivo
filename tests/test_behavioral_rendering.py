import importlib
import hashlib
import json
import sys
import time
import types

import pytest


class CharacterTokenizer:
    bos_token = '<BOS>'

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


@pytest.fixture
def renderer(monkeypatch):
    behavioral = types.ModuleType('liquid_mcdm.data.behavioral')

    def inference_input(episode, feature_policy):
        assert feature_policy['context_features'] == ['purpose']
        assert feature_policy['alternative_features'] == ['cost']
        keys = ('task_type', 'context', 'criteria', 'constraints', 'preferences', 'alternatives')
        projection = {key: episode[key] for key in keys}
        if episode['task_type'] == 'observed_choice':
            projection['prediction_event'] = feature_policy['prediction_event']
        return projection

    behavioral.inference_input = inference_input
    monkeypatch.setitem(sys.modules, behavioral.__name__, behavioral)
    module = importlib.import_module('liquid_mcdm.training.behavioral_rendering')
    monkeypatch.setattr(module, 'inference_input', inference_input)
    return module.render_behavioral


def episode():
    return {'task_type': 'stated_choice', 'context': {'purpose': 'commute'},
            'criteria': [], 'constraints': [], 'preferences': {},
            'alternatives': [{'id': 'private-id-1', 'features': {'cost': 3}, 'available': True},
                             {'id': 'private-id-2', 'features': {'cost': 5}, 'available': False}],
            'supervision': {'selected_id': 'private-id-1'},
            'exposure': {'position_by_id': {'private-id-1': 2}},
            'source': 'private-source', 'group_id': 'private-group',
            'episode_id': 'private-episode', 'provenance': {'raw': 'private-provenance'}}


def test_aliases_and_leakage_boundary(renderer):
    row = renderer(episode(), CharacterTokenizer(),
                   {'context_features': ['purpose'], 'alternative_features': ['cost']})
    assert row['candidate_ids'] == ['private-id-1', 'private-id-2']
    assert row['eligible'] == [True, False]
    assert len(row['readout_groups']) == 2
    for secret in ('private-id', 'private-source', 'private-group', 'private-episode',
                   'private-provenance', 'selected_id', 'position_by_id'):
        assert secret not in row['prompt']


def test_rejects_truncation_and_missing_coverage(renderer):
    policy = {'context_features': ['purpose'], 'alternative_features': ['cost']}
    with pytest.raises(ValueError, match='truncation forbidden'):
        renderer(episode(), CharacterTokenizer(), policy, max_length=4)
    modified = episode()
    modified['alternatives'][1]['id'] = 'private-id-1'
    with pytest.raises(ValueError, match='unique'):
        renderer(modified, CharacterTokenizer(), policy)


def test_observed_event_is_visible_and_target_is_not(renderer):
    observed = episode()
    observed['task_type'] = 'observed_choice'
    policy = {'context_features': ['purpose'], 'alternative_features': ['cost'],
              'prediction_event': 'conditional_booking_choice'}
    booking = renderer(observed, CharacterTokenizer(), policy)
    assert 'exactly one booking' in booking['prompt']
    assert 'conditional_booking_choice' in booking['prompt']
    assert 'selected_id' not in booking['prompt']
    policy['prediction_event'] = 'combined_click_booking_relevance'
    click = renderer(observed, CharacterTokenizer(), policy)
    assert booking['prompt'] != click['prompt']
    assert 'click' in click['prompt']


def test_predictor_only_request_preserves_frozen_prompt_bytes(renderer):
    from liquid_mcdm.training.behavioral_rendering import render_predictor_input

    policy = {'context_features': ['purpose'], 'alternative_features': ['cost']}
    projected = {key: episode()[key] for key in (
        'task_type', 'context', 'criteria', 'constraints', 'preferences', 'alternatives')}
    request = {'schema_version': 'ordivo-behavioral-v1', **projected}
    supervised = renderer(episode(), CharacterTokenizer(), policy)
    predicted = render_predictor_input(request, CharacterTokenizer(), policy)
    assert predicted == supervised
    assert 'private-id' not in predicted['prompt']


def test_predictor_only_rejects_labels_extra_features_and_event_drift(renderer):
    from liquid_mcdm.training.behavioral_rendering import render_predictor_input

    policy = {'context_features': ['purpose'], 'alternative_features': ['cost']}
    base = {'schema_version': 'ordivo-behavioral-v1', **{
        key: episode()[key] for key in (
            'task_type', 'context', 'criteria', 'constraints', 'preferences', 'alternatives')}}
    import copy
    for change in ('supervision', 'extra_feature', 'nested_label', 'schema', 'event'):
        request = copy.deepcopy(base)
        if change == 'supervision':
            request['supervision'] = {'selected_id': 'private-id-1'}
        elif change == 'extra_feature':
            request['alternatives'][0]['features']['post_choice'] = 1
        elif change == 'nested_label':
            request['preferences']['selected_id'] = 'private-id-1'
        elif change == 'schema':
            request['schema_version'] = 'other'
        else:
            request['task_type'] = 'observed_choice'
            request['prediction_event'] = 'conditional_booking_choice'
        with pytest.raises(ValueError):
            render_predictor_input(request, CharacterTokenizer(), policy)


def test_supervised_labels_respect_event_population_and_unjudged_mask():
    from scripts.train_v2_supervised import _label

    ids = ['a', 'b', 'c']
    available = [True, True, True]
    booking = {'task_type': 'observed_choice', 'supervision': {'outcomes_by_id': {
        'a': {'booking_bool': 1, 'click_bool': 1},
        'b': {'booking_bool': 0, 'click_bool': 1},
        'c': {'booking_bool': 0, 'click_bool': 0}}}}
    kind, target, mask = _label(booking, ids, available,
                                {'prediction_event': 'conditional_booking_choice'})
    assert kind == 'choice' and target.item() == 0 and mask.tolist() == available
    booking['supervision']['outcomes_by_id']['b']['booking_bool'] = 1
    assert _label(booking, ids, available,
                  {'prediction_event': 'conditional_booking_choice'}) is None
    relevance = {'task_type': 'graded_relevance', 'supervision': {
        'grades_by_id': {'a': 'E', 'b': 'I'}, 'unjudged_ids': ['c']}}
    kind, target, mask = _label(relevance, ids, available, {})
    assert kind == 'ranking' and target.tolist() == [1., 0., 0.]
    assert mask.tolist() == [True, True, False]


def test_supervised_exclusion_precedes_conditional_projection(monkeypatch):
    from scripts import train_v2_supervised as trainer

    called = []
    def render(episode, tokenizer, policy, max_length):
        called.append(episode['episode_id'])
        return {'candidate_ids': ['a', 'b'], 'eligible': [True, True]}

    monkeypatch.setattr(trainer, 'render_behavioral', render)
    zero = {'episode_id': 'zero', 'task_type': 'observed_choice',
            'alternatives': [{'id': 'a'}, {'id': 'b'}],
            'supervision': {'outcomes_by_id': {
                'a': {'booking_bool': 0, 'click_bool': 0},
                'b': {'booking_bool': 0, 'click_bool': 0}}}}
    one = {'episode_id': 'one', 'task_type': 'observed_choice',
           'alternatives': [{'id': 'a'}, {'id': 'b'}],
           'supervision': {'outcomes_by_id': {
               'a': {'booking_bool': 1, 'click_bool': 1},
               'b': {'booking_bool': 0, 'click_bool': 0}}}}
    rows, excluded = trainer._examples([zero, one], None,
                                        {'prediction_event': 'conditional_booking_choice'},
                                        100, time.monotonic() + 10)
    assert len(rows) == 1 and called == ['one']
    assert excluded['no_supervised_event_or_positive_gain'] == 1


def test_pilot_group_subset_is_order_invariant_and_panel_atomic():
    from scripts.train_v2_supervised import _select_unit_ids

    units = ['source:respondent-a', 'source:respondent-b', 'source:respondent-c']
    left = _select_unit_ids(units, limit=2, seed=42, source='source', split='train')
    right = _select_unit_ids(list(reversed(units)), limit=2, seed=42,
                             source='source', split='train')
    assert left == right and len(left) == 2
    # The selector operates on respondent split units, not individual rows.
    panel_rows = ['source:respondent-a'] * 3 + ['source:respondent-b'] * 2
    selected = _select_unit_ids(set(panel_rows), limit=1, seed=42,
                                source='source', split='train')
    kept = [unit for unit in panel_rows if unit in selected]
    assert len(kept) in (2, 3)


def test_archive_identity_required_without_git_checkout(monkeypatch):
    from scripts.train_v2_supervised import _source_identity

    monkeypatch.delenv('ORDIVO_SOURCE_COMMIT', raising=False)
    monkeypatch.delenv('ORDIVO_SOURCE_SHA256', raising=False)
    with pytest.raises(ValueError, match='ORDIVO_SOURCE_COMMIT'):
        _source_identity()
    monkeypatch.setenv('ORDIVO_SOURCE_COMMIT', 'a' * 40)
    with pytest.raises(ValueError, match='ORDIVO_SOURCE_SHA256'):
        _source_identity()
    monkeypatch.setenv('ORDIVO_SOURCE_SHA256', 'b' * 64)
    assert _source_identity() == {'commit': 'a' * 40, 'archive_sha256': 'b' * 64}


def test_sql_pilot_selection_is_order_invariant_and_panel_atomic(tmp_path):
    from scripts.train_v2_supervised import _load_source

    train = [
        {'episode_id': f'swissmetro:a{i}', 'source': 'swissmetro', 'group_id': f'a{i}',
         'supervision': {'respondent_id': 'panel-a'}} for i in range(9)
    ] + [
        {'episode_id': 'swissmetro:b1', 'source': 'swissmetro', 'group_id': 'b1',
         'supervision': {'respondent_id': 'panel-b'}}]
    other = {split: [{'episode_id': f'swissmetro:{split}', 'source': 'swissmetro',
                      'group_id': split, 'supervision': {'respondent_id': split}}]
             for split in ('validation', 'calibration', 'test')}

    def materialize(order):
        hashes = {}
        for split, rows in {'train': order, **other}.items():
            content = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            (tmp_path / f'{split}.jsonl').write_bytes(content)
            hashes[split] = hashlib.sha256(content).hexdigest()
        (tmp_path / 'manifest.json').write_text(json.dumps({
            'source': 'swissmetro', 'split_sha256': hashes,
            'split_policy_id': 'swissmetro-respondent-v2-calibration',
            'candidate_pool_scope': 'three_survey_modes',
            'target_policy_id': 'swissmetro-choice-v1'}))
        return _load_source({'directory': str(tmp_path), 'name': 's',
                             'expected_source': 'swissmetro'}, seed=42,
                            max_train_groups=1, max_validation_groups=1,
                            deadline=time.monotonic() + 10)

    left_rows, left_info = materialize(train)
    right_rows, right_info = materialize(list(reversed(train)))
    assert left_info['selected_unit_ids'] == right_info['selected_unit_ids']
    assert left_info['selected_split_unit_sha256'] == right_info['selected_split_unit_sha256']
    assert {row['episode_id'] for row in left_rows['train']} == {
        row['episode_id'] for row in right_rows['train']}
    assert len(left_rows['train']) in (1, 9)  # whole nine-scenario respondent panel

    with pytest.raises(ValueError, match='source contract'):
        _load_source({'directory': str(tmp_path), 'name': 's',
                      'expected_source': 'amazon_esci'}, seed=42,
                     max_train_groups=1, max_validation_groups=1,
                     deadline=time.monotonic() + 10)


def test_pilot_rejects_duplicate_domain_specs():
    from scripts.train_v2_supervised import _validate_source_specs

    with pytest.raises(ValueError, match='distinct canonical domains'):
        _validate_source_specs([
            {'name': 'a', 'expected_source': 'swissmetro', 'directory': '/tmp/a'},
            {'name': 'b', 'expected_source': 'swissmetro', 'directory': '/tmp/b'}])
