import importlib
import sys
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
