"""Pure helper regressions for the remote d1 overfit entry point."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'train_d1_overfit', Path(__file__).resolve().parents[1] / 'scripts' / 'train_d1_overfit.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_probability_validation_rejects_missing_and_nonfinite_options():
    assert training.finite_probabilities({'A': .25, 'B': .75}, ['A', 'B'])
    assert not training.finite_probabilities({'A': 1.0}, ['A', 'B'])
    assert not training.finite_probabilities({'A': float('nan'), 'B': 0.0}, ['A', 'B'])


def test_prediction_comparison_measures_argmax_and_probability_delta():
    report = training.compare_predictions({'one': {'A': .7, 'B': .3}},
                                          {'one': {'A': .6, 'B': .4}})
    assert report['argmax_agreement'] == 1
    assert report['max_absolute_probability_delta'] == pytest.approx(.1)


def test_winner_nll_uses_training_teacher_probability():
    episodes = [{'episode_id': 'one', 'target': {'winner': 'B'}}]
    assert training.winner_nll(episodes, {'one': {'A': .25, 'B': .75}}) == pytest.approx(
        -training.math.log(.75))


def test_eligible_requires_reviewed_sample_count():
    episodes = [{'target': {'winner': 'A'}}]
    with pytest.raises(RuntimeError, match='Need 2'):
        training.eligible(episodes, 2)


def test_custom_prediction_uses_schema_mapping_across_remote_code_boundary(monkeypatch):
    episode = {'episode_id': 'one', 'target': {'winner': 'A'}, 'method': 'TOPSIS',
               'method_params': {},
               'criteria': [{'name': 'c', 'direction': 'max', 'weight': 1.0}],
               'alternatives': [{'id': 'A', 'values': {'c': 2.0}},
                                {'id': 'B', 'values': {'c': 1.0}}]}

    class Model:
        def system_one_batch(self, requests):
            question = requests[0][1]['winner']
            assert isinstance(question, dict)
            assert question['type'] == 'choice'
            return [{'answers': {'winner': {'probabilities': {'A': .8, 'B': .2}}}}]

    result = training.predict_custom(Model(), [episode])
    assert result == {'one': {'A': .8, 'B': .2}}
