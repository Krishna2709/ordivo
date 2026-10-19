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


def test_training_limits_accept_v2_and_reject_unbounded_runs():
    training.validate_training_limits({'max_steps': 400, 'train_examples': 512,
                                       'eval_examples': 128})
    with pytest.raises(RuntimeError, match='max_steps'):
        training.validate_training_limits({'max_steps': 501, 'train_examples': 512,
                                           'eval_examples': 128})
    with pytest.raises(RuntimeError, match='train_examples'):
        training.validate_training_limits({'max_steps': 400, 'train_examples': 1001,
                                           'eval_examples': 128})


def test_prediction_rows_keep_teacher_and_before_after_probabilities():
    episodes = [{'episode_id': 'one', 'method': 'TOPSIS',
                 'criteria': [{'name': 'c'}],
                 'alternatives': [{'id': 'A'}, {'id': 'B'}],
                 'target': {'winner': 'B'}}]
    rows = training.prediction_rows(
        episodes, {'one': {'A': .6, 'B': .4}}, {'one': {'A': .2, 'B': .8}})
    assert rows == [{'episode_id': 'one', 'method': 'TOPSIS', 'teacher': 'B',
                     'base_prediction': 'A', 'post_prediction': 'B',
                     'base_correct': False, 'post_correct': True,
                     'alternatives_count': 2, 'criteria_count': 1,
                     'base_probabilities': '{"A": 0.6, "B": 0.4}',
                     'post_probabilities': '{"A": 0.2, "B": 0.8}'}]


def test_reload_decision_acceptance_separates_argmax_from_exact_probabilities():
    report = {'argmax_agreement': 1.0, 'max_absolute_probability_delta': .03}
    result = training.reload_decision_acceptance(report, probability_tolerance=.05)
    assert result == {'decision_equivalent': True, 'within_probability_tolerance': True,
                      'exact_probability_parity': False}
