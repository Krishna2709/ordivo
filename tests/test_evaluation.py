import pytest
from liquid_mcdm.evaluation import evaluate_predictions


def test_accuracy_brier_and_ties_have_correct_denominators():
    examples = [
        {'episode_id': 'a', 'method': 'TOPSIS', 'target': {'winner': 'A'}},
        {'episode_id': 'b', 'method': 'TOPSIS', 'target': {'winner': 'B'}},
        {'episode_id': 'c', 'method': 'VIKOR', 'target': {'winner': None}},
    ]
    predictions = {'a': {'A': .8, 'B': .2}, 'b': {'A': .6, 'B': .4}}
    r = evaluate_predictions(examples, predictions)
    assert r['eligible'] == 2 and r['excluded_nonunique'] == 1
    assert r['accuracy'] == .5
    assert r['brier_multiclass'] == pytest.approx(.4)
    assert r['ece_10_bins'] == pytest.approx(.4)
    assert r['by_method']['TOPSIS']['accuracy'] == .5


def test_missing_prediction_is_not_silently_dropped():
    with pytest.raises(ValueError, match='missing'):
        evaluate_predictions([{'episode_id':'a','method':'TOPSIS','target':{'winner':'A'}}], {})


def test_invalid_probabilities_rejected():
    e = [{'episode_id':'a','method':'TOPSIS','target':{'winner':'A'}}]
    with pytest.raises(ValueError):
        evaluate_predictions(e, {'a':{'A':2.,'B':-1.}})
