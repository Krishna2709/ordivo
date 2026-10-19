"""Decision contracts. Execute on remote CI; no model or data downloads."""
from copy import deepcopy

import pytest

from liquid_mcdm.decision import resolve_weights, verify_decision


@pytest.fixture
def vendors():
    return {
        'scope': 'hospital/procurement',
        'criteria': [
            {'id': 'cost', 'direction': 'min'},
            {'id': 'quality', 'direction': 'max'},
            {'id': 'delivery', 'direction': 'max'},
        ],
        'alternatives': [
            {'id': 'A', 'values': {'cost': 100, 'quality': .90, 'delivery': .95}},
            {'id': 'B', 'values': {'cost': 80, 'quality': .75, 'delivery': .80}},
            {'id': 'C', 'values': {'cost': 120, 'quality': .98, 'delivery': .99}},
            {'id': 'D', 'values': {'cost': 130, 'quality': .70, 'delivery': .70}},
        ],
        'method': 'WASPAS',
        'weights': None,
        'constraints': [],
    }


def user_weights(cost=1, quality=0, delivery=0):
    return {'mode': 'user', 'values': dict(cost=cost, quality=quality, delivery=delivery)}


def test_absent_preferences_preserve_pareto_without_inventing_winner(vendors):
    report = verify_decision(vendors)
    assert report['status'] == 'preferences_unknown'
    assert report['decision'] is None
    assert report['pareto']['frontier'] == ['A', 'B', 'C']
    assert report['pareto']['dominated_by']['D'] == ['A', 'B', 'C']
    assert report['checks']['scores_verified'] is None


def test_explicit_weights_are_normalized_and_input_is_not_mutated(vendors):
    vendors['weights'] = user_weights(2, 1, 1)
    before = deepcopy(vendors)
    result = resolve_weights(vendors)
    assert result['profiles'][0]['values'] == {'cost': .5, 'quality': .25, 'delivery': .25}
    assert result['source']['mode'] == 'user'
    assert vendors == before


@pytest.mark.parametrize('values', [
    dict(cost=-1, quality=1, delivery=1), dict(cost=0, quality=0, delivery=0),
    dict(cost=float('nan'), quality=1, delivery=1), dict(cost=1, quality=1),
    dict(cost=True, quality=1, delivery=1),
])
def test_invalid_weights_fail_closed(vendors, values):
    vendors['weights'] = {'mode': 'user', 'values': values}
    with pytest.raises(ValueError):
        resolve_weights(vendors)


def test_policy_requires_trusted_registry_and_matching_scope_version(vendors):
    vendors['weights'] = {'mode': 'approved_policy', 'policy_id': 'procurement', 'version': '1'}
    assert resolve_weights(vendors)['status'] == 'preferences_unknown'
    policy = {'scope': vendors['scope'], 'version': '1', 'values': user_weights()['values']}
    resolved = resolve_weights(vendors, policies={'procurement': policy})
    assert resolved['source']['policy_id'] == 'procurement'
    assert resolved['profiles'][0]['values']['cost'] == 1
    policy['scope'] = 'another-tenant'
    with pytest.raises(ValueError, match='scope'):
        resolve_weights(vendors, policies={'procurement': policy})
    policy['scope'], policy['version'] = vendors['scope'], '2'
    with pytest.raises(ValueError, match='version'):
        resolve_weights(vendors, policies={'procurement': policy})


def test_learned_profiles_keep_uncertainty_and_evidence(vendors):
    vendors['weights'] = {'mode': 'learned'}
    assert resolve_weights(vendors)['status'] == 'preferences_unknown'
    learned = {'scope': vendors['scope'], 'evidence_ref': 'artifact:preferences:v1', 'profiles': [
        {'values': user_weights()['values'], 'probability': .4},
        {'values': user_weights(0, 1, 0)['values'], 'probability': .6},
    ]}
    report = verify_decision(vendors, learned=learned)
    assert report['status'] == 'preference_uncertainty'
    assert report['decision'] is None
    assert [row['decision'] for row in report['analyses']] == ['B', 'C']
    assert report['weights']['source']['evidence_ref'] == learned['evidence_ref']


def test_constraints_select_single_feasible_without_fake_scores(vendors):
    vendors['constraints'] = [
        {'criterion': 'cost', 'op': '<=', 'value': 110},
        {'criterion': 'quality', 'op': '>=', 'value': .85},
    ]
    report = verify_decision(vendors, {'decision': 'A'})
    assert report['decision'] == 'A'
    assert report['status'] == 'verified'
    assert report['checks']['constraints_satisfied'] is True
    assert report['checks']['scores_verified'] is None
    assert report['weights']['status'] == 'preferences_unknown'
    assert report['analyses'] == []
    assert verify_decision(vendors, {'decision': 'C'})['status'] == 'rejected'


def test_no_feasible_and_unknown_evidence_are_distinct(vendors):
    vendors['constraints'] = [{'criterion': 'cost', 'op': '<=', 'value': 70}]
    assert verify_decision(vendors)['status'] == 'no_feasible_alternative'
    vendors['alternatives'][1]['values']['cost'] = None
    report = verify_decision(vendors)
    assert report['status'] == 'missing_evidence'
    assert report['feasibility']['B'] == 'unknown'
    assert report['decision'] is None
    assert report['pareto']['complete'] is False


def test_known_violation_overrides_missing_constraint(vendors):
    vendors['constraints'] = [
        {'criterion': 'cost', 'op': '<=', 'value': 110},
        {'criterion': 'quality', 'op': '>=', 'value': .85},
    ]
    vendors['alternatives'][3]['values']['quality'] = None
    assert verify_decision(vendors)['feasibility']['D'] == 'infeasible'


def test_score_claims_recomputed_and_forged_flags_rejected(vendors):
    vendors['weights'] = user_weights()
    # With cost-only WASPAS, each score is exactly minimum cost / cost.
    scores = {'A': .8, 'B': 1., 'C': 2 / 3, 'D': 8 / 13}
    report = verify_decision(vendors, {'decision': 'B', 'scores': scores})
    assert report['status'] == 'verified'
    assert report['checks']['scores_verified'] is True
    scores['B'] = .1
    assert verify_decision(vendors, {'decision': 'B', 'scores': scores})['status'] == 'rejected'
    with pytest.raises(ValueError, match='proposal'):
        verify_decision(vendors, {'decision': 'D', 'checks': {'scores_verified': True}})
    with pytest.raises(ValueError):
        verify_decision(vendors, {'scores': {'B': 1}})


def test_missing_attributes_do_not_certify_partial_frontier(vendors):
    vendors['weights'] = user_weights()
    vendors['alternatives'][2]['values']['quality'] = None
    report = verify_decision(vendors)
    assert report['status'] == 'missing_evidence'
    assert report['pareto']['complete'] is False
    assert 'C' in report['pareto']['unresolved']
    assert report['decision'] is None


def test_ties_are_not_unique_winners(vendors):
    vendors['alternatives'] = vendors['alternatives'][:2]
    vendors['alternatives'][1]['values'] = deepcopy(vendors['alternatives'][0]['values'])
    vendors['weights'] = user_weights()
    report = verify_decision(vendors)
    assert report['status'] == 'no_unique_winner'
    assert report['decision'] is None
    assert report['analyses'][0]['ranking'] == [['A', 'B']]
    assert report['pareto']['frontier'] == ['A', 'B']
    assert verify_decision(vendors, {'decision': 'A'})['status'] == 'rejected'


def test_sensitivity_is_measured_scenarios_not_a_probability(vendors):
    vendors['weights'] = user_weights()
    report = verify_decision(vendors, scenarios=[user_weights(), user_weights(0, 1, 0)])
    assert report['decision'] == 'B'
    assert report['sensitivity']['tested'] is True
    assert report['sensitivity']['winner_stability'] == .5
    assert report['sensitivity']['scenario_count'] == 2
    assert [r['decision'] for r in report['sensitivity']['results']] == ['B', 'C']
    assert verify_decision(vendors)['sensitivity']['winner_stability'] is None


def test_permutations_preserve_decision_and_named_scores(vendors):
    vendors['weights'] = user_weights()
    first = verify_decision(vendors)
    vendors['alternatives'].reverse()
    vendors['criteria'].reverse()
    second = verify_decision(vendors)
    assert first['decision'] == second['decision'] == 'B'
    assert first['analyses'][0]['scores'] == second['analyses'][0]['scores']
    assert first['input_sha256'] != second['input_sha256']


@pytest.mark.parametrize('mutation', ['duplicate', 'direction', 'infinity', 'operator'])
def test_malformed_specs_rejected(vendors, mutation):
    if mutation == 'duplicate':
        vendors['alternatives'][1]['id'] = 'A'
    elif mutation == 'direction':
        vendors['criteria'][0]['direction'] = 'up'
    elif mutation == 'infinity':
        vendors['alternatives'][0]['values']['cost'] = float('inf')
    else:
        vendors['constraints'] = [{'criterion': 'cost', 'op': 'eval', 'value': 1}]
    with pytest.raises(ValueError):
        verify_decision(vendors)


def test_objective_entropy_has_declared_convention_and_no_uniform_fallback(vendors):
    vendors['weights'] = {'mode': 'objective', 'method': 'entropy'}
    for i, row in enumerate(vendors['alternatives']):
        row['values'] = dict(cost=1, quality=int(i == 0), delivery=1)
    result = resolve_weights(vendors)
    assert result['profiles'][0]['values'] == dict(cost=0., quality=1., delivery=0.)
    assert result['source']['convention'] == 'entropy_raw_nonnegative_v1'
    for row in vendors['alternatives']:
        row['values'] = dict(cost=1, quality=1, delivery=1)
    assert resolve_weights(vendors)['status'] == 'preferences_unknown'


def test_critic_symmetric_conflict_and_constant_column(vendors):
    vendors['weights'] = {'mode': 'objective', 'method': 'critic'}
    for i, row in enumerate(vendors['alternatives']):
        row['values'] = dict(cost=i + 1, quality=i + 1, delivery=1)
    result = resolve_weights(vendors)
    assert result['profiles'][0]['values'] == pytest.approx(dict(cost=.5, quality=.5, delivery=0.))
    assert result['source']['convention'] == 'critic_benefit_minmax_population_v1'
    vendors['criteria'][0]['direction'] = 'max'
    assert resolve_weights(vendors)['status'] == 'preferences_unknown'


def test_vikor_preserves_compromise_semantics(vendors):
    vendors['method'] = 'VIKOR'
    vendors['weights'] = user_weights(1, 1, 1)
    report = verify_decision(vendors)
    analysis = report['analyses'][0]
    assert analysis['metadata']['higher_is_better'] is False
    if not analysis['metadata']['acceptable_advantage']:
        assert report['decision'] is None
        assert report['status'] == 'no_unique_winner'
