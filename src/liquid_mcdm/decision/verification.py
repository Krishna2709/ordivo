"""Recompute numerical claims independently of neural output.

Independence means the model cannot supply truth flags or substitute its scores
for the exact solver. Formula correctness also relies on the reference suite.
"""
from copy import deepcopy
import hashlib
import json
import operator

import numpy as np

from liquid_mcdm.mcdm.solvers import solve
from .specification import number, validate_decision
from .weights import resolve_weights

OPS = {'<=': operator.le, '>=': operator.ge, '<': operator.lt,
       '>': operator.gt, '==': operator.eq}
SCORE_ATOL = 1e-9
SCORE_RTOL = 1e-7


def _feasibility(spec):
    statuses, evidence = {}, {}
    for row in spec['alternatives']:
        checks = []
        for constraint in spec.get('constraints', []):
            value = row['values'][constraint['criterion']]
            passed = None if value is None else OPS[constraint['op']](value, constraint['value'])
            checks.append(dict(constraint=deepcopy(constraint), observed=value, passed=passed))
        flags = [c['passed'] for c in checks]
        statuses[row['id']] = ('infeasible' if False in flags else
                               'unknown' if None in flags else 'feasible')
        evidence[row['id']] = checks
    return statuses, evidence


def _pareto(spec, feasibility, ids):
    complete, unresolved = [], []
    for row in spec['alternatives']:
        state = feasibility[row['id']]
        if state == 'infeasible':
            continue
        if state == 'unknown' or any(row['values'][key] is None for key in ids):
            unresolved.append(row['id'])
        else:
            complete.append(row)
    signs = [1 if c['direction'] == 'max' else -1 for c in spec['criteria']]
    dominated_by = {}
    for row in complete:
        a = [row['values'][key] * sign for key, sign in zip(ids, signs)]
        dominators = []
        for other in complete:
            b = [other['values'][key] * sign for key, sign in zip(ids, signs)]
            if all(y >= x for x, y in zip(a, b)) and any(y > x for x, y in zip(a, b)):
                dominators.append(other['id'])
        dominated_by[row['id']] = dominators
    return dict(frontier=[row['id'] for row in complete if not dominated_by[row['id']]],
                dominated_by=dominated_by, unresolved=unresolved, complete=not unresolved,
                population='complete_feasible_alternatives', comparison='exact_numeric')


def _analysis(spec, rows, ids, profile):
    result = solve([[row['values'][key] for key in ids] for row in rows],
                   [profile['values'][key] for key in ids],
                   [c['direction'] for c in spec['criteria']], spec['method'],
                   **spec.get('method_params', {}))
    names = [row['id'] for row in rows]
    metadata = deepcopy(result.metadata)
    if 'compromise_set' in metadata:
        metadata['compromise_set_ids'] = [names[i] for i in metadata['compromise_set']]
    return dict(profile=deepcopy(profile), scores=dict(zip(names, result.scores)),
                ranking=[[names[i] for i in group] for group in result.ranking],
                decision=names[result.winner] if result.winner is not None else None,
                metadata=metadata)


def _check_proposal(proposal, report, names):
    if not isinstance(proposal, dict) or not proposal or set(proposal) - {
        'decision', 'scores', 'ranking', 'pareto_frontier'
    }:
        raise ValueError('proposal may contain only decision, scores, ranking, pareto_frontier')
    checks = report['checks']
    complete = report['analysis_status'] not in {
        'missing_evidence', 'preferences_unknown', 'preference_uncertainty'
    }
    if 'decision' in proposal:
        candidate = proposal['decision']
        if candidate is not None and (not isinstance(candidate, str) or candidate not in names):
            raise ValueError('proposal decision must identify an alternative or be null')
        checks['constraints_satisfied'] = (
            report['feasibility'][candidate] == 'feasible' if candidate is not None else None)
        checks['decision_verified'] = candidate == report['decision'] if complete else None
    if 'scores' in proposal:
        scores = proposal['scores']
        feasible = {key for key, state in report['feasibility'].items() if state == 'feasible'}
        if not isinstance(scores, dict) or set(scores) != feasible:
            raise ValueError('proposal scores must cover exactly the feasible alternatives')
        for value in scores.values():
            number(value, 'proposed score')
        if len(report['analyses']) == 1:
            expected = report['analyses'][0]['scores']
            checks['scores_verified'] = all(np.isclose(scores[key], expected[key],
                atol=SCORE_ATOL, rtol=SCORE_RTOL) for key in expected)
    if 'ranking' in proposal:
        ranking = proposal['ranking']
        if (not isinstance(ranking, list) or not ranking or
                any(not isinstance(group, list) or not group for group in ranking)):
            raise ValueError('proposal ranking must contain nonempty tie groups')
        flattened = [name for group in ranking for name in group]
        if any(not isinstance(name, str) for name in flattened):
            raise ValueError('proposal ranking ids must be strings')
        feasible = {key for key, state in report['feasibility'].items() if state == 'feasible'}
        if len(set(flattened)) != len(flattened) or set(flattened) != feasible:
            raise ValueError('proposal ranking must cover feasible alternatives once')
        if len(report['analyses']) == 1:
            expected = report['analyses'][0]['ranking']
            checks['ranking_verified'] = [set(g) for g in ranking] == [set(g) for g in expected]
    if 'pareto_frontier' in proposal:
        frontier = proposal['pareto_frontier']
        if (not isinstance(frontier, list) or any(not isinstance(key, str) for key in frontier)
                or len(set(frontier)) != len(frontier) or set(frontier) - names):
            raise ValueError('proposal frontier must contain distinct known ids')
        if report['pareto']['complete']:
            checks['pareto_verified'] = set(frontier) == set(report['pareto']['frontier'])
    required = {'decision': 'decision_verified', 'scores': 'scores_verified',
                'ranking': 'ranking_verified', 'pareto_frontier': 'pareto_verified'}
    evaluated = [checks[required[key]] for key in proposal]
    if False in evaluated or checks['constraints_satisfied'] is False:
        report['status'] = 'rejected'
    elif all(flag is True for flag in evaluated):
        report['status'] = 'verified'
    else:
        report['status'] = 'unverifiable'


def verify_decision(spec, proposal=None, *, policies=None, learned=None, scenarios=None):
    """Return a JSON trace; never execute workloads or call a model/provider.

    Scenarios are explicit user weight requests, recomputed independently. Their
    winner frequency measures the supplied scenarios, not calibrated confidence.
    """
    ids = validate_decision(spec)
    weights = resolve_weights(spec, policies=policies, learned=learned)
    feasibility, constraint_evidence = _feasibility(spec)
    pareto = _pareto(spec, feasibility, ids)
    rows = [r for r in spec['alternatives'] if feasibility[r['id']] == 'feasible']
    unknown = any(state == 'unknown' for state in feasibility.values())
    analyses, decision = [], None
    if unknown:
        status = 'missing_evidence'
    elif not rows:
        status = 'no_feasible_alternative'
    elif len(rows) == 1:
        decision, status = rows[0]['id'], 'single_feasible_alternative'
    elif not pareto['complete']:
        status = 'missing_evidence'
    elif weights['status'] != 'resolved':
        status = 'preferences_unknown'
    else:
        analyses = [_analysis(spec, rows, ids, profile) for profile in weights['profiles']]
        winners = {result['decision'] for result in analyses}
        if len(winners) == 1 and None not in winners:
            decision, status = next(iter(winners)), 'analysis_complete'
        elif len(analyses) > 1:
            status = 'preference_uncertainty'
        else:
            status = 'no_unique_winner'
    report = dict(schema_version='ordivo.decision.v2.1', status=status, analysis_status=status,
                  decision=decision, method=spec['method'], weights=weights,
                  feasibility=feasibility, constraint_evidence=constraint_evidence,
                  pareto=pareto, analyses=analyses,
                  checks=dict(constraints_satisfied=None, decision_verified=None,
                              scores_verified=None, ranking_verified=None, pareto_verified=None),
                  tolerances=dict(score_atol=SCORE_ATOL, score_rtol=SCORE_RTOL,
                                  solver_tie_atol=1e-12),
                  sensitivity=dict(tested=False, scenario_count=0, winner_stability=None, results=[]))
    if proposal is not None:
        _check_proposal(proposal, report, {r['id'] for r in spec['alternatives']})
    if scenarios is not None:
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError('scenarios must be a nonempty list')
        results = []
        for scenario in scenarios:
            if not isinstance(scenario, dict) or scenario.get('mode') != 'user':
                raise ValueError('sensitivity scenarios require explicit user weight vectors')
            variant = deepcopy(spec)
            variant['weights'] = scenario
            result = verify_decision(variant)
            results.append({key: result[key] for key in (
                'analysis_status', 'decision', 'weights', 'analyses', 'input_sha256')})
        complete = all(r['analysis_status'] in {'analysis_complete', 'no_unique_winner',
                       'single_feasible_alternative', 'no_feasible_alternative'} for r in results)
        stability = (sum(r['decision'] == decision for r in results) / len(results)
                     if complete and decision is not None else None)
        report['sensitivity'] = dict(tested=True, scenario_count=len(results),
                                     winner_stability=stability, results=results,
                                     interpretation='unweighted_supplied_scenario_frequency')
    encoded = json.dumps({'spec': spec, 'resolved_weights': weights, 'scenarios': scenarios},
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    report['input_sha256'] = hashlib.sha256(encoded).hexdigest()
    # Convert NumPy scalar predicates to ordinary JSON booleans at the boundary.
    return json.loads(json.dumps(report, default=lambda v: v.item(), allow_nan=False))
