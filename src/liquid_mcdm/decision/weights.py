"""Resolve declared weighting conventions without inventing human priorities."""
from copy import deepcopy

import numpy as np

from .specification import identifier, number, validate_decision


def normalize(values, ids):
    if not isinstance(values, dict) or set(values) != set(ids):
        raise ValueError('weights must name exactly every criterion')
    w = np.array([number(values[key], 'weight') for key in ids])
    if (w < 0).any() or not (w > 0).any():
        raise ValueError('weights must be nonnegative with positive total')
    w = w / w.max()
    w /= w.sum()
    return dict(zip(ids, w.tolist()))


def _unknown(source, reason):
    return dict(status='preferences_unknown', source=source, profiles=[], reason=reason)


def _objective(spec, method, ids):
    rows = [a['values'] for a in spec['alternatives']]
    if len(rows) < 2 or any(row[key] is None for row in rows for key in ids):
        return None, 'insufficient_complete_alternatives'
    x = np.array([[row[key] for key in ids] for row in rows], dtype=float)
    # Scaling before differences and sums protects finite extreme source units.
    scale = np.max(np.abs(x), axis=0)
    x = np.divide(x, scale, out=np.zeros_like(x), where=scale != 0)
    active = np.ptp(x, axis=0) > 0
    information = np.zeros(len(ids))
    if method == 'entropy':
        if (x < 0).any():
            raise ValueError('entropy_raw_nonnegative_v1 requires nonnegative values')
        if active.any():
            p = x[:, active] / x[:, active].sum(axis=0)
            logp = np.zeros_like(p)
            np.log(p, out=logp, where=p > 0)
            information[active] = np.maximum(0, 1 + (p * logp).sum(axis=0) / np.log(len(x)))
    elif active.sum() >= 2:
        z = (x[:, active] - x[:, active].min(axis=0)) / np.ptp(x[:, active], axis=0)
        costs = np.array([c['direction'] == 'min' for c in spec['criteria']])[active]
        z[:, costs] = 1 - z[:, costs]
        correlation = np.clip(np.corrcoef(z, rowvar=False), -1, 1)
        information[active] = z.std(axis=0, ddof=0) * (1 - correlation).sum(axis=1)
    if not np.isfinite(information).all() or information.sum() <= 1e-12:
        return None, 'degenerate_objective_information'
    return normalize(dict(zip(ids, information)), ids), None


def resolve_weights(spec, *, policies=None, learned=None):
    """Trusted registry/profile arguments are supplied by the application, not the model.

    `learned` is externally estimated evidence, not a preference learner implemented here.
    Objective weights describe the supplied alternative set, including infeasible rows.
    """
    ids = validate_decision(spec)
    request = spec.get('weights')
    if request is None:
        return _unknown({'mode': 'unknown'}, 'no_preference_evidence')
    if not isinstance(request, dict):
        raise ValueError('weights must be an object or null')
    mode = request.get('mode')
    source = {'mode': mode}
    if mode == 'unknown':
        return _unknown(source, 'no_preference_evidence')
    if mode == 'user':
        values = normalize(request.get('values'), ids)
    elif mode == 'approved_policy':
        key = identifier(request.get('policy_id'), 'policy_id')
        version = identifier(request.get('version'), 'policy version')
        source.update(policy_id=key, version=version, scope=spec['scope'])
        policy = (policies or {}).get(key)
        if policy is None:
            return _unknown(source, 'policy_not_found')
        if policy.get('scope') != spec['scope']:
            raise ValueError('policy scope mismatch')
        if policy.get('version') != version:
            raise ValueError('policy version mismatch')
        values = normalize(policy.get('values'), ids)
    elif mode == 'learned':
        if learned is None:
            return _unknown(source, 'learned_evidence_unavailable')
        if learned.get('scope') != spec['scope']:
            raise ValueError('learned evidence scope mismatch')
        source.update(evidence_ref=identifier(learned.get('evidence_ref'), 'evidence_ref'),
                      scope=spec['scope'])
        profiles = learned.get('profiles')
        if not isinstance(profiles, list) or not profiles:
            raise ValueError('learned profiles must be nonempty')
        resolved = []
        for profile in profiles:
            probability = number(profile.get('probability'), 'profile probability')
            if not 0 < probability <= 1:
                raise ValueError('profile probabilities must be in (0, 1]')
            resolved.append({'values': normalize(profile.get('values'), ids),
                             'probability': probability})
        if not np.isclose(sum(p['probability'] for p in resolved), 1, atol=1e-12, rtol=0):
            raise ValueError('profile probabilities must sum to one')
        return dict(status='resolved', source=source, profiles=resolved)
    elif mode == 'objective':
        method = request.get('method')
        if method not in ('entropy', 'critic'):
            raise ValueError('objective method must be entropy or critic')
        source.update(method=method, population='all_supplied_alternatives', convention=(
            'entropy_raw_nonnegative_v1' if method == 'entropy'
            else 'critic_benefit_minmax_population_v1'))
        values, reason = _objective(spec, method, ids)
        if values is None:
            return _unknown(source, reason)
    else:
        raise ValueError('unknown weight mode')
    return dict(status='resolved', source=deepcopy(source),
                profiles=[{'values': values, 'probability': 1.0}])
