"""Strict numerical decision boundary; JSON data, never executable expressions."""
from math import isfinite
from numbers import Real


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def validate_decision(spec):
    if not isinstance(spec, dict):
        raise ValueError('decision must be an object')
    identifier(spec.get('scope'), 'scope')
    criteria, alternatives = spec.get('criteria'), spec.get('alternatives')
    if not isinstance(criteria, list) or not criteria:
        raise ValueError('criteria must be nonempty')
    ids = []
    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise ValueError('criterion must be an object')
        ids.append(identifier(criterion.get('id'), 'criterion id'))
        if criterion.get('direction') not in ('min', 'max'):
            raise ValueError('direction must be min or max')
    if len(set(ids)) != len(ids):
        raise ValueError('criterion ids must be unique')
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError('alternatives must be nonempty')
    names = []
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            raise ValueError('alternative must be an object')
        names.append(identifier(alternative.get('id'), 'alternative id'))
        values = alternative.get('values')
        if not isinstance(values, dict) or set(values) != set(ids):
            raise ValueError('values must name every criterion; use null for missing evidence')
        for name, value in values.items():
            if value is not None:
                number(value, name)
    if len(set(names)) != len(names):
        raise ValueError('alternative ids must be unique')
    constraints = spec.get('constraints', [])
    if not isinstance(constraints, list):
        raise ValueError('constraints must be a list')
    for constraint in constraints:
        if not isinstance(constraint, dict) or set(constraint) != {'criterion', 'op', 'value'}:
            raise ValueError('constraint requires criterion, op, value')
        if constraint['criterion'] not in ids or constraint['op'] not in ('<=', '>=', '<', '>', '=='):
            raise ValueError('unknown constraint criterion or operator')
        number(constraint['value'], 'constraint value')
    if spec.get('method') not in ('COPRAS', 'WASPAS', 'VIKOR', 'TOPSIS'):
        raise ValueError('method must be COPRAS, WASPAS, VIKOR or TOPSIS')
    params = spec.get('method_params', {})
    allowed = {'WASPAS': {'lambda_'}, 'VIKOR': {'v'}}.get(spec['method'], set())
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError('unknown method parameter')
    for key, value in params.items():
        if not 0 <= number(value, key) <= 1:
            raise ValueError('method parameters must be in [0, 1]')
    return ids
