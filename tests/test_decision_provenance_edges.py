"""Missing evidence must not become a failed or passed executed check."""
from liquid_mcdm.decision import verify_decision


def test_unknown_constraint_is_unverifiable_not_known_violation():
    spec = {
        'scope': 'example', 'criteria': [{'id': 'cost', 'direction': 'min'}],
        'alternatives': [{'id': 'A', 'values': {'cost': None}}],
        'constraints': [{'criterion': 'cost', 'op': '<=', 'value': 10}],
        'weights': None, 'method': 'TOPSIS',
    }
    result = verify_decision(spec, {'decision': 'A'})
    assert result['checks']['constraints_satisfied'] is None
    assert result['status'] == 'unverifiable'
