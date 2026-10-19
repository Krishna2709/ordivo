import numpy as np
import pytest

from liquid_mcdm.mcdm import solve

METHODS = ['COPRAS', 'WASPAS', 'VIKOR', 'TOPSIS']


@pytest.mark.parametrize('method,expected', [
    ('WASPAS', [1., .5]), ('TOPSIS', [1., 0.]),
    ('COPRAS', [2/3, 1/3]), ('VIKOR', [0., 1.]),
])
def test_hand_calculated_dominance(method, expected):
    # Equal weights, best alternative twice the benefit and half the cost.
    result = solve([[2., 1.], [1., 2.]], [.5, .5], ['max', 'min'], method)
    np.testing.assert_allclose(result.scores, expected)
    assert result.ranking == ((0,), (1,))
    assert result.winner == 0


@pytest.mark.parametrize('method', METHODS)
def test_permutation_and_positive_column_scale(method):
    x = np.array([[3., 7., 8.], [6., 4., 5.], [8., 9., 1.]])
    w, d = [.2, .3, .5], ['max', 'min', 'max']
    a = solve(x, w, d, method)
    p = [2, 0, 1]
    b = solve(x[p] * [1000, .01, 7], w, d, method)
    np.testing.assert_allclose(b.scores, np.array(a.scores)[p], atol=1e-12)


@pytest.mark.parametrize('method', METHODS)
def test_identical_alternatives_have_no_arbitrary_winner(method):
    r = solve([[2, 3], [2, 3], [2, 3]], [.5, .5], ['max', 'min'], method)
    assert r.ranking == ((0, 1, 2),)
    assert r.winner is None
    assert np.isfinite(r.scores).all()


@pytest.mark.parametrize('method', METHODS)
def test_zero_weight_is_irrelevant(method):
    r = solve([[2, 100], [1, 1]], [1, 0], ['max', 'min'], method)
    assert r.winner == 0


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('matrix,weights,directions', [
    ([[1, float('nan')], [2, 3]], [.5, .5], ['max', 'min']),
    ([[1, 2], [2, 3]], [0, 0], ['max', 'min']),
    ([[1, 2], [2, 3]], [-1, 2], ['max', 'min']),
    ([[1, 2], [2, 3]], [1], ['max', 'min']),
    ([[1, 2], [2, 3]], [.5, .5], ['up', 'min']),
    ([[1, 2]], [.5, .5], ['max', 'min']),
    ([[1, 2], [2, float('inf')]], [.5, .5], ['max', 'min']),
])
def test_invalid_inputs_rejected(method, matrix, weights, directions):
    with pytest.raises(ValueError):
        solve(matrix, weights, directions, method)


@pytest.mark.parametrize('method', ['WASPAS', 'COPRAS'])
def test_nonpositive_values_rejected_without_epsilon_repair(method):
    with pytest.raises(ValueError, match='positive'):
        solve([[0, 1], [2, 3]], [.5, .5], ['max', 'min'], method)


@pytest.mark.parametrize('method,key', [('WASPAS', 'lambda_'), ('VIKOR', 'v')])
def test_parameters_are_bounded(method, key):
    with pytest.raises(ValueError):
        solve([[1, 2], [3, 4]], [.5, .5], ['max', 'min'], method, **{key: 2})


def test_vikor_compromise_is_not_always_argmin():
    # Opposed extremes tie in S; middle wins Q but gap=.25 < 1/(3-1).
    r = solve([[10, 1], [1, 10], [6, 6]], [.5, .5], ['max', 'max'], 'VIKOR', v=0)
    assert r.winner == 2
    r = solve([[10, 1], [1, 10], [5.5, 5.5]], [.5, .5], ['max', 'max'], 'VIKOR', v=.75)
    assert r.ranking[0] == (2,)
    assert r.winner is None
    assert not r.metadata['acceptable_advantage']
    assert set(r.metadata['compromise_set']) == {0, 1, 2}


def test_method_disagreement_and_weight_sensitivity():
    x = [[9, 2], [5, 5], [2, 9]]
    a = solve(x, [.5, .5], ['max', 'max'], 'WASPAS', lambda_=0)
    assert a.winner == 1
    b = solve(x, [.9, .1], ['max', 'max'], 'WASPAS')
    assert b.winner == 0
    c = solve(x, [.5, .5], ['max', 'max'], 'WASPAS', lambda_=1)
    assert c.ranking[0] == (0, 2)


def test_direction_reversal():
    assert solve([[1], [2]], [1], ['min'], 'TOPSIS').winner == 0
    assert solve([[1], [2]], [1], ['max'], 'TOPSIS').winner == 1


def test_independent_library_crosschecks():
    from pymcdm.methods import WASPAS, VIKOR, TOPSIS
    from pymcdm.normalizations import vector_normalization
    rng = np.random.default_rng(92)
    for _ in range(10):
        x = rng.lognormal(size=(7, 4))
        w = rng.dirichlet(np.ones(4))
        d = ['max', 'min', 'min', 'max']
        for method, oracle in [('WASPAS', WASPAS(l=.3)),
                               ('VIKOR', VIKOR(v=.7)),
                               ('TOPSIS', TOPSIS(normalization_function=vector_normalization))]:
            params = {'lambda_': .3} if method == 'WASPAS' else {'v': .7} if method == 'VIKOR' else {}
            actual = np.array(solve(x, w, d, method, **params).scores)
            if method == 'COPRAS':
                actual = actual / actual.max()  # library returns relative utility
            expected = oracle(x, w, [1, -1, -1, 1])
            np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_copras_independent_pydecision_crosscheck():
    # Execute the audited standalone function unchanged, without its plot dependency.
    import ast
    from pathlib import Path
    source = Path(__file__).parent/'reference'/'copras.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'copras_method']
    namespace = {'np': np}
    exec(compile(tree, str(source), 'exec'), namespace)
    rng = np.random.default_rng(32)
    for _ in range(15):
        x = rng.lognormal(size=(6,4))
        w = rng.dirichlet(np.ones(4))
        d = ['max','min','max','min']
        actual = np.array(solve(x,w,d,'COPRAS').scores)
        expected = namespace['copras_method'](x,w,d,graph=False,verbose=False)[:,1]
        np.testing.assert_allclose(actual/actual.max(),expected,atol=1e-12)


def test_copras_cost_improvement_with_identical_benefits():
    r=solve([[1,1],[1,2]],[.5,.5],['max','min'],'COPRAS')
    np.testing.assert_allclose(r.scores,[7/12,5/12],atol=1e-12)
    assert r.winner == 0


def test_topsis_tiny_but_discriminating_weight():
    r=solve([[1,1],[1,2]],[1,1e-170],['max','max'],'TOPSIS')
    np.testing.assert_allclose(r.scores,[0,1],atol=1e-12)
    assert r.winner==1


def test_topsis_can_reverse_ranks_when_adding_dominated_alternative():
    a=solve([[28,13],[15,25]],[.5,.5],['max','max'],'TOPSIS')
    b=solve([[28,13],[15,25],[9,10]],[.5,.5],['max','max'],'TOPSIS')
    assert a.scores[0] < a.scores[1]
    assert b.scores[0] > b.scores[1]


def test_vikor_stability_failure_includes_tied_second_candidates():
    x=[[1,13,14],[21,27,21],[2,17,2],[7,19,1]]
    r=solve(x,[.3,.4,.3],['max','min','max'],'VIKOR')
    assert r.metadata['acceptable_advantage']
    assert not r.metadata['acceptable_stability']
    assert r.winner is None
    assert r.metadata['compromise_set']==[0,1,2]
