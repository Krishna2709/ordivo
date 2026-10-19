from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Result:
    scores: tuple[float, ...]
    ranking: tuple[tuple[int, ...], ...]  # ordered tie groups, best first
    winner: int | None
    metadata: dict[str, Any]


def _groups(scores, higher, atol):
    order = np.argsort(-scores if higher else scores, kind='stable')
    groups = []
    for i in order:
        if not groups or abs(scores[i] - scores[groups[-1][0]]) > atol:
            groups.append([int(i)])
        else:
            groups[-1].append(int(i))
    return tuple(tuple(sorted(group)) for group in groups)


def _range_normalize(values):
    span = np.ptp(values)
    return (values - values.min()) / span if span > 32 * np.finfo(float).eps else np.zeros_like(values)


def solve(matrix, weights, directions, method, *, lambda_=0.5, v=0.5, tie_atol=1e-12):
    """Return raw scores, ordered tie groups and a defensible unique winner.

    Weights are normalized to sum one. COPRAS/WASPAS require positive data.
    TOPSIS/VIKOR accept any finite values. Missing values are rejected.
    VIKOR winner is set only when advantage AND stability conditions hold.
    """
    x = np.asarray(matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    d = np.asarray(directions)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError('matrix requires at least two alternatives and one criterion')
    if w.shape != (x.shape[1],) or d.shape != w.shape:
        raise ValueError('weights and directions must match matrix columns')
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('missing or non-finite data is unsupported')
    if (w < 0).any() or not (w > 0).any():
        raise ValueError('weights must be nonnegative with positive total')
    if not np.isin(d, ['min', 'max']).all():
        raise ValueError('directions must be min or max')
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError('tie_atol must be finite and nonnegative')
    method = str(method).upper()
    if method not in ('COPRAS', 'WASPAS', 'VIKOR', 'TOPSIS'):
        raise ValueError('unknown MCDM method')
    for name, value in [('lambda_', lambda_), ('v', v)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    w = w / w.max()
    w = w / w.sum()
    benefit = d == 'max'
    metadata = {'method': method, 'higher_is_better': method != 'VIKOR',
                'tie_atol': tie_atol, 'weights': w.tolist(),
                'constant_criteria': np.flatnonzero(np.ptp(x, axis=0) == 0).tolist()}
    if method in ('COPRAS', 'WASPAS') and (x <= 0).any():
        raise ValueError(f'{method} requires strictly positive values')
    # Column rescaling protects norms and sums without changing these formulas.
    scale = np.max(np.abs(x), axis=0)
    scaled = np.divide(x, scale, out=np.zeros_like(x), where=scale != 0)
    if method == 'COPRAS':
        weighted = scaled / scaled.sum(axis=0) * w
        profit = weighted[:, benefit].sum(axis=1)
        cost = weighted[:, ~benefit].sum(axis=1)
        if np.any(w[~benefit] > 0):
            inverse = cost.min() / cost
            scores = profit + cost.sum() * inverse / inverse.sum()
        else:
            scores = profit
        metadata['normalization'] = 'column_sum'
    elif method == 'WASPAS':
        normalized = np.where(benefit, scaled, scaled.min(axis=0) / scaled)
        product = np.exp((np.log(normalized) * w).sum(axis=1))
        scores = lambda_ * (normalized @ w) + (1 - lambda_) * product
        metadata.update(normalization='max_benefit_min_cost_ratio', lambda_=lambda_)
    elif method == 'TOPSIS':
        norm = np.linalg.norm(scaled, axis=0)
        weighted = np.divide(scaled, norm, out=np.zeros_like(x), where=norm != 0) * w
        best = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
        worst = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
        dp = np.hypot.reduce(weighted - best, axis=1)
        dm = np.hypot.reduce(weighted - worst, axis=1)
        scores = np.divide(dm, dp + dm, out=np.full(x.shape[0], .5), where=dp + dm != 0)
        metadata['normalization'] = 'vector'
    else:
        best = np.where(benefit, scaled.max(axis=0), scaled.min(axis=0))
        worst = np.where(benefit, scaled.min(axis=0), scaled.max(axis=0))
        regret = np.divide(best - scaled, best - worst,
                           out=np.zeros_like(x), where=best != worst) * w
        s, r = regret.sum(axis=1), regret.max(axis=1)
        scores = v * _range_normalize(s) + (1 - v) * _range_normalize(r)
        order = np.argsort(scores, kind='stable')
        first, second = order[:2]
        dq = 1 / (len(scores) - 1)
        advantage = bool(scores[second] - scores[first] >= dq - tie_atol)
        stability = bool(s[first] <= s.min() + tie_atol or r[first] <= r.min() + tie_atol)
        if not advantage:
            compromise = np.flatnonzero(scores - scores[first] < dq - tie_atol).tolist()
            # A zero tolerance boundary must still include the minimum.
            compromise = sorted(set(compromise) | {int(first)})
        elif not stability:
            cutoff = scores[second]
            compromise = np.flatnonzero(scores <= cutoff + tie_atol).tolist()
        else:
            compromise = [int(first)]
        metadata.update(normalization='criterion_range_regret', v=v, S=s.tolist(), R=r.tolist(),
                        acceptable_advantage=advantage, acceptable_stability=stability,
                        advantage_threshold=dq, compromise_set=compromise)
    if not np.isfinite(scores).all():
        raise ValueError('numerical range exceeded; rescale source units')
    ranking = _groups(scores, method != 'VIKOR', tie_atol)
    winner = ranking[0][0] if len(ranking[0]) == 1 else None
    if method == 'VIKOR' and not (metadata['acceptable_advantage'] and metadata['acceptable_stability']):
        winner = None
    ordered = np.sort(scores)
    gap = ordered[1] - ordered[0] if method == 'VIKOR' else ordered[-1] - ordered[-2]
    metadata.update(score_gap=float(gap), near_tie=bool(gap <= 1e-4))
    return Result(tuple(map(float, scores)), ranking, winner, metadata)
