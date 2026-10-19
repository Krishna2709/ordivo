"""Seeded solver episodes. Family splits are fixed before sampling/augmentation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from liquid_mcdm.mcdm import solve

VERSION = 'synthetic-v1'
FAMILIES = {
    'train': ('iid_uniform', 'correlated_quality', 'lognormal_tradeoff'),
    'val': ('clustered',),
    'test': ('anticorrelated', 'near_tie'),
}
METHODS = ('COPRAS', 'WASPAS', 'VIKOR', 'TOPSIS')


def _metadata_equal(left, right):
    """Compare semantic metadata while tolerating harmless float reduction order."""
    if isinstance(left, dict) and isinstance(right, dict):
        return (left.keys() == right.keys()
                and all(_metadata_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _metadata_equal(a, b) for a, b in zip(left, right))
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return bool(np.isclose(left, right, rtol=1e-12, atol=1e-14))
    return left == right


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _episode(seed, split_index, index, family, scheme):
    rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, index]))
    n, m = int(rng.integers(3, 9)), int(rng.integers(2, 7))
    if family == 'iid_uniform':
        x = rng.uniform(.2, 2, (n, m))
    elif family == 'correlated_quality':
        x = np.exp(rng.normal(0, .7, (n, 1)) + rng.normal(0, .2, (n, m)))
    elif family == 'lognormal_tradeoff':
        x = rng.lognormal(0, 1.1, (n, m))
    elif family == 'clustered':
        x = np.exp(rng.choice([-.7, .7], (n, 1)) + rng.normal(0, .15, (n, m)))
    elif family == 'anticorrelated':
        z = rng.normal(0, .7, (n, 1))
        x = np.exp(z * np.where(np.arange(m) % 2, -1, 1) + rng.normal(0, .1, (n, m)))
    else:
        x = np.exp(rng.normal(0, .5, (1, m)) + rng.normal(0, 1e-7, (n, m)))
    scales = 10. ** rng.uniform(-2, 3, m)
    x *= scales
    if scheme == 'equal':
        w = np.ones(m) / m
    elif scheme == 'dirichlet':
        w = rng.dirichlet(np.ones(m))
    else:
        w = rng.dirichlet(np.full(m, .3))
        w[rng.integers(m)] = 0
        w /= w.sum()
    directions = rng.choice(['min', 'max'], m).tolist()
    method = METHODS[index % len(METHODS)]
    params = {'lambda': float(rng.choice([0., .5, 1.]))} if method == 'WASPAS' else (
        {'v': float(rng.choice([0., .5, 1.]))} if method == 'VIKOR' else {})
    kwargs = {'lambda_' if k == 'lambda' else k: v for k, v in params.items()}
    result = solve(x, w, directions, method, **kwargs)
    names = [f'criterion_{j}' for j in range(m)]
    ids = [chr(65+i) for i in range(n)]
    return {
        'schema_version': 1, 'episode_id': f'{VERSION}-{seed}-{split_index}-{index:06d}',
        'source': 'synthetic_solver',
        'context': {'domain': 'abstract_multi_attribute', 'scenario_family': family},
        'generation': {'version': VERSION, 'seed_lineage': [seed, split_index, index],
                       'distribution': family, 'column_scales': scales.tolist(),
                       'weighting_scheme': scheme, 'augmentation': 'none'},
        'method': method, 'method_params': params,
        'criteria': [{'name': name, 'direction': d, 'weight': float(weight), 'unit': 'synthetic'}
                     for name, d, weight in zip(names, directions, w)],
        'alternatives': [{'id': aid, 'values': dict(zip(names, map(float, row)))}
                         for aid, row in zip(ids, x)],
        'target': {'algorithm': method,
                   'ranking': [[ids[i] for i in group] for group in result.ranking],
                   'winner': ids[result.winner] if result.winner is not None else None,
                   'raw_scores': dict(zip(ids, result.scores)),
                   'tie_policy': 'explicit_groups_atol_1e-12', 'metadata': result.metadata},
    }


def validate_episode(episode):
    """Recompute labels from unrounded source measurements and validate identities."""
    criteria, alternatives = episode['criteria'], episode['alternatives']
    names = [c['name'] for c in criteria]
    ids = [a['id'] for a in alternatives]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError('duplicate criterion or alternative identity')
    if any(set(a['values']) != set(names) for a in alternatives):
        raise ValueError('missing or extra criteria')
    if episode['source'] != 'synthetic_solver':
        raise ValueError('only synthetic solver provenance may be teacher-validated')
    params = {'lambda_' if k == 'lambda' else k: v for k, v in episode['method_params'].items()}
    result = solve([[a['values'][c] for c in names] for a in alternatives],
                   [c['weight'] for c in criteria], [c['direction'] for c in criteria],
                   episode['method'], **params)
    target = episode['target']
    expected_ranking = [[ids[i] for i in group] for group in result.ranking]
    expected_winner = ids[result.winner] if result.winner is not None else None
    if (target['algorithm'] != episode['method'] or target['ranking'] != expected_ranking
            or target['winner'] != expected_winner or set(target['raw_scores']) != set(ids)
            or not np.allclose([target['raw_scores'][i] for i in ids], result.scores,
                               rtol=1e-12, atol=1e-14)
            or not _metadata_equal(target['metadata'], result.metadata)):
        raise ValueError('target does not match exact solver')
    return result


def generate_dataset(output, *, count=1000, seed=42):
    if not isinstance(count, int) or count < 20 or not isinstance(seed, int) or seed < 0:
        raise ValueError('count must be an integer >=20 and seed a nonnegative integer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = [int(.8 * count), int(.1 * count)]
    counts.append(count - sum(counts))
    manifest = {'schema_version': 1, 'generator': VERSION, 'seed': seed, 'count': count,
                'numpy_version': np.__version__, 'family_assignments': FAMILIES,
                'split_policy': 'disjoint_families_before_sampling_no_augmentation',
                'label_provenance': 'synthetic_solver',
                'generator_source_sha256': sha256(Path(__file__)),
                'solver_source_sha256': sha256(Path(__file__).parents[1]/'mcdm'/'solvers.py'),
                'splits': {}}
    for split_index, (split, families) in enumerate(FAMILIES.items()):
        path = output / f'{split}.jsonl'
        with path.open('w') as f:
            for index in range(counts[split_index]):
                family = families[(index // len(METHODS)) % len(families)]
                scheme = ('equal', 'dirichlet', 'sparse')[(index // (len(METHODS)*len(families))) % 3]
                e = _episode(seed, split_index, index, family, scheme)
                validate_episode(e)
                f.write(canonical_json(e)+'\n')
        manifest['splits'][split] = {'count': counts[split_index], 'sha256': sha256(path)}
    # Round-trip JSON makes the returned object identical to the persisted manifest.
    manifest = json.loads(canonical_json(manifest))
    (output/'manifest.json').write_text(canonical_json(manifest)+'\n')
    return manifest


def load_dataset(directory, split):
    if split not in FAMILIES:
        raise ValueError('split must be train, val or test')
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    path = directory/f'{split}.jsonl'
    if sha256(path) != manifest['splits'][split]['sha256']:
        raise ValueError('dataset hash mismatch')
    episodes = [json.loads(line) for line in path.read_text().splitlines()]
    if len(episodes) != manifest['splits'][split]['count']:
        raise ValueError('dataset count mismatch')
    for episode in episodes:
        validate_episode(episode)
    return episodes
