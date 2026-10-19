"""Single-choice metrics. These do not imply full-ranking performance."""
import numpy as np


def evaluate_predictions(episodes, predictions):
    rows = []
    excluded = 0
    for e in episodes:
        winner = e['target']['winner']
        if winner is None:
            excluded += 1
            continue
        if e['episode_id'] not in predictions:
            raise ValueError('missing eligible prediction')
        probs = predictions[e['episode_id']]
        keys, values = list(probs), np.asarray(list(probs.values()), dtype=float)
        if (winner not in keys or values.size < 2 or not np.isfinite(values).all()
                or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-8, rtol=0)):
            raise ValueError('invalid probability distribution')
        if 'alternatives' in e and set(keys) != {a['id'] for a in e['alternatives']}:
            raise ValueError('probability options do not match alternatives')
        predicted = keys[int(values.argmax())]  # documented first-option tie break
        target = np.array([float(k == winner) for k in keys])
        rows.append((e, float(predicted == winner), float(np.sum((values-target)**2)),
                     float(values.max())))
    by_method = {}
    for method in sorted({r[0]['method'] for r in rows}):
        subset = [r for r in rows if r[0]['method'] == method]
        by_method[method] = {'eligible': len(subset), 'accuracy': float(np.mean([r[1] for r in subset]))}
    ece = 0.
    for b in range(10):
        bucket = [r for r in rows if min(int(r[3]*10), 9) == b]
        if bucket:
            ece += len(bucket)/len(rows) * abs(np.mean([r[1] for r in bucket])-np.mean([r[3] for r in bucket]))
    slices = {}
    for field, extract in {
        'alternatives_count': lambda e: len(e.get('alternatives', [])),
        'criteria_count': lambda e: len(e.get('criteria', [])),
        'scenario_family': lambda e: e.get('context', {}).get('scenario_family', 'unknown'),
        'near_tie': lambda e: e['target'].get('metadata', {}).get('near_tie', False),
        'weighting_scheme': lambda e: e.get('generation', {}).get('weighting_scheme', 'unknown'),
    }.items():
        groups = {}
        for row in rows:
            groups.setdefault(str(extract(row[0])), []).append(row[1])
        slices[field] = {key: {'eligible': len(v), 'accuracy': float(np.mean(v))} for key, v in groups.items()}
    return {'eligible': len(rows), 'excluded_nonunique': excluded,
            'accuracy': float(np.mean([r[1] for r in rows])) if rows else None,
            'brier_multiclass': float(np.mean([r[2] for r in rows])) if rows else None,
            'ece_10_bins': float(ece) if rows else None, 'by_method': by_method,
            'slices': slices, 'prediction_tie_break': 'first_option'}


def baseline_report(episodes):
    uniform, teacher = {}, {}
    for e in episodes:
        ids = [a['id'] for a in e['alternatives']]
        uniform[e['episode_id']] = {aid: 1/len(ids) for aid in ids}
        teacher[e['episode_id']] = {aid: float(aid == e['target']['winner']) for aid in ids}
    return {'scope': 'CPU sanity controls; no learned model evaluated',
            'uniform_first_option': evaluate_predictions(episodes, uniform),
            'exact_teacher_self_consistency': evaluate_predictions(episodes, teacher)}


def evaluate_dataset(directory, split='test'):
    import json
    from pathlib import Path
    from liquid_mcdm.data import load_dataset
    from liquid_mcdm.data.synthetic import sha256
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    episodes=load_dataset(directory,split)
    report=baseline_report(episodes)
    report['evaluation']={'version':'winner-eval-v1','split':split,
                          'split_sha256':manifest['splits'][split]['sha256'],
                          'manifest_sha256':sha256(directory/'manifest.json'),
                          'count':len(episodes)}
    return report
