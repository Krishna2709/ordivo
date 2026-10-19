import hashlib
from copy import deepcopy

import pytest

from liquid_mcdm.data import generate_dataset, validate_episode, load_dataset


def test_reproducibility_split_families_and_teacher(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    ma = generate_dataset(a, count=100, seed=17)
    mb = generate_dataset(b, count=100, seed=17)
    assert ma == mb
    families = {}
    ids = set()
    for split in ('train', 'val', 'test'):
        assert (a/f'{split}.jsonl').read_bytes() == (b/f'{split}.jsonl').read_bytes()
        assert ma['splits'][split]['sha256'] == hashlib.sha256((a/f'{split}.jsonl').read_bytes()).hexdigest()
        episodes = load_dataset(a, split)
        families[split] = {e['context']['scenario_family'] for e in episodes}
        for e in episodes:
            validate_episode(e)
            assert e['episode_id'] not in ids
            ids.add(e['episode_id'])
            assert e['source'] == 'synthetic_solver'
    assert len(ids) == 100
    assert not families['train'] & families['val']
    assert not families['train'] & families['test']
    assert not families['val'] & families['test']


def test_corruption_is_detected(tmp_path):
    generate_dataset(tmp_path/'data', count=20, seed=9)
    e = load_dataset(tmp_path/'data', 'train')[0]
    key = next(iter(e['target']['raw_scores']))
    e['target']['raw_scores'][key] += .1
    with pytest.raises(ValueError, match='target'):
        validate_episode(e)
    p = tmp_path/'data'/'train.jsonl'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError, match='hash'):
        load_dataset(tmp_path/'data', 'train')


def test_teacher_validation_is_stable_under_alternative_reordering(tmp_path):
    generate_dataset(tmp_path/'data', count=40, seed=17)
    episode = deepcopy(load_dataset(tmp_path/'data', 'train')[0])
    episode['alternatives'].reverse()
    result = validate_episode(episode)
    ids = [alternative['id'] for alternative in episode['alternatives']]
    assert ids[result.winner] == episode['target']['winner']


def test_generator_will_not_overwrite_outputs(tmp_path):
    out = tmp_path/'data'
    generate_dataset(out, count=20, seed=9)
    with pytest.raises(FileExistsError):
        generate_dataset(out, count=20, seed=9)


def test_small_or_negative_request_rejected(tmp_path):
    for count, seed in [(0,1), (2,1), (20,-1)]:
        with pytest.raises(ValueError):
            generate_dataset(tmp_path/f'{count}-{seed}', count=count, seed=seed)


def test_each_family_covers_methods_and_weighting_schemes(tmp_path):
    generate_dataset(tmp_path/'data',count=1000,seed=42)
    for split in ('train','val','test'):
        rows=load_dataset(tmp_path/'data',split)
        for family in {e['context']['scenario_family'] for e in rows}:
            subset=[e for e in rows if e['context']['scenario_family']==family]
            assert {e['method'] for e in subset}=={'COPRAS','WASPAS','VIKOR','TOPSIS'}
            assert {e['generation']['weighting_scheme'] for e in subset}=={'equal','dirichlet','sparse'}
