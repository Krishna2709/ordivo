import hashlib
import json

import pytest
import torch

from liquid_mcdm.training.calibration import (fit_temperature,
                                              load_calibration_evidence)


def frozen_bundle(tmp_path):
    hashes = {}
    for split, ids in {'train': ['s:a'], 'validation': ['s:b'],
                       'calibration': ['s:c', 's:d'], 'test': ['s:e']}.items():
        content = ''.join(json.dumps({'episode_id': id_, 'source': 's',
                                      'group_id': id_.split(':')[1],
                                      'supervision': {}}) + '\n' for id_ in ids).encode()
        (tmp_path / f'{split}.jsonl').write_bytes(content)
        hashes[split] = hashlib.sha256(content).hexdigest()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'split_sha256': hashes}))
    return manifest


def test_temperature_fit_is_manifest_bound(tmp_path):
    evidence = load_calibration_evidence(
        frozen_bundle(tmp_path), event='conditional_booking_choice',
        candidate_population='exactly_one_booking_exposed_group')
    scores = torch.tensor([[8., 0., float('-inf')], [0., 8., 1.]])
    target = torch.tensor([0, 0])
    fitted = fit_temperature(scores, target, evidence=evidence, group_ids=['s:c', 's:d'])
    assert fitted.temperature > 1
    assert fitted.calibration_groups_count == 2
    before = torch.nn.functional.cross_entropy(scores, target)
    after_probs = fitted.apply(scores, event=evidence.event,
                               candidate_population=evidence.candidate_population,
                               manifest_sha256=evidence.manifest_sha256)
    assert after_probs[0, 2] == 0
    assert torch.nn.functional.nll_loss(after_probs.log(), target) < before
    for changed in ({'event': 'stated_choice'}, {'candidate_population': 'other'},
                    {'manifest_sha256': 'other'}):
        kwargs = {'event': evidence.event, 'candidate_population': evidence.candidate_population,
                  'manifest_sha256': evidence.manifest_sha256} | changed
        with pytest.raises(ValueError, match='mismatch'):
            fitted.apply(scores, **kwargs)
    with pytest.raises(ValueError, match='calibration split'):
        fit_temperature(scores, target, evidence=evidence, group_ids=['s:c', 's:e'])
    with pytest.raises(ValueError, match='unavailable'):
        fit_temperature(scores, torch.tensor([2, 0]), evidence=evidence,
                        group_ids=['s:c', 's:d'])


def test_manifest_hash_and_split_overlap_guard(tmp_path):
    manifest = frozen_bundle(tmp_path)
    (tmp_path / 'test.jsonl').write_text((tmp_path / 'test.jsonl').read_text() + ' ') 
    with pytest.raises(ValueError, match='frozen manifest'):
        load_calibration_evidence(manifest, event='stated_choice',
                                  candidate_population='eligible_stated_choice_group')
    manifest = frozen_bundle(tmp_path)
    (tmp_path / 'test.jsonl').write_bytes((tmp_path / 'calibration.jsonl').read_bytes())
    data = json.loads(manifest.read_text())
    data['split_sha256']['test'] = hashlib.sha256((tmp_path / 'test.jsonl').read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='group overlap'):
        load_calibration_evidence(manifest, event='stated_choice',
                                  candidate_population='eligible_stated_choice_group')
