"""Contract checks for the matched C1/R0 continuation, without model loads."""
import copy
import pytest
import torch

from scripts.train_v2_continuation import (
    _adapter_digest, _choice_brier, _matched_schedule, _verify_frozen_inputs,
)


def _receipt():
    return {
        'status': 'pilot_completed',
        'source_identity': {'commit': 'a' * 40, 'archive_sha256': 'b' * 64},
        'config_sha256': 'c' * 64,
        'prompt_version': 'behavioral-choice-v1',
        'model_id': 'LiquidAI/d1-3B',
        'revision': '051bcc464b01b9f92942b364d9586b0ef5912432',
        'source_provenance': {
            'swissmetro': {
                'canonical_source': 'swissmetro',
                'manifest_sha256': 'd' * 64,
                'selected_split_unit_sha256': {'train': 'e' * 64, 'validation': 'f' * 64},
                'feature_policy': {'context_features': ['purpose'],
                                   'alternative_features': ['cost_chf']},
            },
            'esci': {
                'canonical_source': 'amazon_esci',
                'manifest_sha256': '1' * 64,
                'selected_split_unit_sha256': {'train': '2' * 64, 'validation': '3' * 64},
                'feature_policy': {'context_features': ['query'],
                                   'alternative_features': ['product_title']},
            },
        },
    }


def _config():
    receipt = _receipt()
    return {
        'event': 'stated_choice',
        'candidate_population': 'eligible_stated_choice_group',
        'pilot_receipt_sha256': '4' * 64,
        'starting_adapter_sha256': '5' * 64,
        'pilot_source_commit': receipt['source_identity']['commit'],
        'pilot_source_archive_sha256': receipt['source_identity']['archive_sha256'],
        'pilot_config_sha256': receipt['config_sha256'],
        'sources': [
            {'name': name, 'expected_source': value['canonical_source'],
             'feature_policy': value['feature_policy'],
             'manifest_sha256': value['manifest_sha256'],
             'selected_split_unit_sha256': value['selected_split_unit_sha256']}
            for name, value in receipt['source_provenance'].items()
        ],
    }


def test_frozen_bindings_reject_event_population_manifest_and_adapter_drift():
    receipt, config = _receipt(), _config()
    _verify_frozen_inputs(config, receipt, '4' * 64, '5' * 64)
    for change in ('event', 'manifest', 'policy', 'selected', 'adapter', 'receipt'):
        bad = copy.deepcopy(config)
        actual_receipt_sha, actual_adapter_sha = '4' * 64, '5' * 64
        if change == 'event':
            bad['event'] = 'conditional_booking_choice'
        elif change == 'manifest':
            bad['sources'][0]['manifest_sha256'] = '0' * 64
        elif change == 'policy':
            bad['sources'][0]['feature_policy']['context_features'] = ['post_choice']
        elif change == 'selected':
            bad['sources'][1]['selected_split_unit_sha256']['validation'] = '0' * 64
        elif change == 'adapter':
            actual_adapter_sha = '0' * 64
        else:
            actual_receipt_sha = '0' * 64
        with pytest.raises(ValueError):
            _verify_frozen_inputs(bad, receipt, actual_receipt_sha, actual_adapter_sha)


def test_schedule_is_matched_bounded_and_balanced():
    first = _matched_schedule({'swissmetro': 9, 'esci': 5}, seed=42, steps=40)
    assert first == _matched_schedule({'swissmetro': 9, 'esci': 5}, seed=42, steps=40)
    assert len(first) == 40
    assert [name for name, _ in first].count('swissmetro') == 20
    assert [name for name, _ in first].count('esci') == 20
    assert len(set(first[:18])) == 14
    assert first != _matched_schedule({'swissmetro': 9, 'esci': 5}, seed=43, steps=40)


def test_direct_brier_masks_unavailable_options_and_has_gradient():
    scores = torch.tensor([0.1, 100., -0.2], requires_grad=True)
    loss = _choice_brier(scores, torch.tensor([True, False, True]), 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad[1] == 0
    assert scores.grad[0] != 0
    with pytest.raises(ValueError):
        _choice_brier(scores, torch.tensor([True, False, True]), 1)


def test_adapter_digest_binds_paths_and_file_bytes(tmp_path):
    (tmp_path / 'adapter_config.json').write_text('{}')
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'weight')
    before = _adapter_digest(tmp_path)
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'other')
    assert _adapter_digest(tmp_path) != before
    (tmp_path / 'adapter_model.safetensors').unlink()
    with pytest.raises(ValueError):
        _adapter_digest(tmp_path)
