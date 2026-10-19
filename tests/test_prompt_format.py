from copy import deepcopy
from pathlib import Path
import pytest

from liquid_mcdm.data import generate_dataset, load_dataset
from liquid_mcdm.training.rendering import render_example, option_probabilities


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    cache=Path('.cache/d1-tokenizer')
    if not (cache/'tokenizer.json').exists():
        pytest.skip('run tokenizer-check to fetch the pinned tokenizer; never fetch weights')
    return AutoTokenizer.from_pretrained(cache,local_files_only=True)


def test_real_tokenizer_prompt_boundary_mask_and_no_label_leak(tokenizer,tmp_path):
    generate_dataset(tmp_path/'data',count=40,seed=17)
    e=next(e for e in load_dataset(tmp_path/'data','train') if e['target']['winner'])
    row=render_example(e,tokenizer)
    assert 'raw_scores' not in row['prompt'] and 'target' not in row['prompt']
    assert row['prompt'].endswith('<|im_start|>assistant\n')
    assert row['completion'].endswith(tokenizer.eos_token)
    assert row['supervised_ids']==[row['alias_token_id'], tokenizer.eos_token_id]
    assert all(v == -100 for v in row['labels'][:-2])
    assert row['labels'][-2:]==row['supervised_ids']
    value=next(iter(e['alternatives'][0]['values'].values()))
    assert repr(value) in row['prompt']  # no rounding-label mismatch
    changed=deepcopy(e)
    changed['alternatives'].reverse()
    permuted=render_example(changed,tokenizer)
    assert permuted['winner']==row['winner']
    assert set(permuted['option_ids'])==set(row['option_ids'])


def test_nonunique_teacher_is_rejected(tokenizer,tmp_path):
    generate_dataset(tmp_path/'data',count=40,seed=9)
    e=load_dataset(tmp_path/'data','train')[0]
    e['target']['winner']=None
    with pytest.raises(ValueError):
        render_example(e,tokenizer)


def test_max_pool_leading_space_before_option_softmax(tokenizer):
    # A base logit loses to B, but the supported leading-space A wins.
    logits={41:0.,334:2.,42:1.,378:0.}
    probabilities=option_probabilities(tokenizer,['A','B'],logits)
    assert probabilities['A']==pytest.approx(.7310585786)
    assert probabilities['B']==pytest.approx(.2689414214)
