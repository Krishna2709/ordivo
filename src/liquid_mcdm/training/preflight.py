"""Small-file tokenizer inspection; never fetches model weights or remote Python."""
import hashlib
import json
from pathlib import Path
import urllib.request

from liquid_mcdm.data import load_dataset
from .rendering import PROMPT_VERSION, render_example

MODEL_ID = 'LiquidAI/d1-3B'
REVISION = '051bcc464b01b9f92942b364d9586b0ef5912432'
TOKENIZER_FILES = ('tokenizer.json','tokenizer_config.json','config.json','chat_template.jinja')


def load_pinned_tokenizer(cache):
    from transformers import AutoTokenizer
    cache = Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    receipt_path = cache/'source_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('model') != MODEL_ID or receipt.get('revision') != REVISION:
            raise ValueError('tokenizer cache revision mismatch')
        for name,digest in receipt['sha256'].items():
            if hashlib.sha256((cache/name).read_bytes()).hexdigest() != digest:
                raise ValueError('tokenizer cache hash mismatch')
    else:
        hashes={}
        for name in TOKENIZER_FILES:
            url=f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
            data=urllib.request.urlopen(url,timeout=60).read()
            (cache/name).write_bytes(data)
            hashes[name]=hashlib.sha256(data).hexdigest()
        receipt={'model':MODEL_ID,'revision':REVISION,'sha256':hashes,'weights_downloaded':False}
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return AutoTokenizer.from_pretrained(cache,local_files_only=True,trust_remote_code=False),receipt


def tokenizer_preflight(data,cache):
    tokenizer,receipt=load_pinned_tokenizer(cache)
    lengths=[]
    counts={}
    excluded=0
    for split in ('train','val','test'):
        counts[split]=0
        for e in load_dataset(data,split):
            if e['target']['winner'] is None:
                excluded+=1
                continue
            row=render_example(e,tokenizer)
            lengths.append(len(row['input_ids']))
            counts[split]+=1
    return {'model':MODEL_ID,'revision':REVISION,'prompt_version':PROMPT_VERSION,
            'source_receipt':receipt,'eligible_by_split':counts,'excluded_nonunique':excluded,
            'max_tokens':max(lengths,default=0),'checked_examples':len(lengths),
            'alias_eos_boundary_passed':bool(lengths),
            'reference_labels_prompt_masked':True,
            'trl_collator_executed':False,'model_loaded':False,'training_validated':False}
