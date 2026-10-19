"""Static bootstrap integrity checks; executed on the approved remote pod."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    'runpod_bootstrap', Path(__file__).resolve().parents[1] / 'scripts' / 'runpod_bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def archive(tmp_path, member='decision-engines/example.py'):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as source:
        source.writestr(member, 'example')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_digest_and_path_are_checked(tmp_path):
    path, digest = archive(tmp_path)
    bootstrap.verify_archive(path, digest)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='SHA256'):
        bootstrap.verify_archive(path, digest)


@pytest.mark.parametrize('member', ['../escape', '/absolute', 'decision-engines/.env'])
def test_unsafe_archive_members_are_rejected(tmp_path, member):
    path, digest = archive(tmp_path, member)
    with pytest.raises(RuntimeError, match='Unsafe'):
        bootstrap.verify_archive(path, digest)


def test_candidate_install_cannot_replace_cuda_stack(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'install': [{'metadata': {'name': 'torch'}}]}))
    with pytest.raises(RuntimeError, match='CUDA stack'):
        bootstrap.reject_torch_replacement(report)
