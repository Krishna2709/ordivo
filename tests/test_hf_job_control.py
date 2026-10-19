"""Mocked control-plane regressions. Never creates real resources; execute remotely."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def control(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('hf_control_under_test', scripts/'prepare_hf_job.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'STAGE', tmp_path/'stage')
    monkeypatch.setattr(module, 'CONFIG', tmp_path/'config.json')
    return module


def test_edit_during_packaging_cannot_certify_stale_archive(control, monkeypatch):
    (control.ROOT/'scripts').mkdir()
    (control.ROOT/'scripts/hf_job_bootstrap.py').write_text('bootstrap')
    control.CONFIG.write_text('{}')

    def package(path):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('decision-engines/scripts/hf_job_bootstrap.py', 'bootstrap')
            archive.writestr('decision-engines/configs/hf_jobs_smoke.json', '{}')
        return path

    monkeypatch.setattr(control, 'package', package)
    monkeypatch.setattr(control, 'source_identity', lambda: {'changed-during-packaging': 'new'})
    with pytest.raises(RuntimeError, match='changed during packaging'):
        control.prepare()
    assert not (control.STAGE/'prepared.json').exists()


def test_completed_submission_is_not_repeated_and_has_no_secrets(control, monkeypatch):
    control.STAGE.mkdir()
    (control.STAGE/'source.zip').write_bytes(b'snapshot')
    (control.STAGE/'hf_job_bootstrap.py').write_bytes(b'bootstrap')
    config = {'flavor': 't4-small', 'timeout_seconds': 600, 'image': 'pinned-image'}
    control.CONFIG.write_text(json.dumps(config))
    receipt = {'config': config, 'source_files': {'example.py': 'hash'},
               'source_sha256': control.digest(control.STAGE/'source.zip'),
               'bootstrap_sha256': control.digest(control.STAGE/'hf_job_bootstrap.py')}
    (control.STAGE/'prepared.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(control, 'source_identity', lambda: receipt['source_files'])
    calls = []

    def create_bucket(bucket, **kwargs):
        assert kwargs == {'private': True, 'exist_ok': False}
        assert (control.STAGE/'submission.json').exists()  # claimed before mutation
        calls.append('create')

    def upload(bucket, **kwargs):
        assert {dest for _, dest in kwargs['add']} == {
            'source.zip', 'hf_job_bootstrap.py', 'prepared.json'}
        calls.append('upload')

    def run_job(**kwargs):
        assert kwargs['timeout'] == 600
        assert 'secrets' not in kwargs
        assert 'HF_TOKEN' not in kwargs['env'] and 'WANDB_API_KEY' not in kwargs['env']
        calls.append('submit')
        return SimpleNamespace(id='mock-job', url='https://example.invalid/mock-job')

    api = SimpleNamespace(create_bucket=create_bucket, batch_bucket_files=upload, run_job=run_job)
    monkeypatch.setattr(control, 'preflight', lambda: (api, 'mock-user', {}))
    control.submit()
    assert calls == ['create', 'upload', 'submit']
    with pytest.raises(RuntimeError, match='already attempted'):
        control.submit()
    assert calls == ['create', 'upload', 'submit']
