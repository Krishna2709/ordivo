"""Remote-only tests of source integrity and safe archive handling."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    'hf_job_bootstrap', Path(__file__).resolve().parents[1]/'scripts/hf_job_bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def make_archive(tmp_path, name):
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr(name, 'source')
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.parametrize('name', ['../escape', '/absolute',
                                  'decision-engines/../escape', 'decision-engines/.env'])
def test_unsafe_source_paths_are_rejected(tmp_path, name):
    archive, digest = make_archive(tmp_path, name)
    with pytest.raises(RuntimeError, match='Unsafe'):
        bootstrap.verify_archive(archive, digest)


def test_changed_archive_is_rejected_before_extraction(tmp_path):
    archive, digest = make_archive(tmp_path, 'decision-engines/src/example.py')
    bootstrap.verify_archive(archive, digest)
    archive.write_bytes(archive.read_bytes()+b'changed')
    with pytest.raises(RuntimeError, match='SHA256'):
        bootstrap.verify_archive(archive, digest)


def test_bootstrap_cannot_install_on_local_runtime(monkeypatch):
    monkeypatch.delenv('JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='do not run locally'):
        bootstrap.main()
