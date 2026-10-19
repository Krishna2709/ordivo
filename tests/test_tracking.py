from pathlib import Path
import pytest
from liquid_mcdm.data import generate_dataset
from liquid_mcdm.tracking import log_cpu_run


def test_real_offline_run_and_allowlist(tmp_path):
    generate_dataset(tmp_path/'data',count=20,seed=1)
    report=tmp_path/'report.json'
    report.write_text('{"accuracy": 1.0}')
    result=log_cpu_run(tmp_path/'data'/'manifest.json', report, tmp_path/'tracking')
    assert result['mode']=='offline'
    assert result['cloud_url'] is None
    assert list(Path(result['directory']).parent.glob('run-*.wandb'))
    assert Path(result['directory']).is_dir()


def test_secret_filename_cannot_be_uploaded(tmp_path):
    secret=tmp_path/'.env'
    secret.write_text('FAKE_SECRET=neverupload')
    with pytest.raises(ValueError,match='manifest'):
        log_cpu_run(secret,secret,tmp_path/'tracking')


def test_unknown_private_report_field_rejected(tmp_path):
    generate_dataset(tmp_path/'data',count=20,seed=1)
    report=tmp_path/'report.json'
    report.write_text('{"private_note":"FAKE_PRIVATE_SENTINEL"}')
    with pytest.raises(ValueError,match='report'):
        log_cpu_run(tmp_path/'data'/'manifest.json',report,tmp_path/'tracking')
