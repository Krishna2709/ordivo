"""Mocked Runpod control-plane regressions; the suite itself runs remotely."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def control(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('runpod_control_under_test',
                                                  scripts / 'runpod_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'STAGE', tmp_path / 'stage')
    monkeypatch.setattr(module, 'CONFIG', tmp_path / 'config.json')
    return module


def test_api_key_is_not_written_to_prepared_receipt(control, monkeypatch):
    control.STAGE.mkdir()
    (control.ROOT / 'scripts').mkdir()
    (control.ROOT / 'scripts' / 'runpod_bootstrap.py').write_text('bootstrap')
    config = {'name_prefix': 'test'}
    control.CONFIG.write_text(json.dumps(config))

    def package(path):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('decision-engines/scripts/runpod_bootstrap.py', 'bootstrap')
        return path

    monkeypatch.setenv('RUNPOD_API_KEY', 'do-not-serialize')
    monkeypatch.setattr(control, 'package', package)
    monkeypatch.setattr(control, 'source_identity', lambda: {
        'scripts/runpod_bootstrap.py': control.digest(control.ROOT / 'scripts' / 'runpod_bootstrap.py')})
    control.prepare()
    assert 'do-not-serialize' not in (control.STAGE / 'prepared.json').read_text()


def test_launch_receipt_precedes_resource_mutation_and_records_ids(control, monkeypatch):
    control.STAGE.mkdir()
    (control.STAGE / 'source.zip').write_bytes(b'source')
    (control.STAGE / 'runpod_bootstrap.py').write_bytes(b'bootstrap')
    config = {'name_prefix': 'test', 'network_volume_gb': 20,
              'data_center_ids': ['CA-MTL-1'], 'cloud_type': 'SECURE',
              'gpu_type_id': 'NVIDIA A40', 'gpu_count': 1, 'image_name': 'image',
              'container_disk_gb': 50, 'volume_mount_path': '/workspace',
              'ports': ['22/tcp'], 'wandb_mode': 'offline', 'max_hourly_usd': .59,
              'max_runtime_seconds': 7200}
    control.CONFIG.write_text(json.dumps(config))
    source_files = {'x': 'y'}
    prepared = {'config': config, 'source_sha256': control.digest(control.STAGE / 'source.zip'),
                'bootstrap_sha256': control.digest(control.STAGE / 'runpod_bootstrap.py'),
                'source_files': source_files}
    (control.STAGE / 'prepared.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(control, 'source_identity', lambda: source_files)
    monkeypatch.setattr(control, 'preflight', lambda: {})
    monkeypatch.setattr(control, 'load_api_key', lambda: 'secret')
    monkeypatch.setattr(control, 'start_reaper', lambda config, state: 12345)
    calls = []

    def request(method, path, **kwargs):
        assert (control.STAGE / 'launch.json').exists()
        calls.append((method, path, kwargs.get('body')))
        if path == '/networkvolumes':
            return {'id': 'volume-1'}
        return {'id': 'pod-1', 'costPerHr': .59}

    monkeypatch.setattr(control, 'request', request)
    control.launch()
    saved = json.loads((control.STAGE / 'launch.json').read_text())
    assert saved['network_volume_id'] == 'volume-1'
    assert saved['pod_id'] == 'pod-1'
    assert calls[1][2]['interruptible'] is False
    assert calls[1][2]['env'] == {'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_MODE': 'offline'}


def test_price_guard_deletes_pod_but_preserves_volume(control, monkeypatch):
    control.STAGE.mkdir()
    (control.STAGE / 'source.zip').write_bytes(b'source')
    (control.STAGE / 'runpod_bootstrap.py').write_bytes(b'bootstrap')
    config = {'name_prefix': 'test', 'network_volume_gb': 20,
              'data_center_ids': ['CA-MTL-1'], 'cloud_type': 'SECURE',
              'gpu_type_id': 'NVIDIA A40', 'gpu_count': 1, 'image_name': 'image',
              'container_disk_gb': 50, 'volume_mount_path': '/workspace',
              'ports': ['22/tcp'], 'wandb_mode': 'offline', 'max_hourly_usd': .59,
              'max_runtime_seconds': 7200}
    control.CONFIG.write_text(json.dumps(config))
    source_files = {'x': 'y'}
    (control.STAGE / 'prepared.json').write_text(json.dumps({
        'config': config, 'source_sha256': control.digest(control.STAGE / 'source.zip'),
        'bootstrap_sha256': control.digest(control.STAGE / 'runpod_bootstrap.py'),
        'source_files': source_files}))
    monkeypatch.setattr(control, 'source_identity', lambda: source_files)
    monkeypatch.setattr(control, 'preflight', lambda: {})
    monkeypatch.setattr(control, 'load_api_key', lambda: 'secret')
    monkeypatch.setattr(control, 'start_reaper', lambda config, state: 12345)
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path))
        if path == '/networkvolumes':
            return {'id': 'volume-1'}
        if path == '/pods':
            return {'id': 'pod-1', 'costPerHr': .60}
        return None

    monkeypatch.setattr(control, 'request', request)
    with pytest.raises(RuntimeError, match='hourly ceiling'):
        control.launch()
    saved = json.loads((control.STAGE / 'launch.json').read_text())
    assert saved['status'] == 'pod_deleted_price_guard'
    assert saved['network_volume_preserved'] is True
    assert ('DELETE', '/pods/pod-1') in calls
    assert ('DELETE', '/networkvolumes/volume-1') not in calls
