"""Remote-executed safety checks for the v2 one-pod controller."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def control(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('runpod_v2_control_under_test',
                                                  scripts / 'runpod_v2_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'RUNS', tmp_path)
    return module


def test_preflight_refuses_existing_pod_before_catalog_lookup(control, monkeypatch):
    monkeypatch.setattr(control, 'pods', lambda: [{'id': 'existing'}])
    monkeypatch.setattr(control, 'request', lambda *a, **kw: pytest.fail('catalog queried'))
    with pytest.raises(RuntimeError, match='already has a pod'):
        control.preflight()


def test_preflight_enforces_secure_hourly_cap(control, monkeypatch):
    monkeypatch.setattr(control, 'pods', lambda: [])
    monkeypatch.setattr(control, 'load_secret', lambda name: 'hidden')
    monkeypatch.setattr(control, 'request', lambda *a, **kw: [
        {'id': 'NVIDIA A40', 'availability': 'HIGH', 'price': {'secure': 0.60}}])
    with pytest.raises(RuntimeError, match='exceeds cap'):
        control.preflight()


def test_preflight_reads_current_catalog_gpus_shape(control, monkeypatch):
    monkeypatch.setattr(control, 'pods', lambda: [])
    monkeypatch.setattr(control, 'load_secret', lambda name: 'hidden')
    monkeypatch.setattr(control, 'request', lambda *a, **kw: {'gpus': [
        {'id': 'NVIDIA A40', 'availability': 'HIGH', 'price': {'secure': 0.59}}]})
    assert control.preflight() == .59


def test_launch_uses_no_volume_and_arms_reaper(control, monkeypatch):
    monkeypatch.setattr(control, 'preflight', lambda: .59)
    monkeypatch.setattr(control, 'active_run', lambda: None)
    bodies = []

    def api(method, path, *, body=None):
        if method == 'POST':
            bodies.append(body)
            return {'id': 'new-pod', 'costPerHr': .59}
        raise AssertionError((method, path))

    monkeypatch.setattr(control, 'api', api)
    monkeypatch.setattr(control.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(
        pid=123, poll=lambda: None))
    result = control.launch()
    assert result['hourly_usd'] == .59
    assert len(bodies) == 1
    assert 'networkVolumeId' not in bodies[0]
    assert bodies[0]['containerDiskInGb'] == 50
    assert bodies[0]['gpuTypeIds'] == ['NVIDIA A40']
    assert bodies[0]['gpuCount'] == 1
    receipt = next(control.RUNS.glob('v2-swarm-*/pod.json'))
    assert 'reaper_pid' in receipt.read_text()


def test_task_rejects_path_escape_before_remote_or_archive_access(control, monkeypatch):
    monkeypatch.setattr(control, 'live_pod', lambda: pytest.fail('remote accessed'))
    with pytest.raises(ValueError, match='relative'):
        control.run_task('x', Path('/unused'), 'a' * 40, 'true', outputs=['../private'])
