"""Acceptance gates; run only as part of the remote suite."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'verify_remote', Path(__file__).resolve().parents[1]/'scripts/verify_remote.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('body', ['', '<testcase><skipped/></testcase>',
                                  '<testcase><failure/></testcase>',
                                  '<testcase><error/></testcase>'])
def test_partial_or_empty_tests_cannot_pass(tmp_path, body):
    report = tmp_path/'pytest.xml'
    report.write_text('<testsuites><testsuite>'+body+'</testsuite></testsuites>')
    with pytest.raises(RuntimeError, match='zero skips'):
        runner.require_complete_tests(report)


def test_completed_tests_are_counted(tmp_path):
    report = tmp_path/'pytest.xml'
    report.write_text('<testsuites><testsuite><testcase/><testcase/></testsuite></testsuites>')
    assert runner.require_complete_tests(report) == 2


def test_identical_manifests_do_not_hide_changed_split(tmp_path):
    first, second = tmp_path/'a', tmp_path/'b'
    for directory in (first, second):
        directory.mkdir()
        for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
            (directory/name).write_text('same')
    runner.require_identical_datasets(first, second)
    (second/'test.jsonl').write_text('different')
    with pytest.raises(RuntimeError, match='test.jsonl differs'):
        runner.require_identical_datasets(first, second)


def test_subprocess_failure_persists_nonpassing_status(tmp_path, monkeypatch):
    import json
    import subprocess
    monkeypatch.setattr(runner, 'require_remote_gpu', lambda: {'provider': 'huggingface_jobs'})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1))
    out = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='pip-check failed'):
        runner.run_acceptance(out)
    state = json.loads((out/'acceptance.json').read_text())
    assert state['passed'] is False
    assert state['status'] == 'failed'
    assert state['active_step'] == 'pip-check'
    assert state['completed_steps'] == []
    assert (out/'pip-check.log').is_file()
