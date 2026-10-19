import json
import subprocess
import sys


def test_generate_and_baseline_cli(tmp_path):
    data = tmp_path/'data'
    run = subprocess.run([sys.executable,'-m','liquid_mcdm.cli','generate','--output',str(data),
                          '--count','40','--seed','7'],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    report = tmp_path/'report.json'
    run = subprocess.run([sys.executable,'-m','liquid_mcdm.cli','baseline','--data',str(data),
                          '--output',str(report)],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(report.read_text())['exact_teacher_self_consistency']['accuracy'] == 1


def test_environment_has_explicit_training_readiness():
    run = subprocess.run([sys.executable,'-m','liquid_mcdm.cli','environment'],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    report=json.loads(run.stdout)
    assert report['training_validated'] is False
    assert 'python' in report


def test_baseline_records_split_identity(tmp_path):
    from liquid_mcdm.data import generate_dataset
    data=tmp_path/'data'
    manifest=generate_dataset(data,count=40,seed=7)
    report=tmp_path/'report.json'
    subprocess.run([sys.executable,'-m','liquid_mcdm.cli','baseline','--data',str(data),
                    '--split','val','--output',str(report)],check=True,capture_output=True)
    result=json.loads(report.read_text())
    assert result['evaluation']['split']=='val'
    assert result['evaluation']['split_sha256']==manifest['splits']['val']['sha256']
    assert result['evaluation']['count']==4
