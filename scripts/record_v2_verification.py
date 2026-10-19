"""Record an actual RunPod verification suite; no training metrics are invented."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from liquid_mcdm.runtime import require_remote_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    runtime = require_remote_gpu()
    cases = ET.parse(args.junit).getroot().findall('.//testcase')
    if not cases or any(case.find(tag) is not None for case in cases
                        for tag in ('failure', 'error', 'skipped')):
        raise RuntimeError('Only complete passing test evidence may be recorded as passed')
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('Record a clean committed source snapshot')
    report = dict(schema_version='ordivo.verification-run.v1', code_commit=commit,
                  created_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime,
                  tests_passed=len(cases), failures=0, errors=0, skipped=0,
                  test_seconds=sum(float(case.get('time', 0)) for case in cases),
                  junit_sha256=hashlib.sha256(args.junit.read_bytes()).hexdigest(),
                  training_started=False, training_validated=False,
                  v2_model_created=False, mode='online' if args.online else 'offline')
    import wandb
    args.output.mkdir(parents=True, exist_ok=False)
    run = wandb.init(project='liquid-mcdm-lab', entity='black-order',
                     job_type='numerical-verification', name='ordivo-v2-verifier',
                     mode=report['mode'], dir=str(args.output), save_code=False,
                     settings=wandb.Settings(disable_git=True), config=report)
    report.update(wandb_run_id=run.id, wandb_url=run.url if args.online else None)
    path = args.output / 'verification.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    run.log({'verification/tests_passed': len(cases), 'verification/failures': 0,
             'verification/skipped': 0})
    artifact = wandb.Artifact('ordivo-v2-verifier-evidence', type='verification', metadata=report)
    artifact.add_file(str(path), name='verification.json')
    artifact.add_file(str(args.junit), name='pytest.xml')
    logged = run.log_artifact(artifact)
    if args.online:
        logged.wait()
        report['wandb_artifact'] = logged.qualified_name
    run.finish()
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
