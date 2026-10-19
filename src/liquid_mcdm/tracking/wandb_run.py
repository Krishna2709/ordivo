"""Allowlisted experiment telemetry. Offline by default; never loads .env."""
import json
import math
from pathlib import Path
import tempfile

from liquid_mcdm.data.synthetic import sha256

METRICS = ('accuracy', 'eligible', 'excluded_nonunique', 'brier_multiclass', 'ece_10_bins')
BASELINES = ('uniform_first_option', 'exact_teacher_self_consistency')


def _safe_report(report):
    """Build the artifact ourselves; never forward arbitrary input JSON contents."""
    if set(report) - {'scope','evaluation','accuracy',*BASELINES}:
        raise ValueError('unknown report fields')
    safe = {}
    if 'scope' in report:
        safe['scope'] = 'CPU sanity controls; no learned model evaluated'
    if 'accuracy' in report:
        if not isinstance(report['accuracy'], (int,float)) or not math.isfinite(report['accuracy']):
            raise ValueError('invalid report metric')
        safe['accuracy']=report['accuracy']
    for baseline in BASELINES:
        if baseline not in report:
            continue
        values=report[baseline]
        if set(values) - {*METRICS,'by_method','slices','prediction_tie_break'}:
            raise ValueError('unknown report metric fields')
        safe[baseline]={}
        for metric in METRICS:
            value=values.get(metric)
            if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)):
                raise ValueError('invalid report metric')
            safe[baseline][metric]=value
        # Detailed local slice report stays local; the artifact contains only this numeric summary.
    if 'evaluation' in report:
        evaluation=report['evaluation']
        allowed={'version','split','split_sha256','manifest_sha256','count'}
        if set(evaluation) != allowed or evaluation['split'] not in ('train','val','test'):
            raise ValueError('invalid report evaluation identity')
        for field in ('split_sha256','manifest_sha256'):
            value=evaluation[field]
            if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid report hash')
        if evaluation['version']!='winner-eval-v1' or not isinstance(evaluation['count'],int):
            raise ValueError('invalid report evaluation version/count')
        safe['evaluation']=evaluation
    return safe


def log_cpu_run(manifest_path, report_path, directory, *, mode='offline', entity=None):
    manifest_path, report_path = Path(manifest_path), Path(report_path)
    if manifest_path.name != 'manifest.json' or report_path.suffix != '.json':
        raise ValueError('only manifest.json and a JSON evaluation report are accepted')
    manifest = json.loads(manifest_path.read_text())
    report = _safe_report(json.loads(report_path.read_text()))
    if manifest.get('label_provenance') != 'synthetic_solver':
        raise ValueError('only synthetic manifests are permitted')
    if mode not in ('offline', 'online'):
        raise ValueError('mode must be offline or online')
    # Fixed fields plus numeric values only; raw manifest is never uploaded.
    if manifest.get('generator')!='synthetic-v1' or any(
            not isinstance(manifest.get(k),int) for k in ('seed','count')):
        raise ValueError('invalid manifest generator/seed/count')
    config = {'generator':'synthetic-v1', 'seed':manifest['seed'], 'count':manifest['count'],
              'dataset_manifest_sha256':sha256(manifest_path), 'report_sha256':sha256(report_path),
              'stage':'cpu_smoke','trainer':'none'}
    if 'evaluation' in report:
        identity=report['evaluation']
        if (identity['manifest_sha256']!=sha256(manifest_path)
                or identity['split_sha256']!=manifest['splits'][identity['split']]['sha256']
                or identity['count']!=manifest['splits'][identity['split']]['count']):
            raise ValueError('report does not match manifest')
        config.update(evaluation_split=identity['split'],split_sha256=identity['split_sha256'],
                      evaluation_count=identity['count'],evaluator_version=identity['version'])
    safe_manifest = {key:config[key] for key in ('generator','seed','count','dataset_manifest_sha256')}
    safe_manifest['label_provenance']='synthetic_solver'
    import wandb
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = wandb.Settings(console='off', disable_code=True, disable_git=True,
                              x_disable_stats=True, x_disable_meta=True)
    with tempfile.TemporaryDirectory(prefix='mcdm-artifacts-') as scratch:
        safe_manifest_path=Path(scratch)/'manifest-summary.json'
        safe_report_path=Path(scratch)/'evaluation-summary.json'
        safe_manifest_path.write_text(json.dumps(safe_manifest,sort_keys=True)+'\n')
        safe_report_path.write_text(json.dumps(report,sort_keys=True)+'\n')
        with wandb.init(project='liquid-mcdm-lab', entity=entity, mode=mode,
                        dir=str(directory), config=config, settings=settings,
                        job_type='cpu-smoke') as run:
            run.log({'dataset/episodes':manifest['count'],'cpu/pipeline_smoke':1})
            for baseline in BASELINES:
                for metric,value in report.get(baseline,{}).items():
                    if isinstance(value,(int,float)):
                        run.log({f'{baseline}/{metric}':value})
            artifact = wandb.Artifact('synthetic-cpu-evidence',type='evaluation',
                                     metadata={'manifest_sha256':config['dataset_manifest_sha256']})
            artifact.add_file(str(safe_manifest_path),name='manifest-summary.json')
            artifact.add_file(str(safe_report_path),name='evaluation-summary.json')
            run.log_artifact(artifact)
            result={'run_id':run.id,'mode':run.settings.mode,'directory':run.dir,
                    'cloud_url':run.url if mode=='online' else None}
    return result
