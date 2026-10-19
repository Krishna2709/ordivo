"""Runtime inspection with no package installation, authentication or model load."""
import importlib.metadata
import platform
import subprocess

from liquid_mcdm.runtime import find_nvidia_smi


def environment_report():
    packages = {}
    for name in ('numpy','torch','torchvision','transformers','trl','peft','datasets',
                 'accelerate','torchao','wandb'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'packages': packages, 'gpu': None, 'training_validated': False,
              'note': 'Hardware/package inventory only; no model forward or backward executed.'}
    executable = find_nvidia_smi()
    if executable:
        p = subprocess.run([executable,'--query-gpu=name,memory.total,driver_version',
                            '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        report['gpu'] = p.stdout.strip() if p.returncode == 0 else 'nvidia-smi failed'
    if packages['torch']:
        try:
            import torch
            report['torch_runtime'] = {'cuda_build': torch.version.cuda,
                                       'cuda_available': torch.cuda.is_available()}
            if torch.cuda.is_available():
                report['torch_runtime']['bf16_native'] = torch.cuda.is_bf16_supported(including_emulation=False)
                free, total = torch.cuda.mem_get_info()
                report['torch_runtime'].update(free_vram_bytes=free,total_vram_bytes=total)
        except (ImportError, RuntimeError, OSError) as exc:
            report['torch_import_error'] = type(exc).__name__
    return report
