"""Backward-compatible Colab entry point for the shared acceptance pipeline."""
from datetime import datetime, timezone
from pathlib import Path

from liquid_mcdm.runtime import require_colab_gpu
from verify_remote import run_acceptance


def main():
    require_colab_gpu()
    root = Path(__file__).resolve().parents[1]
    out = root/'outputs'/('colab-check-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run_acceptance(out)


if __name__ == '__main__':
    main()
