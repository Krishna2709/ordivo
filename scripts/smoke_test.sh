#!/usr/bin/env bash
set -euo pipefail
python -c "from liquid_mcdm.runtime import require_colab_gpu; require_colab_gpu()"
python -m pytest
SMOKE_ROOT=$(mktemp -d)
python -m liquid_mcdm.cli generate --output "$SMOKE_ROOT/data" --count 1000 --seed 42
python -m liquid_mcdm.cli baseline --data "$SMOKE_ROOT/data" --output "$SMOKE_ROOT/report.json" --wandb offline
python -m liquid_mcdm.cli environment
