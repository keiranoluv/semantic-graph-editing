#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MAGICBRUSH_RUNTIME="${MAGICBRUSH_PYTHON:-$PROJECT_ROOT/myenv/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs/magicbrush-manual-experiment-v1
"$MAGICBRUSH_RUNTIME" -u -m tools.magicbrush_experiment_v1.pipeline \
  --config configs/level2/magicbrush-manual-experiment-v1.json --stage run "$@" \
  2>&1 | tee -a logs/magicbrush-manual-experiment-v1/launcher.log
