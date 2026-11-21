#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-evaluation-v1/venv/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
COMMAND="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
# No image/SGG generation, training, or automatic next experiment.
exec "$PYTHON" -m tools.level3_evaluation "$COMMAND" --output "${OUTPUT:-runs/level3/evaluation-v1/seed42}" "$@"
