#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-sgg-pretrained-v2/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/qwen-graph-score-v1/run-$(date -u +%Y%m%dT%H%M%SZ)}"
export CUDA_VISIBLE_DEVICES='' WANDB_MODE=online OMP_NUM_THREADS=4
# CPU only; 100 frozen pairs, no image model, no label loading or next-run queue.
exec "$PYTHON" -m tools.level3_qwen_scoring_v1 score --output "$OUTPUT"
