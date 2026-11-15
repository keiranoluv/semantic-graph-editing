#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-sgg-pretrained-v2/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/qwen-react100-v1/run-$(date -u +%Y%m%dT%H%M%SZ)}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online OMP_NUM_THREADS=4
# Exactly 100 Qwen pairs / 200 images, REACT++ only, batch 1, seed 42.
# Fresh output per invocation; no editor, training, next seed or next-model queue.
exec "$PYTHON" -m tools.level3_qwen_graphs_v1 infer --output "$OUTPUT"
