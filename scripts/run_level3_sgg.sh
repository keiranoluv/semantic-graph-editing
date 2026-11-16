#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-sgg-pretrained-v2/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/qwen-sgg/seed42}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online OMP_NUM_THREADS=4
COMMAND="${1:-infer}"
if [[ $# -gt 0 ]]; then shift; fi
# REACT++, batch 1, seed 42. Import valid predictions, infer <=100 missing images, then stop.
exec "$PYTHON" -m tools.level3_qwen_sgg "$COMMAND" --output "$OUTPUT" "$@"
