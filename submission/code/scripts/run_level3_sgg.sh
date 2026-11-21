#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-sgg-pretrained-v2/venv/bin/python}"
RUN_ID="${RUN_ID:-seed42}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens." >&2
  exit 2
fi
OUTPUT="${OUTPUT:-runs/level3/qwen-sgg/$RUN_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online OMP_NUM_THREADS=4
COMMAND="${1:-infer}"
if [[ $# -gt 0 ]]; then shift; fi
# REACT++, batch 1, seed 42. Read Qwen output with the same run name; resume up to 300 images.
exec "$PYTHON" -m tools.level3_qwen_sgg "$COMMAND" --output "$OUTPUT" --no-import --max-new-images 300 "$@"
