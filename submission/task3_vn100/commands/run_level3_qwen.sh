#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-data/cache/level3-qwen21-v1/venv/bin/python}"
RUN_ID="${RUN_ID:-seed42}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens." >&2
  exit 2
fi
OUTPUT="${OUTPUT:-runs/level3/qwen200/$RUN_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=online
# One 200-task cohort. Resume this run, infer missing tasks, then stop.
COMMAND="${1:-infer}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON" -m task3_vn100.qwen_unified "$COMMAND" --output "$OUTPUT" --no-import "$@"
