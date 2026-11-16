#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-qwen21-v1/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/qwen200/seed42}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=online
# One 200-task cohort. Default: import successes, infer pending tasks, stop. Same output resumes.
COMMAND="${1:-infer}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON" -m tools.level3_qwen_unified "$COMMAND" --output "$OUTPUT" "$@"
