#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-qwen21-v1/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/qwen100-v1/seed42-v1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=online
# Exactly 100 frozen tasks, batch 1, BF16, seed 42, 40 steps; no next-seed queue.
if [[ -n "${RESUME_FROM:-}" ]]; then
  exec "$PYTHON" -m tools.level3_qwen100_v1 infer --output "$OUTPUT" --resume-from "$RESUME_FROM"
fi
exec "$PYTHON" -m tools.level3_qwen100_v1 infer --output "$OUTPUT"
