#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-qwen21-v1/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/qwen200-extension-v1/seed42-v1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=online
# Qwen 2.1: ONLY 100 new independent edits, BF16, batch 1, seed 42, 40 steps.
# Existing environment and model CPU offload; no next-seed queue or graph inference.
if [[ -n "${RESUME_FROM:-}" ]]; then
  exec "$PYTHON" -m tools.level3_qwen200_v1 infer --output "$OUTPUT" --resume-from "$RESUME_FROM"
fi
exec "$PYTHON" -m tools.level3_qwen200_v1 infer --output "$OUTPUT"
