#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-42}"
case "$SEED" in 42|123|3407) ;; *) echo 'Seed must be 42, 123 or 3407' >&2; exit 2;; esac
OUTPUT="${OUTPUT:-runs/level3/task3-v1/pilot-five-seed${SEED}-v1}"
TASKS="${TASKS:-runs/level3/task3-v1/tasks-approved-v1}"
PYTHON="${PYTHON:-data/cache/level3-task3-v1/venv/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
# Single seed only, batch 1. Within the run: editor -> REACT++ -> CLIP -> LPIPS.
# Each model is released before the next stage. No next-seed queue or training.
exec "$PYTHON" -m tools.level3_task3_v1 infer --tasks "$TASKS" --seed "$SEED" --device cuda --output "$OUTPUT"
