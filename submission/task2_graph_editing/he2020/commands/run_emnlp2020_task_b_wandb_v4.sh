#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
EMNLP_PYTHON="${EMNLP_PYTHON:-$PROJECT_ROOT/myenv/bin/python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=online
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs/emnlp2020-task-b-wandb-v4
exec 9>logs/emnlp2020-task-b-wandb-v4/queue.lock
flock -n 9 || { echo "EMNLP v4 queue is already running." >&2; exit 1; }
"$EMNLP_PYTHON" -u -m task2_graph_editing.he2020.training --stage train "$@" \
  2>&1 | tee -a logs/emnlp2020-task-b-wandb-v4/launcher.log
