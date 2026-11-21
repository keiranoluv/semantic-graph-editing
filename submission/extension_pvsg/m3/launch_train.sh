#!/usr/bin/env bash
set -euo pipefail
PVSG_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${PVSG_GPU:-0}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=online
PVSG_PYTHON="data/external/pvsg_m3_env/bin/python"
PVSG_PROFILE="${1:-reports/pvsg-trd-m3-v1/gpu_profile.json}"
"$PVSG_PYTHON" -m extension_pvsg.m3.preflight --profile "$PVSG_PROFILE"
# This explicit queue contains nine runs. Never run two models simultaneously.
mkdir -p reports/pvsg-trd-m3-v1
exec 9>reports/pvsg-trd-m3-v1/.queue.lock
flock -n 9
for PVSG_MODEL in mlp static_gat st_gat; do
  for PVSG_SEED in 17 42 2026; do
    "$PVSG_PYTHON" -m extension_pvsg.m3.runner --model "$PVSG_MODEL" --seed "$PVSG_SEED" --profile "$PVSG_PROFILE"
  done
done
"$PVSG_PYTHON" -m extension_pvsg.m3.summarize
