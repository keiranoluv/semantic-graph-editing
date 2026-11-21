#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${PVSG_GPU:-0}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=online
exec data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.min50_followup.run "$@"
