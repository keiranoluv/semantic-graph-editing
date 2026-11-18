#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${PVSG_GPU:-0}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 WANDB_MODE=online
exec data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.m4_150.run "$@"
