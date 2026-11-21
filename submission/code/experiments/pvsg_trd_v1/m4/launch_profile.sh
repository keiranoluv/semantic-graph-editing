#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
# Exactly one model. No automatic training/queue after profiling.
exec data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.m4.probe "$@"
