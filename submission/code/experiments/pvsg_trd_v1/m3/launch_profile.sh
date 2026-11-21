#!/usr/bin/env bash
set -euo pipefail
PVSG_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PVSG_PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${PVSG_GPU:-0}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec data/external/pvsg_m3_env/bin/python -m experiments.pvsg_trd_v1.m3.probe "$@"
