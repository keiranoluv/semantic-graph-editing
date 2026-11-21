#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${PVSG_GPU:-0}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
exec data/external/pvsg_m3_env/bin/python -m extension_pvsg.m5.run "$@"
