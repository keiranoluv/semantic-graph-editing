#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=online
# --model and --seed are mandatory: ONE run; never advances a queue.
exec data/external/pvsg_m3_env/bin/python -m extension_pvsg.m4.runner "$@"
