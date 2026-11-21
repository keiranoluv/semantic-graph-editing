#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=online
exec "${MAGICBRUSH_PYTHON:-myenv/bin/python}" -u -m experiments.magicbrush_prompt_intent_v1.run "$@"
