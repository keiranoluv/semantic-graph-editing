#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Same five images, SAM ViT-H -> ROBIN-3B, image batch 1, SAM points batch 32,
# seed 42. New CUDA 12.8 runtime; no follow-up queue or training.
export PYTHON="${PYTHON:-data/cache/level3-sgg-v1/venv-cu128-v2/bin/python}"
export OUTPUT="${OUTPUT:-runs/level3/sgg-compare-v1/robin-five-cu128-v2}"
exec bash scripts/run_level3_sgg_robin_v1.sh "$@"
