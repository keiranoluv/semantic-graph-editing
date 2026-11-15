#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-data/cache/level3-sgg-v1/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/sgg-compare-v1/robin-five-v1}"
mkdir -p data/cache/level3-sgg-v1
exec 9>data/cache/level3-sgg-v1/inference.lock
flock -n 9 || { printf 'Another SGG pilot launcher is running.\n' >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# SAM ViT-H on five images, unload, ROBIN-3B on same five. No training/next queue.
# Image batch 1; SAM point batch 32; seed 42; greedy decode <=2048 new tokens.
exec "$PYTHON" -m tools.level3_sgg_compare_v1 infer --device cuda --output "$OUTPUT" "$@"
