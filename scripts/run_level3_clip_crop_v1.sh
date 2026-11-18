#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
logdir=runs/level3/clip-crop-v1/seed42/logs
mkdir -p "$logdir"
logfile=$(mktemp "$logdir/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
echo "Log: $logfile"
data/cache/level3-evaluation-v1/venv/bin/python -u -m tools.level3_clip_crop_v1 --device cuda "$@" 2>&1 | tee "$logfile"
