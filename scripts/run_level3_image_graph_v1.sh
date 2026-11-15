#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-myenv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/image-graph-v1/pilot30}"
mkdir -p data/cache/level3-image-graph-v1
exec 9>data/cache/level3-image-graph-v1/inference.lock
flock -n 9 || { printf 'Another Level 3 image graph launcher is running.\n' >&2; exit 1; }
# One job; all 30 RelTR images, unload, then all 30 CLIP region attributes.
# Batch 1, seed 42. No training, no follow-up experiment queue, no auto-resume.
exec "$PYTHON" -m tools.level3_image_graph_v1 infer --device cuda --output "$OUTPUT" "$@"
