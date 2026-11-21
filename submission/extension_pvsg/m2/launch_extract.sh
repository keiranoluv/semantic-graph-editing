#!/usr/bin/env bash
set -euo pipefail

PVSG_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PVSG_PYTHON="data/external/pvsg_m2_cpu_env/bin/python"
"$PVSG_PYTHON" -m extension_pvsg.m2.preflight
export CUDA_VISIBLE_DEVICES="${PVSG_GPU:-0}"
exec "$PVSG_PYTHON" -m extension_pvsg.m2.features \
  --device cuda:0 \
  --batch-size 64 \
  --cache-root data/external/pvsg_m2_cache_v4 \
  --output data/external/pvsg_clip_cache_v1
