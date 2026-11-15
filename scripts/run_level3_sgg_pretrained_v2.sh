#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?Usage: bash scripts/run_level3_sgg_pretrained_v2.sh react|relate}"
case "$MODEL" in react|relate) ;; *) echo 'Model must be react or relate' >&2; exit 2;; esac
OUTPUT="${OUTPUT:-runs/level3/sgg-pretrained-v2/${MODEL}-five-v1}"
PYTHON="${PYTHON:-data/cache/level3-sgg-pretrained-v2/venv/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
# Exactly one requested pipeline. Relate runs OWLv2, releases it, then relation head.
# CLI shares the ROBIN inference lock. No next-model queue, no training.
exec "$PYTHON" -m tools.level3_sgg_pretrained_v2 infer \
  --model "$MODEL" --device cuda --output "$OUTPUT"
