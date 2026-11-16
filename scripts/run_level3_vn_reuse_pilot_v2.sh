#!/usr/bin/env bash
set -euo pipefail
# Reuses ONLY downloaded Sidewalk/Travel originals and pinned receipts from v1;
# creates a NEW v2 candidate manifest and review.csv. Does not approve photos.
PYTHON="${PYTHON:-python}"
RAW="${RAW:-data/raw/level3/vietnam-benchmark-v1}"
for source in sidewalk travel; do
  test -f "$RAW/$source/source-receipt.json" || { echo "Missing pinned receipt for $source: $RAW" >&2; exit 1; }
done
"$PYTHON" -m tools.level3_vietnam_benchmark_v2 scan --target pilot --raw-root "$RAW"
printf '\nReview data/processed/level3/vietnam-benchmark-v2/pilot/review.csv; do not finalize until 15+15 are accepted.\n'
