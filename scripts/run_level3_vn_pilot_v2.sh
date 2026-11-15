#!/usr/bin/env bash
set -euo pipefail
# Two-source 15+15 pilot. Fetches 45 candidates per source for manual review.
PYTHON="${PYTHON:-python}"
"$PYTHON" -m tools.level3_vietnam_benchmark_v2 fetch --source sidewalk --target pilot
"$PYTHON" -m tools.level3_vietnam_benchmark_v2 fetch --source travel --target pilot
"$PYTHON" -m tools.level3_vietnam_benchmark_v2 scan --target pilot
printf '\nReview both image sets, fill review.csv, then explicitly run finalize. Nothing auto-approved.\n'
