#!/usr/bin/env bash
set -euo pipefail
# From project root, after reading docs/LEVEL3_VIETNAM_BENCHMARK_V1.md.
PYTHON="${PYTHON:-python}"
# Explicit network downloads. No model/GPU.
"$PYTHON" -m tools.level3_vietnam_benchmark_v1 fetch --source sidewalk --target pilot
"$PYTHON" -m tools.level3_vietnam_benchmark_v1 fetch --source streetfood --target pilot
"$PYTHON" -m tools.level3_vietnam_benchmark_v1 fetch --source travel --target pilot
"$PYTHON" -m tools.level3_vietnam_benchmark_v1 scan --target pilot
printf '\nReview the CSV and contact sheets before running finalize; nothing is auto-approved.\n'
