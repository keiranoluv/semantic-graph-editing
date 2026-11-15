#!/usr/bin/env bash
set -euo pipefail
# Local-only source selection; preserves existing upstream images and immutable cohort.
# Does NOT attest per-image license, privacy clearance, or scene graph correctness.
PYTHON="${PYTHON:-python}"
"$PYTHON" -m tools.level3_vietnam_benchmark_v2 finalize --target pilot --provenance-only
printf '\nLocal pilot selection and scene-graph tasks: data/processed/level3/vietnam-benchmark-v2/pilot/locked/\n'
