#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-data/cache/level3-evaluation-v1/venv/bin/python}"
COMMAND="${1:-serve}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON" -m tools.level3_label_recheck "$COMMAND" --output "${OUTPUT:-runs/level3/evaluation-v1/seed42/label-recheck-v1}" "$@"
