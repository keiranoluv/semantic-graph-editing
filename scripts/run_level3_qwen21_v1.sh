#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
PYTHON="${PYTHON:-data/cache/level3-qwen21-v1/venv/bin/python}"
OUTPUT="${OUTPUT:-runs/level3/qwen21-v1/${MODE}-v1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "$MODE" in
  smoke) exec "$PYTHON" -m tools.level3_qwen21_v1 smoke --output "$OUTPUT" ;;
  pilot) exec "$PYTHON" -m tools.level3_qwen21_v1 pilot --output "$OUTPUT" --smoke-run "${SMOKE_RUN:-runs/level3/qwen21-v1/smoke-v1}" ;;
  *) echo 'Usage: launcher smoke|pilot (one explicit batch, no next-run queue)' >&2; exit 2 ;;
esac
