#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ $# -eq 0 ]]; then set -- build; fi
exec "${PYTHON:-data/cache/level3-evaluation-v1/venv/bin/python}" -m tools.level3_submission_v1 "$@"
