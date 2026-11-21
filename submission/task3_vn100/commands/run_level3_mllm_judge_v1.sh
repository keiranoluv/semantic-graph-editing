#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
command="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  smoke|score) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ;;
  *) export CUDA_VISIBLE_DEVICES='' ;;
esac
if [[ "$command" != prepare ]]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "${PYTHON:-data/cache/level3-evaluation-v1/venv/bin/python}" -m task3_vn100.mllm_judge "$command" "$@"
