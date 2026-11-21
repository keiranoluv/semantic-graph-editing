#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
stage=${1:-train}
if [[ "$stage" == smoke-cpu ]]; then export CUDA_VISIBLE_DEVICES=''; fi
python_bin="$PWD/data/cache/gqa-gat-loss-torch270/venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Thiếu venv GQA. Xem docs/LEVEL2_GQA_GAT_LOSS_ABLATION_V1.md để cài đặt." >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH
mkdir -p logs
logfile="logs/gqa-gat-loss-v1-$stage-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Log: $logfile"
exec "$python_bin" -u -m tools.gqa_loss_ablation_v1 "$stage" > >(tee -i "$logfile") 2>&1
