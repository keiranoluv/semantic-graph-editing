#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
session_name="aligng-gqa-fast"

if [[ "${1:-}" == "--worker" ]]; then
    mkdir -p logs
    exec myenv/bin/python -u -m task2_gqa.training.runner --stage train-all >> logs/aligng-gqa-fast.log 2>&1
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash task2_gqa/commands/run_aligng_gqa_fast.sh" >&2
    exit 2
fi
if tmux has-session -t "=$session_name" 2>/dev/null; then
    echo "Session $session_name already exists; no duplicate queue started."
    echo "View: tmux attach -t $session_name"
    exit 0
fi
# CPU-only preflight fails visibly before detaching; does not start a W&B run.
myenv/bin/python -m task2_gqa.training.runner --stage check
tmux new-session -d -s "$session_name" bash "$repo_root/task2_gqa/commands/run_aligng_gqa_fast.sh" --worker
echo "Started six sequential jobs in tmux: $session_name"
echo "Train/val/effective batch512; seeds42,123,3407; AlignG then PE-PLPR for each seed."
echo "Log: $repo_root/logs/aligng-gqa-fast.log"
echo "Status: $repo_root/runs/level2/aligng-gqa-fast-v1/full/queue-status.json"
