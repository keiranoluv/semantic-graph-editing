#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
session_name="aligng-ablation-v1"

if [[ "${1:-}" == "--worker" ]]; then
    mkdir -p logs
    exec myenv/bin/python -u -m experiments.aligng_ablation_v1.runner --stage train-all >> logs/aligng-ablation-v1.log 2>&1
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/run_aligng_ablation_v1.sh" >&2
    exit 2
fi
if tmux has-session -t "=$session_name" 2>/dev/null; then
    echo "Session $session_name already exists; no duplicate queue started."
    echo "View: tmux attach -t $session_name"
    exit 0
fi
# CPU-only preflight fails visibly before detaching; does not start a W&B run.
myenv/bin/python -m experiments.aligng_ablation_v1.runner --stage check
tmux new-session -d -s "$session_name" bash "$repo_root/scripts/run_aligng_ablation_v1.sh" --worker
echo "Started six sequential jobs in tmux: $session_name"
echo "Train/val/effective batch512; seeds42,123,3407; Edge-only then PE-CE+AlignG for each seed."
echo "Log: $repo_root/logs/aligng-ablation-v1.log"
echo "Status: $repo_root/runs/level2/aligng-ablation-v1/full/queue-status.json"
