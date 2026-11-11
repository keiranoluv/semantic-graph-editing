#!/usr/bin/env bash
# Two isolated runs; coordinator SIGTERM safely stops and saves both workers.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p logs
log_path="logs/gqa-full-parallel-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Log: $repo_root/$log_path"
echo "Coordinator PID: $$ (send SIGTERM to save and stop both trainers)"
exec "${GQA_PYTHON:-myenv/bin/python}" -u -m tools.run_gqa_full_parallel "$@" > >(tee -i "$log_path") 2>&1
