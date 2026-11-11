#!/usr/bin/env bash
# Exact graph/supervision RAM cache; resumes the same full GQA checkpoints.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p logs
log_path="logs/gqa-full-prepared-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
if (( $# == 0 )); then
  set -- --stage train
fi
echo "Log: $repo_root/$log_path"
echo "PID: $$ (send SIGTERM to this PID to save and stop)"
exec "${GQA_PYTHON:-myenv/bin/python}" -u -m tools.run_gqa_full_prepared "$@" > >(tee -i "$log_path") 2>&1
