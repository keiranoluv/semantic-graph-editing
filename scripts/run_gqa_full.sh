#!/usr/bin/env bash
# Run from any directory. Each invocation appends a separate timestamped log.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p logs
log_path="logs/gqa-full-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
if (( $# == 0 )); then
  set -- --stage all
fi
echo "Log: $repo_root/$log_path"
echo "PID: $$ (send SIGTERM to this PID to save and stop)"
# exec keeps the launcher PID equal to the Python PID, including under nohup.
# tee ignores Ctrl+C so the Python handler can finish saving its checkpoint.
exec "${GQA_PYTHON:-myenv/bin/python}" -u -m tools.run_gqa_full "$@" > >(tee -i "$log_path") 2>&1
