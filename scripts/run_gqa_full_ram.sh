#!/usr/bin/env bash
# Resume the frozen GQA recipe using identical graphs preloaded into CPU RAM.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p logs
log_path="logs/gqa-full-ram-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
if (( $# == 0 )); then
  set -- --stage train
fi
echo "Log: $repo_root/$log_path"
echo "PID: $$ (send SIGTERM to this PID to save and stop)"
exec "${GQA_PYTHON:-myenv/bin/python}" -u -m tools.run_gqa_full_ram "$@" > >(tee -i "$log_path") 2>&1
