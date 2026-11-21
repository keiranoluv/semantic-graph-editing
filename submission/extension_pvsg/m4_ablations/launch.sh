#!/usr/bin/env bash
# Four M4 comparisons with the locked M3/M4 batch-32 protocol.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
usage() {
  echo "Usage: bash $0 {plan|profile|train} [17|42|2026|all]"
}
(( $# <= 2 )) || { usage >&2; exit 2; }
action=${1:-plan}
case "$action" in plan|train|profile) ;; *) usage >&2; exit 2 ;; esac
case "${2:-all}" in
  all) seeds=(17 42 2026) ;;
  17|42|2026) seeds=("$2") ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $action == profile && $# == 2 ]]; then
  echo "Profiling uses the fixed probe seed 17; omit the seed argument." >&2
  exit 2
fi
models=(st_no_temporal st_no_spatial st_no_edges mlp_conv_k3_l2)
launcher=extension_pvsg/m4_150_bs32/launch.sh
if [[ $action != plan ]]; then
  mkdir -p reports/pvsg-trd-m4-150-bs32-v1 reports/pvsg-trd-m4-150-v1
  exec 9>reports/pvsg-trd-m4-150-bs32-v1/.ablation-queue.lock
  flock -n 9 || { echo "An ablation queue is already running." >&2; exit 1; }
fi
run() {
  if [[ $action == plan ]]; then
    printf 'bash %s' "$launcher"
    printf ' %s' "$@"
    printf '\n'
  else
    # Wait for the current standalone M4 run. The downstream launcher itself
    # takes this same mutex during the actual operation.
    flock reports/pvsg-trd-m4-150-v1/.gpu.lock true
    bash "$launcher" "$@"
  fi
}
for model in "${models[@]}"; do
  run profile --model "$model"
  if [[ $action != profile ]]; then
    for seed in "${seeds[@]}"; do
      run train --model "$model" --seed "$seed"
    done
  fi
done
