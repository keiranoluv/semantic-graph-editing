#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
stage=${1:-train}
if [[ "$stage" == smoke-cpu ]]; then export CUDA_VISIBLE_DEVICES=''; fi
# Use the isolated compatible protobuf already used with this W&B SDK.
if ! myenv/bin/python -c 'import wandb' 2>/dev/null; then
    proto="$PWD/data/cache/magicbrush-prompt-intent-v1/protobuf-5.29.5"
    if [[ ! -d "$proto/google/protobuf" ]]; then
        proto="$PWD/data/cache/gqa-gat-loss-v1/protobuf-5.29.5"
        mkdir -p "$(dirname "$proto")"
        (
            flock -x 9
            if [[ ! -d "$proto" ]]; then
                staging=$(mktemp -d "$(dirname "$proto")/.protobuf-XXXXXX")
                trap 'rm -rf -- "$staging"' EXIT
                myenv/bin/python -m pip install --no-deps --only-binary=:all: --target "$staging" protobuf==5.29.5
                mv "$staging" "$proto"
            fi
        ) 9>"$proto.lock"
    fi
    export PYTHONPATH="$proto${PYTHONPATH:+:$PYTHONPATH}"
fi
myenv/bin/python -c 'import wandb; print("W&B SDK", wandb.__version__)'
mkdir -p logs
logfile="logs/gqa-gat-loss-v1-$stage-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Log: $logfile"
exec myenv/bin/python -u -m tools.gqa_loss_ablation_v1 "$stage" > >(tee -i "$logfile") 2>&1
