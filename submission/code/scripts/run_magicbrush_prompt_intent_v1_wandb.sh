#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
runtime="${MAGICBRUSH_PYTHON:-myenv/bin/python}"
cache="$PWD/data/cache/magicbrush-prompt-intent-v1"
protobuf_dir="$cache/protobuf-5.29.5"
mkdir -p "$cache"
(
  flock -x 9
  if [[ ! -d "$protobuf_dir" ]]; then
    stage=$(mktemp -d "$cache/.protobuf-XXXXXX")
    trap 'rm -rf -- "$stage"' EXIT
    "$runtime" -m pip install --disable-pip-version-check --only-binary=:all: --no-deps \
      --target "$stage" protobuf==5.29.5
    mv -- "$stage" "$protobuf_dir"
  fi
) 9>"$cache/.protobuf-install.lock"

export PYTHONPATH="$protobuf_dir${PYTHONPATH:+:$PYTHONPATH}"
"$runtime" - <<'PY'
import wandb
from google import protobuf
from wandb.proto.wandb_telemetry_pb2 import Deprecated

if protobuf.__version__ != "5.29.5":
    raise RuntimeError(f"Unexpected protobuf {protobuf.__version__}")
print(f"MagicBrush tracking: wandb {wandb.__version__}, protobuf {protobuf.__version__}", flush=True)
PY

if [[ "${1:-}" == "--cpu-check" ]]; then
  CUDA_VISIBLE_DEVICES='' "$runtime" - <<'PY'
import tempfile
import wandb

with tempfile.TemporaryDirectory() as directory:
    with wandb.init(project="magicbrush-cpu-check", mode="offline", dir=directory) as run:
        run.log({"cpu_check": 1})
PY
fi
exec bash scripts/run_magicbrush_prompt_intent_v1.sh "$@"
