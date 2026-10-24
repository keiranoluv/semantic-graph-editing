#!/usr/bin/env bash
# Reproducible VG150 annotation download.
# Run from the repository root.

set -eu

ROOT="data/raw/vg150"
mkdir -p "$ROOT"

H5_URL="https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/main/VG-SGG-with-attri.h5"
DICT_URL="https://raw.githubusercontent.com/Prk10/VG150_VisualGenome/main/VG-SGG-dicts-with-attri.json"
META_URL="https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"

H5_SHA256="ad52a01f8e6142bb7bc62d91f97dad1567f44772a23eac68d60a5b7766ffe4b1"
DICT_SHA256="1255873ae0250555f59d20f8d1e5a4bdf22b331119502c850f765173b6963ca1"
META_JSON_SHA256="93d0053d7e8f451497646fa513e8fb9b1a889df6e757010b88e51e2e21803487"

download() {
    local url="$1"
    local dest="$2"

    if [ -s "$dest" ]; then
        echo "[SKIP] $dest already exists"
    else
        echo "[DOWNLOAD] $dest"
        curl -fL --retry 3 -o "${dest}.part" "$url"
        mv "${dest}.part" "$dest"
    fi
}

download "$H5_URL" "$ROOT/VG-SGG-with-attri.h5"
download "$DICT_URL" "$ROOT/VG-SGG-dicts-with-attri.json"
download "$META_URL" "$ROOT/image_data.json.zip"

echo "$H5_SHA256  $ROOT/VG-SGG-with-attri.h5" | sha256sum -c
echo "$DICT_SHA256  $ROOT/VG-SGG-dicts-with-attri.json" | sha256sum -c

python - "$ROOT" <<'INNER_PY'
import json
import sys
from pathlib import Path
from zipfile import ZipFile

root = Path(sys.argv[1])
raw_path = root / "image_data_raw.json"

if not raw_path.exists():
    with ZipFile(root / "image_data.json.zip") as archive:
        raw_path.write_bytes(archive.read("image_data.json"))

with (root / "VG-SGG-dicts-with-attri.json").open() as f:
    vocabulary = json.load(f)

with raw_path.open() as f:
    metadata = json.load(f)

for key in ("label_to_idx", "predicate_to_idx", "attribute_to_idx"):
    if key not in vocabulary:
        raise ValueError(f"Missing vocabulary key: {key}")

print(f"[OK] Raw metadata records: {len(metadata)}")
print("[OK] VG150 annotations downloaded.")
INNER_PY

echo "$META_JSON_SHA256  $ROOT/image_data_raw.json" | sha256sum -c
