"""Copy or verify the exact external inputs of the CPU submission demos."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("chair", "live", "overview"))
    parser.add_argument("--from-dir", type=Path, help="Original workspace or transferred directory tree")
    parser.add_argument("--to-dir", type=Path, default=ROOT, help="Destination, default: this checkout")
    args = parser.parse_args()
    index = json.loads((ROOT / "submission/assets/demo-assets.json").read_text())
    files = {name: row for name, row in index.items() if args.profile in row["profiles"]}
    for name, record in files.items():
        dest = args.to_dir / name
        if matches(dest, record):
            continue
        if dest.exists():
            parser.exit(1, f"Existing file differs from manifest: {dest}\n")
        if args.from_dir is None:
            parser.exit(1, f"Missing {name}. Supply --from-dir to restore it.\n")
        source = args.from_dir / name
        if not source.is_file():
            parser.exit(1, f"Missing in source: {source}\n")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            staging = Path(tmp.name)
        try:
            shutil.copyfile(source, staging)
            if not matches(staging, record):
                parser.exit(1, f"Size or SHA-256 mismatch: {source}\n")
            # Publish only complete, verified files; preserve an existing destination.
            try:
                os.link(staging, dest)
            except FileExistsError:
                if not matches(dest, record):
                    raise
        finally:
            staging.unlink(missing_ok=True)
    size = sum(row["bytes"] for row in files.values())
    print(f"{args.profile}: verified {len(files)} files, {size:,} bytes")


if __name__ == "__main__":
    main()
