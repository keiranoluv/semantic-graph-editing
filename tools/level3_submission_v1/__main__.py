import argparse
import json
from pathlib import Path

from .bundle import DEFAULT, build, prepare, verify

parser = argparse.ArgumentParser(description="CPU-only Level 3 archive; no inference or upload.")
parser.add_argument("stage", choices=["prepare", "build", "verify"])
parser.add_argument("--output", type=Path, default=DEFAULT)
parser.add_argument("--archive", type=Path)
args = parser.parse_args()
if args.stage == "prepare":
    m = prepare(args.output)
    result = {
        "output": str(args.output),
        "counts": m["counts"],
        "artifact_files": len(m["files"]),
        "uncompressed_bytes": sum(x["size"] for x in m["files"].values()),
    }
elif args.stage == "build":
    result = build(args.output)
else:
    if args.archive is None:
        parser.error("verify requires --archive")
    result = verify(args.archive)
print(json.dumps(result, ensure_ascii=False))
