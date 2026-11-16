import argparse
import json
from pathlib import Path

from .core import build

parser = argparse.ArgumentParser(description="Render the finalized Level 3 analysis on CPU.")
parser.add_argument("--notes", type=Path, default=Path("reports/level3/evaluation200-v1/visual-review.json"))
parser.add_argument("--output", type=Path)
args = parser.parse_args()
print(json.dumps(build(args.notes, args.output), ensure_ascii=False))
