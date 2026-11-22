"""Train one VG150 condition using arguments recorded for the reported runs."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from common.paths import ROOT


def main():
    recipes = json.loads((ROOT / "assets/training_recipes.json").read_text())["vg150"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted({r["condition"] for r in recipes}), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 3407), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recipe = next(r for r in recipes if (r["condition"], r["seed"]) == (args.condition, args.seed))
    name = f"{args.condition}-seed{args.seed}"
    output = args.output_dir if recipe["nested_run_directory"] else args.output_dir / name
    command = [sys.executable, *recipe["argv"], "--output-dir", str(output)]
    if recipe["nested_run_directory"]:
        command.extend(("--run-name", name))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
