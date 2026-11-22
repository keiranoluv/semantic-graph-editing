"""Convert the official archive into the fixed graph-edit train/validation split."""

import argparse
import json
from pathlib import Path

from task2_graph_editing.he2020.data_processing.data import prepare
from task2_graph_editing.he2020.preprocessing.dataset import ARCHIVE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--output", type=Path, default=Path("data/processed/level2/emnlp2020-task-b-v2"))
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    print(json.dumps(prepare(a.archive, a.output, test=a.test), indent=2))


if __name__ == "__main__":
    main()
