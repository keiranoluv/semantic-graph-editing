"""Score one MagicBrush checkpoint on a pinned split."""

import argparse
import json
from pathlib import Path

from task2_graph_editing.magicbrush.dataset import rows
from task2_graph_editing.magicbrush.examples import example
from task2_graph_editing.magicbrush.predict import load_model
from task2_graph_editing.magicbrush.runtime import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seed", type=int, choices=(42, 123, 3407), default=123)
    p.add_argument("--policy", choices=("prompt", "graph_diff"), default="prompt")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--max-examples", type=int)
    a = p.parse_args()
    if a.max_examples is not None and a.max_examples < 1:
        p.error("--max-examples must be positive")
    field = "operations" if a.policy == "prompt" else "graph_diff_operations"
    examples = [example(r, r[field]) for r in rows(a.split)]
    if a.max_examples is not None:
        examples = examples[: a.max_examples]
    model, store, config, entry = load_model(examples, seed=a.seed, policy=a.policy, checkpoint=a.checkpoint)
    metrics, _ = validate(model, examples, store, config, "cpu")
    print(
        json.dumps(
            {"checkpoint": entry["checkpoint"], "split": a.split, "examples": len(examples), "metrics": metrics},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
