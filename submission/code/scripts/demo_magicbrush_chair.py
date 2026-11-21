"""Run the submission notebook's held-out chair edit on CPU."""

import torch

from experiments.magicbrush_prompt_intent_v1.dataset import rows
from experiments.magicbrush_prompt_pilot_v1.run import example
from src.editing.operations import EditGraph
from tools.graph_editing_live_demo_v1 import predict


def main():
    torch.set_num_threads(2)
    row = next(r for r in rows("test") if r["id"] == "magicbrush:190052:2")
    ops, graph, checkpoint = predict(EditGraph.from_dict(row["source_graph"]), row["instruction"], seed=123)
    reference = example(row, row["operations"])
    print(row["id"], row["instruction"], checkpoint["checkpoint"], sep="\n")
    for op in ops:
        print(op.kind, op.before)
    exact_ops = set(ops) == set(reference.operations)
    exact_graph = graph == reference.target
    print("Khớp toàn bộ thao tác:", exact_ops)
    print("Khớp đồ thị đích:", exact_graph)
    if not (exact_ops and exact_graph):
        raise SystemExit("Prediction differs from the frozen reference")


if __name__ == "__main__":
    main()
