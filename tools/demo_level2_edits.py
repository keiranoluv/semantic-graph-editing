"""Small synthetic edit example: validates mechanics, NOT a trained model result."""

import json

from src.editing.evaluation import evaluate_edits
from src.editing.operations import Edge, EditGraph, Node, Operation, apply_operations, diff_aligned_graphs


def main():
    girl, shirt = Node("girl-1", "girl"), Node("shirt-1", "shirt", ("red",))
    source = EditGraph((girl, shirt), (Edge("wear-1", girl.id, "wearing", shirt.id),))
    instruction = "Change the girl's red shirt to blue."
    target = EditGraph((girl, Node(shirt.id, "shirt", ("blue",))), source.edges)
    expected = diff_aligned_graphs(source, target)
    # Intentionally edit the wrong object to demonstrate payload matching.
    wrong = Operation("replace_node", girl, Node(girl.id, girl.label, ("blue",)))
    metrics = evaluate_edits([{"source": source, "target": target, "expected": expected, "predicted": [wrong]}])
    print(
        json.dumps(
            {
                "example_type": "synthetic_mechanics_demo_not_MagicBrush_not_model_prediction",
                "source": source.to_dict(),
                "instruction": instruction,
                "expected_operations": [o.to_dict() for o in expected],
                "reconstructed_target": apply_operations(source, expected).to_dict(),
                "deliberately_wrong_prediction": wrong.to_dict(),
                "wrong_prediction_metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
