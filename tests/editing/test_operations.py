import itertools
import json

import pytest

from src.editing.evaluation import evaluate_edits
from src.editing.operations import Edge, EditGraph, Node, Operation, apply_operations, diff_aligned_graphs
from src.editing.splits import grouped_split


def graphs():
    source = EditGraph(
        (Node("girl", "girl"), Node("shirt", "shirt", ("red",)), Node("hat", "hat")),
        (Edge("wear", "girl", "wearing", "shirt"), Edge("has", "girl", "has", "hat")),
    )
    target = EditGraph(
        (Node("girl", "girl"), Node("shirt", "shirt", ("blue",)), Node("bag", "bag")),
        (Edge("wear", "girl", "holding", "shirt"), Edge("bag-edge", "girl", "carrying", "bag")),
    )
    return source, target


def test_json_round_trip_preserves_canonical_attributes_and_operations():
    source, target = graphs()
    restored = EditGraph.from_dict(json.loads(json.dumps(source.to_dict())))
    assert restored == source
    ops = diff_aligned_graphs(source, target)
    restored_ops = [Operation.from_dict(json.loads(json.dumps(op.to_dict()))) for op in ops]
    assert apply_operations(restored, restored_ops) == target
    assert Node("a", "shirt", ("red", "cotton")) == Node("a", "shirt", ("cotton", "red"))


def test_all_six_operations_round_trip_and_order_independence():
    source, target = graphs()
    ops = diff_aligned_graphs(source, target)
    assert {o.kind for o in ops} == {
        "add_node",
        "delete_node",
        "replace_node",
        "add_edge",
        "delete_edge",
        "replace_edge",
    }
    for ordering in itertools.permutations(ops):
        assert apply_operations(source, ordering) == target
    assert apply_operations(target, diff_aligned_graphs(target, source)) == source


def test_failed_node_delete_is_atomic_and_requires_incident_edge_edits():
    source, _ = graphs()
    original = source.to_dict()
    hat = next(n for n in source.nodes if n.id == "hat")
    with pytest.raises(ValueError, match="Dangling"):
        apply_operations(source, [Operation("delete_node", before=hat)])
    assert source.to_dict() == original


def test_source_preconditions_conflicting_writes_and_id_preservation():
    source, _ = graphs()
    with pytest.raises(ValueError, match="precondition"):
        apply_operations(source, [Operation("delete_node", before=Node("hat", "wrong"))])
    op = Operation("replace_node", before=Node("hat", "hat"), after=Node("hat", "cap"))
    with pytest.raises(ValueError, match="Multiple writes"):
        apply_operations(source, [op, op])
    with pytest.raises(ValueError, match="preserve"):
        Operation("replace_node", before=Node("hat", "hat"), after=Node("new", "cap"))


def test_multirelation_pair_allowed_duplicate_semantic_edge_rejected():
    nodes = (Node("a", "girl"), Node("b", "hair"))
    EditGraph(nodes, (Edge("e1", "a", "has", "b"), Edge("e2", "a", "with", "b")))
    with pytest.raises(ValueError, match="Duplicate semantic"):
        EditGraph(nodes, (Edge("e1", "a", "has", "b"), Edge("e2", "a", "has", "b")))


@pytest.mark.parametrize(
    "kind,before,after",
    [
        ("unknown", None, None),
        ("add_node", None, Edge("e", "a", "on", "b")),
        ("replace_node", "wrong", "wrong"),
        ("delete_edge", None, None),
        ("replace_node", Node("a", "girl"), Node("a", "girl")),
    ],
)
def test_invalid_operation_payload_rejected(kind, before, after):
    with pytest.raises(ValueError):
        Operation(kind, before, after)


def test_metrics_penalize_duplicate_predictions_and_report_invalid_graph():
    source, target = graphs()
    ops = diff_aligned_graphs(source, target)
    summary = evaluate_edits([{"source": source, "target": target, "expected": ops, "predicted": ops + ops}])
    assert summary["micro"]["tp"] == 6
    assert summary["micro"]["fp"] == 6
    assert summary["micro"]["fn"] == 0
    assert summary["micro"]["f1"] == pytest.approx(2 / 3)
    assert summary["exact_operation_set_accuracy"] == summary["valid_transaction_rate"] == 0


def test_empty_and_noop_metrics_not_fake_perfect_f1():
    empty = evaluate_edits([])
    assert empty["micro"]["f1"] is None and empty["exact_operation_set_accuracy"] is None
    graph = EditGraph((Node("a", "girl"),))
    result = evaluate_edits([{"source": graph, "target": graph, "expected": [], "predicted": []}])
    assert result["micro"]["f1"] is None and result["exact_operation_set_accuracy"] == 1


def test_wrong_entity_same_label_is_not_a_hit_and_invalid_gt_rejected():
    source = EditGraph((Node("girl1", "girl"), Node("girl2", "girl")))
    expected = [Operation("replace_node", Node("girl1", "girl"), Node("girl1", "woman"))]
    wrong = [Operation("replace_node", Node("girl2", "girl"), Node("girl2", "woman"))]
    target = apply_operations(source, expected)
    row = {"source": source, "target": target, "expected": expected, "predicted": wrong}
    result = evaluate_edits([row])
    assert result["micro"]["fp"] == result["micro"]["fn"] == 1
    assert result["micro"]["tp"] == 0 and result["valid_transaction_rate"] == 1
    with pytest.raises(ValueError, match="GT"):
        evaluate_edits([{**row, "expected": []}])


def test_grouped_split_keeps_multiturn_images_together():
    records = [{"id": f"{g}-{turn}", "source_group_id": str(g)} for g in range(20) for turn in range(3)]
    splits = grouped_split(records)
    assert splits == grouped_split(reversed(records))
    assert sorted(map(len, splits.values())) == [9, 9, 42]
    for group in range(20):
        assert sum(any(f"{group}-{turn}" in ids for turn in range(3)) for ids in splits.values()) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        grouped_split(records + records[:1])
