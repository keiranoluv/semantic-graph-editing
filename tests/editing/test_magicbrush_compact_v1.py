"""CPU regression tests for information-preserving boundaries of coarse labels."""
from dataclasses import replace
from pathlib import Path

import pytest

from src.editing.operations import Edge, EditGraph, Node, apply_operations, diff_aligned_graphs
from tools.edit_model_v1.data import Example
from tools.emnlp2020_graph_modification_v1.dataset import read
from tools.magicbrush_compact_v1.projection import project_attribute, project_example, project_graph

POLICY = read(Path(__file__).resolve().parents[2] / "configs/level2/magicbrush-compact-v1-policy.json")


@pytest.mark.parametrize(
    "original,expected",
    [
        ("white and pale green", ("green", "white")),
        ("dark blue", ("blue",)),
        ("white icing", ("icing white",)),
        ("blue shirt", ("shirt blue",)),
        ("green leaves", ("leaves green",)),
        ("wooden", ("wood",)),
        ("dark wood", ("wood",)),
        ("clear plastic", ("plastic", "transparent")),
        ("large blue star-like graphic", ("star patterned",)),
        ("door closed", ("door closed",)),
        ("white BRR text", ("text BRR",)),
        ("number 18", ("number 18",)),
        ("not red", ()),
        ("blue monster illustration", ()),
        ("partially visible", ()),
        ("upper right", ("position right", "position upper")),
        ("blond mohawk", ("hair blond", "hair mohawk")),
    ],
)
def test_coarse_attributes_keep_scope_literal_and_negation(original, expected):
    got, _ = project_attribute(original, POLICY)
    assert got == tuple(sorted(expected))
    for attr in got:
        assert project_attribute(attr, POLICY)[0] == (attr,)


def example(source, target):
    return Example("toy", "train", "group", "Make a change", source, target, diff_aligned_graphs(source, target))


def test_identical_projection_both_roles_and_recomputed_payloads():
    a = EditGraph((Node("n1", "table", ("wooden", "left", "partially visible")),))
    b = EditGraph((Node("n1", "table", ("wood", "left", "red")),))
    compact, audit = project_example(example(a, b), POLICY)
    assert apply_operations(compact.source, compact.operations) == compact.target
    assert compact.operations[0].before == compact.source.nodes[0]
    assert compact.source == project_example(example(a, a), POLICY)[0].source
    assert compact.source == project_example(example(b, a), POLICY)[0].target
    assert compact == project_example(compact, POLICY)[0]
    assert any(x["original"] == "partially visible" and x["core"] == [] for x in audit["source_attributes"])


def test_noop_record_is_flagged_without_inventing_an_edit():
    a = EditGraph((Node("n", "object", ("partly obscured",)),))
    b = EditGraph((Node("n", "object", ("partially visible",)),))
    compact, audit = project_example(example(a, b), POLICY)
    assert compact.operations == () and "no_core_edit" in audit["flags"]
    assert len(audit["removed_operations"]) == 1


def test_coarse_positions_keep_duplicate_instances_distinguishable():
    source = EditGraph((Node("n", "water"),))
    target = EditGraph(
        (*source.nodes, Node("d1", "duck", ("left", "brown")), Node("d2", "duck", ("right", "brown"))),
        (Edge("e1", "d1", "in", "n"), Edge("e2", "d2", "in", "n")),
    )
    compact, audit = project_example(example(source, target), POLICY)
    assert not audit["flags"] and len(compact.target.nodes) == 3
    assert compact.target.edges == target.edges
    ambiguous = replace(
        target,
        nodes=(*source.nodes, Node("d1", "duck", ("brown", "tiny mark")), Node("d2", "duck", ("brown", "large mark"))),
    )
    _, audit = project_example(example(source, ambiguous), POLICY)
    assert "unsupported_ambiguous_new_instance_identity" in audit["flags"]


def test_changed_instruction_cannot_change_source_or_target_projection():
    graph = EditGraph((Node("n", "ball", ("white",)),))
    e = example(graph, graph)
    assert (
        project_example(e, POLICY)[0].source == project_example(replace(e, instruction="Make it red"), POLICY)[0].source
    )
    assert project_graph(graph, POLICY)[0].nodes[0].attributes == ("white",)
