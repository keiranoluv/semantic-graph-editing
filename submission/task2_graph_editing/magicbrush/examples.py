"""Build graph-edit supervision and verify pinned JSON inputs."""

from common.editing.codec import canonical_actions
from common.editing.data import Example
from common.editing.operations import EditGraph, Operation, apply_operations
from common.io import read, sha256


def checked_json(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Changed input: {path}")
    return read(path)


def example(row, ops):
    source = EditGraph.from_dict(row["source_graph"])
    actions = canonical_actions(source, [Operation.from_dict(op) for op in ops])
    return Example(
        row["id"],
        row["split"],
        row["source_group_id"],
        row["instruction"],
        source,
        apply_operations(source, actions),
        actions,
    )
