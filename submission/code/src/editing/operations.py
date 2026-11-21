"""Atomic, order-independent edits with explicit stable node and edge identities.

Source/target identity alignment must happen before diffing. This module does
not infer correspondences between outputs of independent scene graph models.
"""

from dataclasses import asdict, dataclass
from typing import Literal

KINDS = ("add_node", "delete_node", "replace_node", "add_edge", "delete_edge", "replace_edge")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    attributes: tuple[str, ...] = ()

    def __post_init__(self):
        if not _text(self.id) or not _text(self.label) or any(not _text(a) for a in self.attributes):
            raise ValueError("Node ID, label and attributes must be nonempty strings")
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("Repeated node attribute")
        object.__setattr__(self, "attributes", tuple(sorted(self.attributes)))


@dataclass(frozen=True)
class Edge:
    id: str
    subject: str
    predicate: str
    object: str

    def __post_init__(self):
        if any(not _text(x) for x in (self.id, self.subject, self.predicate, self.object)):
            raise ValueError("Edge fields must be nonempty strings")


@dataclass(frozen=True)
class EditGraph:
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda x: x.id)))
        object.__setattr__(self, "edges", tuple(sorted(self.edges, key=lambda x: x.id)))
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes) or len({e.id for e in self.edges}) != len(self.edges):
            raise ValueError("Node/edge IDs must be unique within their namespace")
        if any(e.subject not in ids or e.object not in ids for e in self.edges):
            raise ValueError("Dangling edge endpoint")
        triples = {(e.subject, e.predicate, e.object) for e in self.edges}
        if len(triples) != len(self.edges):
            raise ValueError("Duplicate semantic edge; deduplicate annotation rows before conversion")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != {"nodes", "edges"}:
            raise ValueError("Graph JSON requires nodes and edges only")
        return cls(tuple(Node(**n) for n in payload["nodes"]), tuple(Edge(**e) for e in payload["edges"]))


@dataclass(frozen=True)
class Operation:
    kind: Literal["add_node", "delete_node", "replace_node", "add_edge", "delete_edge", "replace_edge"]
    before: Node | Edge | None = None
    after: Node | Edge | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("Unknown edit operation")
        action, entity = self.kind.split("_")
        expected_type = Node if entity == "node" else Edge
        if any(type(x) is not expected_type for x in (self.before, self.after) if x is not None):
            raise ValueError("Operation payload type differs from kind")
        if action == "add" and (self.before is not None or self.after is None):
            raise ValueError("Add requires after only")
        if action == "delete" and (self.before is None or self.after is not None):
            raise ValueError("Delete requires before only")
        if action == "replace":
            if self.before is None or self.after is None or self.before == self.after:
                raise ValueError("Replace requires distinct before and after")
            if self.before.id != self.after.id:
                raise ValueError("Replace must preserve instance ID")

    @property
    def entity_id(self):
        return (self.before if self.before is not None else self.after).id

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != {"kind", "before", "after"} or payload["kind"] not in KINDS:
            raise ValueError("Invalid operation JSON fields")
        entity = Node if payload["kind"].endswith("_node") else Edge
        before = entity(**payload["before"]) if payload["before"] is not None else None
        after = entity(**payload["after"]) if payload["after"] is not None else None
        return cls(payload["kind"], before, after)


def apply_operations(source: EditGraph, operations) -> EditGraph:
    """Validate all preconditions against source, then apply as one transaction.

    Deleting a node does NOT silently delete its edges: those deletions or endpoint
    replacements must be explicit. A failed transaction leaves source unchanged."""
    nodes, edges = {x.id: x for x in source.nodes}, {x.id: x for x in source.edges}
    touched = set()
    for op in operations:
        action, entity = op.kind.split("_")
        key = (entity, op.entity_id)
        if key in touched:
            raise ValueError("Multiple writes to the same entity; use one canonical replacement")
        touched.add(key)
        table = nodes if entity == "node" else edges
        if action == "add":
            if op.entity_id in table:
                raise ValueError("Cannot add an existing entity ID")
        elif table.get(op.entity_id) != op.before:
            raise ValueError("Edit precondition does not match source")
        if action == "delete":
            del table[op.entity_id]
        else:
            table[op.entity_id] = op.after
    return EditGraph(tuple(nodes.values()), tuple(edges.values()))


def diff_aligned_graphs(source: EditGraph, target: EditGraph) -> tuple[Operation, ...]:
    """Canonical set of six operation types, for ALREADY aligned graph IDs."""
    result = []
    for entity, left, right in (("node", source.nodes, target.nodes), ("edge", source.edges, target.edges)):
        old, new = {x.id: x for x in left}, {x.id: x for x in right}
        for key in sorted(old.keys() | new.keys()):
            if key not in old:
                result.append(Operation(f"add_{entity}", after=new[key]))
            elif key not in new:
                result.append(Operation(f"delete_{entity}", before=old[key]))
            elif old[key] != new[key]:
                result.append(Operation(f"replace_{entity}", before=old[key], after=new[key]))
    return tuple(result)
