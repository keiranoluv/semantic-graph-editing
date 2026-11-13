"""Independent supported instruction grammar and exact change agreement."""

import re

from tools.instructpix2pix_caption_graph_v1.parser import ATTRIBUTES, RELATION_PATTERN, RELATIONS, clean, noun_phrase


def intent(instruction):
    text = instruction.casefold().replace("’", "'") if isinstance(instruction, str) else ""
    text = re.sub(r"^(?:please )", "", text.strip()).rstrip(".")
    # Normalize a possessive color request without inventing an object or value.
    text = re.sub(r"^(change|set) (.+?)'s (?:color|colour) to (\w+)$", r"make \2 \3", text)
    text = re.sub(r"^change the (?:color|colour) of (.+?) to (\w+)$", r"make \1 \2", text)
    text = clean(text)

    def phrase(value):
        label, attrs, _ = noun_phrase(value)
        return {"label": label, "attributes": attrs}

    match = re.fullmatch(r"(?:replace|swap) (.+?) (?:with|for) (.+)", text)
    if not match:
        match = re.fullmatch(r"(?:change|turn|make) (.+?) (?:to|into) (.+)", text)
    if match:
        # Attribute instructions have the same object on both sides.
        if match[2] in ATTRIBUTES:
            return {"kind": "attribute", "before": phrase(match[1]), "attribute": match[2]}
        return {"kind": "replace", "before": phrase(match[1]), "after": phrase(match[2])}
    match = re.fullmatch(r"(?:make|paint|color|colour) (.+?) (" + "|".join(sorted(ATTRIBUTES)) + r")", text)
    if match:
        return {"kind": "attribute", "before": phrase(match[1]), "attribute": match[2]}
    match = re.fullmatch(r"(?:move|put|place) (.+?) (" + RELATION_PATTERN + r") (.+)", text)
    if match:
        return {
            "kind": "relation",
            "before": phrase(match[1]),
            "context": phrase(match[3]),
            "predicate": RELATIONS[match[2]],
        }
    match = re.fullmatch(r"(?:take|move) (.+?) off (.+)", text)
    if match:
        return {"kind": "delete_relation", "before": phrase(match[1]), "context": phrase(match[2]), "predicate": "on"}
    match = re.fullmatch(r"(?:add|insert) (.+)", text)
    if match:
        parts = re.split(r" (" + RELATION_PATTERN + r") ", match[1])
        if len(parts) == 3:
            return {
                "kind": "add",
                "after": phrase(parts[0]),
                "context": phrase(parts[2]),
                "predicate": RELATIONS[parts[1]],
            }
        return {"kind": "add", "after": phrase(match[1])}
    match = re.fullmatch(r"(?:remove|delete|erase) (.+)", text)
    if match:
        return {"kind": "delete", "before": phrase(match[1])}
    raise ValueError("unsupported_instruction_grammar")


def matches(node, description):
    return node.label == description["label"] and set(description["attributes"]) <= set(node.attributes)


def consistent(source, target, operations, request):
    """Require *all* caption changes to match one supported instruction."""
    old, new = {n.id: n for n in source.nodes}, {n.id: n for n in target.nodes}
    kind = request["kind"]
    node_ops = [o for o in operations if o.kind.endswith("_node")]
    edge_ops = [o for o in operations if o.kind.endswith("_edge")]
    if kind in ("replace", "attribute"):
        if len(node_ops) != 1 or node_ops[0].kind != "replace_node" or edge_ops:
            return False
        before, after = node_ops[0].before, node_ops[0].after
        if not matches(before, request["before"]):
            return False
        if kind == "replace":
            return (
                before.label != after.label
                and matches(after, request["after"])
                and before.attributes == after.attributes == request["after"]["attributes"]
            )
        added = set(after.attributes) - set(before.attributes)
        removed = set(before.attributes) - set(after.attributes)
        from tools.magicbrush_grounded_v1.language import COLORS

        # Color changes preserve every noncolor attribute. Other attributes can
        # only be added; replacing an arbitrary adjective has unclear semantics.
        return (
            before.label == after.label
            and added == {request["attribute"]}
            and (not removed or (request["attribute"] in COLORS and removed <= set(COLORS)))
        )
    if kind in ("add", "delete"):
        op_kind, field = ("add_node", "after") if kind == "add" else ("delete_node", "before")
        if len(node_ops) != 1 or node_ops[0].kind != op_kind:
            return False
        node = getattr(node_ops[0], field)
        if not matches(node, request[field]) or node.attributes != request[field]["attributes"]:
            return False
        expected_edge_kind = "add_edge" if kind == "add" else "delete_edge"
        if any(
            o.kind != expected_edge_kind or node.id not in (getattr(o, field).subject, getattr(o, field).object)
            for o in edge_ops
        ):
            return False
        if "context" in request:
            return len(edge_ops) == 1 and any(
                getattr(o, field).subject == node.id
                and getattr(o, field).predicate == request["predicate"]
                and matches(new[getattr(o, field).object], request["context"])
                for o in edge_ops
            )
        # Bare "add pillow" cannot justify inventing a new spatial relationship.
        return not edge_ops if kind == "add" else True
    if kind == "relation":
        if node_ops or len(edge_ops) != 1 or edge_ops[0].kind not in ("add_edge", "replace_edge"):
            return False
        edge = edge_ops[0].after
        return (
            edge.predicate == request["predicate"]
            and matches(old[edge.subject], request["before"])
            and matches(old[edge.object], request["context"])
        )
    if kind == "delete_relation":
        if node_ops or len(edge_ops) != 1 or edge_ops[0].kind != "delete_edge":
            return False
        edge = edge_ops[0].before
        return (
            edge.predicate == request["predicate"]
            and matches(old[edge.subject], request["before"])
            and matches(old[edge.object], request["context"])
        )
    return False
