"""Conservative full-phrase attribute projection, without target/instruction peeking."""
import re
from dataclasses import replace

from src.editing.operations import EditGraph, apply_operations, diff_aligned_graphs
from tools.edit_model_v1.codec import canonical_actions


def normalized(text):
    return " ".join(re.sub(r"[-_/]", " ", text.lower()).split())


def tables(policy):
    aliases, families = {}, {}
    for family, entries in policy["families"].items():
        for canonical, variants in entries.items():
            families[canonical] = family
            for text in [canonical, *variants]:
                key = normalized(text)
                if key in aliases and aliases[key] != (canonical,):
                    raise ValueError("Conflicting alias: " + key)
                aliases[key] = (canonical,)
    for key, values in policy["explicit_aliases"].items():
        aliases[normalized(key)] = tuple(sorted(set(values)))
    return aliases, families


def project_attribute(text, policy):
    aliases, families = tables(policy)
    value = normalized(text)
    if value in aliases:
        return aliases[value], "exact_alias"
    if value in {"position " + w for w in policy["position_tokens"]}:
        return (value,), "coarse_position"
    position_words = value.split()
    if position_words and all(w in policy["position_tokens"] + policy["position_qualifiers"] for w in position_words):
        positions = {policy["position_aliases"].get(w, w) for w in position_words}
        kept = tuple(sorted("position " + w for w in positions if w in policy["position_tokens"]))
        return kept, "coarse_position" if kept else "detail_only"
    if re.fullmatch(r"number \d+", value):
        return (value,), "literal_number"
    literal = re.fullmatch(
        r"(?:(?:white|black|red|blue|green|yellow) )?([A-Z0-9]+(?: [A-Z0-9]+)*) (?:text|lettering)", text.strip()
    )
    if literal:
        return ("text " + literal[1],), "literal_text"
    if re.fullmatch(r"text [A-Z0-9]+(?: [A-Z0-9]+)*", text):
        return (text,), "literal_text"

    parts = policy["scoped_parts"]
    words = value.split()
    descriptors = set(policy["scoped_descriptors"]) | set(policy["families"]["color"])
    if len(words) >= 2 and (words[-1] in parts or words[0] in parts):
        part = words[-1] if words[-1] in parts else words[0]
        rest = words[:-1] if words[-1] in parts else words[1:]
        if rest and all(w in descriptors or w in ("and", "with") for w in rest):
            rest = [w for w in rest if w not in ("and", "with")]
            aliases_scoped = policy["scoped_descriptor_aliases"]
            part = aliases_scoped.get(part, part)
            return tuple(sorted({part + " " + aliases_scoped.get(w, w) for w in rest})), "scoped_property"

    def colors(phrase):
        # Parse the ENTIRE phrase. "red shirt", "not red" or "white icing"
        # must not become a color attribute of the whole object.
        words = re.sub(r"[,;&+]", " ", phrase).split()
        result, modifier_pending = set(), False
        for word in words:
            if word in policy["color_modifiers"]:
                modifier_pending = True
            elif word in aliases and len(aliases[word]) == 1 and families.get(aliases[word][0]) == "color":
                result.update(aliases[word])
                modifier_pending = False
            elif word in policy["color_separators"] and result and not modifier_pending:
                continue
            else:
                return ()
        return tuple(sorted(result)) if not modifier_pending else ()

    values = colors(value)
    if values:
        return values, "coarse_color_phrase"
    for modifier in policy["material_modifiers"]:
        prefix = modifier + " "
        if value.startswith(prefix):
            rest = aliases.get(value[len(prefix) :], ())
            if len(rest) == 1 and families.get(rest[0]) == "material":
                return rest, "material_without_shade"
    # A pattern's color is not necessarily the object's global color.
    for phrase, values in aliases.items():
        if len(values) == 1 and families.get(values[0]) == "pattern" and value.endswith(" " + phrase):
            if colors(value[: -(len(phrase) + 1)]):
                return values, "pattern_without_detail_color"
    return (), "detail_only"


def project_graph(graph, policy):
    nodes, details = [], []
    for node in graph.nodes:
        attributes = set()
        for value in node.attributes:
            core, reason = project_attribute(value, policy)
            attributes.update(core)
            details.append({"node_id": node.id, "original": value, "core": list(core), "rule": reason})
        nodes.append(replace(node, attributes=tuple(sorted(attributes))))
    return EditGraph(tuple(nodes), graph.edges), details


def project_example(example, policy):
    source, source_details = project_graph(example.source, policy)
    target, target_details = project_graph(example.target, policy)
    operations = diff_aligned_graphs(source, target)
    if apply_operations(source, operations) != target:
        raise AssertionError("Projected operations do not reconstruct target")
    flags = []
    if not operations:
        flags.append("no_core_edit")
    try:
        canonical_actions(source, operations)
    except ValueError as error:
        flags.append(str(error))
    old = {(o.kind, o.entity_id): o for o in example.operations}
    new = {(o.kind, o.entity_id): o for o in operations}
    removed = [op.to_dict() for key, op in old.items() if key not in new]
    # A graph projection may remove replacements; never invent structural edits.
    if any(o.kind != "replace_node" for o in example.operations if (o.kind, o.entity_id) not in new):
        raise AssertionError("Projection removed structural operation")
    return replace(example, source=source, target=target, operations=operations), {
        "source_attributes": source_details,
        "target_attributes": target_details,
        "removed_operations": removed,
        "flags": flags,
    }
