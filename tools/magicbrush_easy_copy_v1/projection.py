"""Conservative simplification: no semantic whitelist, no erased property edits.

Source graphs and existing target nodes are untouched. For newly added nodes,
unspecified appearance is optional only for plain object-addition requests.
The decision uses the instruction and source, never target attributes. Edges,
labels, multiplicity and observed incidental structural edits remain supervised.
"""
import re
from dataclasses import replace

from src.editing.operations import EditGraph, apply_operations, diff_aligned_graphs
from tools.magicbrush_lexical_pilot_v1.features import words

# Conservative, versioned guard. Unrecognized request forms keep all details.
PROTECTED = set(
    """red green blue yellow orange purple pink black white grey gray brown
blond blonde gold golden silver bright dark light pale colorful colour color coloured
colored wooden wood metal metallic leather rubber glass plastic fabric furry striped
stripe spot spotted pattern plaid checkered round square oval large small big tiny tall
taller short long thin thick wide narrow huge little old young new broken open closed
empty full wet dry bald hairy smiling happy sad angry facing looking sitting standing
upright lying screen monitor display picture image photo photograph print printed
wallpaper graphic drawing text word letter lettering number logo symbol face design
page webpage content coffee tea hair hat shirt dress tie sleeve cap shoe coat pant
snow snowy rain rainy fog foggy shadow reflection flame fire smoke dense sparse
sleep sleeping asleep awake waking crouching running chasing laughing crying painting painted drawing
ball scoop more less bigger smaller longer shorter third second first two three four five""".split()
)
PLAIN_ADD = re.compile(r"^(?:(?:please|can you|could you)\s+)?(?:add|insert|put|place)\b|^have there be\b", re.I)


def omit_added_appearance(source, instruction):
    tokens = set(words(instruction))
    if not PLAIN_ADD.search(instruction.strip()) or tokens & PROTECTED:
        return False
    # A source property mentioned to identify an object must remain available.
    # Exclude glue words; otherwise an attribute 'on left' matches every 'on'.
    glue = {"a", "an", "the", "on", "in", "of", "with", "and", "to", "at", "by", "be", "there"}
    return not any((set(words(a)) - glue) & tokens for n in source.nodes for a in n.attributes)


def project_example(example):
    simplify = omit_added_appearance(example.source, example.instruction)
    source_ids = {n.id for n in example.source.nodes}
    # Retain attributes when multiple new same-name objects need distinct IDs.
    new_labels = [n.label for n in example.target.nodes if n.id not in source_ids]
    nodes, removed = [], []
    for node in example.target.nodes:
        drop = simplify and node.id not in source_ids and new_labels.count(node.label) == 1
        after = replace(node, attributes=()) if drop else node
        if after != node:
            removed.append({"node_id": node.id, "label": node.label, "attributes": list(node.attributes)})
        nodes.append(after)
    target = EditGraph(tuple(nodes), example.target.edges)
    operations = diff_aligned_graphs(example.source, target)
    if apply_operations(example.source, operations) != target:
        raise AssertionError("Target not reconstructed")
    if {(o.kind, o.entity_id) for o in operations} != {(o.kind, o.entity_id) for o in example.operations}:
        raise AssertionError("Simplification changed edit types or IDs")
    return replace(example, target=target, operations=operations), {
        "plain_addition": simplify,
        "omitted_new_node_appearance": removed,
        "source_unchanged": True,
        "all_existing_target_nodes_unchanged": True,
    }
