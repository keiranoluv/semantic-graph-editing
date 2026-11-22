"""Dataset-specific provenance loader, sharing the existing Task B Example/codec.

The frozen MagicBrush loader requires images and official-train provenance;
inventing pixels or claiming crowdsourced dev is train would violate its contract.
This versioned entry point keeps those contracts intact and returns identical
Example/model_input objects for the graph predictor.
"""

from itertools import permutations

from common.io import digest
from task2_graph_editing.he2020.preprocessing.alignment import validate_raw


def source_fingerprint(raw):
    """ID/order independent graph group; no target information, max seven nodes."""
    nodes, edges = validate_raw(raw)
    if len(nodes) > 7:
        raise ValueError("fingerprint_search_budget")
    # Only permute within equal labels; all permutations across labels would
    # produce a larger label tuple and cannot be the lexicographic minimum.
    groups = {}
    for k, v in nodes.items():
        groups.setdefault(v, []).append(k)
    orders = [()]
    for label in sorted(groups):
        orders = [prefix + order for prefix in orders for order in permutations(groups[label])]
    encodings = []
    for order in orders:
        ids = {k: i for i, k in enumerate(order)}
        triples = []
        for (u, v), p in edges.items():
            a, b = ids[u], ids[v]
            if not raw["directed"]:
                a, b = sorted((a, b))
            triples.append((a, p, b))
        encodings.append((tuple(nodes[k] for k in order), tuple(sorted(triples))))
    return digest([raw["directed"], min(encodings)])
