"""Deterministic lexical features from source graph + instruction ONLY."""
import re
from functools import lru_cache

import torch

from tools.edit_model_v1.model import collate_inputs as base_collate

LABEL_FEATURES = ("exact", "head_only", "new_exact", "first_new_exact", "after_substitution", "existing_exact")
NODE_FEATURES = (
    "exact",
    "head",
    "attribute_overlap",
    "unique_exact",
    "cosine",
    "remove_exact",
    "change_exact",
    "add_exact",
    "remove_head",
    "change_head",
    "remove_attribute",
    "change_attribute",
)
EDGE_FEATURES = (
    "subject_exact",
    "object_exact",
    "both_exact",
    "predicate_exact",
    "remove_subject",
    "remove_object",
    "change_subject",
    "change_object",
)
IRREGULAR = {
    "men": "man",
    "women": "woman",
    "people": "person",
    "children": "child",
    "mice": "mouse",
    "geese": "goose",
    "knives": "knife",
    "leaves": "leaf",
    "feet": "foot",
    "teeth": "tooth",
}
PROTECTED = {"glasses", "grass", "bus", "gas", "tennis", "skis", "pants", "trousers", "jeans", "news"}
REMOVE = {"remove", "delete", "erase", "without", "disappear"}
ADD = {"add", "insert", "put", "place"}
CHANGE = {"change", "replace", "turn", "make", "let", "have", "become"}


def words(text):
    result = []
    for w in re.findall(r"[a-z0-9]+", text.lower().replace("'s", "")):
        if w in IRREGULAR:
            w = IRREGULAR[w]
        elif w not in PROTECTED:
            if len(w) > 4 and w.endswith("ies"):
                w = w[:-3] + "y"
            elif len(w) > 4 and w.endswith(("ches", "shes", "xes", "zes")):
                w = w[:-2]
            elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
                w = w[:-1]
        result.append(w)
    return tuple(result)


def spans(phrase, tokens):
    return [i for i in range(len(tokens) - len(phrase) + 1) if phrase and tokens[i : i + len(phrase)] == phrase]


@lru_cache(maxsize=4096)
def lexical(instruction, node_payloads, edge_payloads, labels):
    tokens = words(instruction)
    bag = set(tokens)
    node_names = [words(name) for name, attrs in node_payloads]
    source_names = set(node_names)
    exact = [bool(spans(name, tokens)) for name in node_names]
    head = [bool(name) and name[-1] in bag for name in node_names]
    remove = bool(bag & REMOVE) or bool(spans(("get", "rid", "of"), tokens))
    add = bool(bag & ADD) or bool(spans(("there", "be"), tokens))
    change = bool(bag & CHANGE)
    substitution = bool(bag & {"replace", "turn", "change", "instead"})
    connectors = [i for i, t in enumerate(tokens) if t in {"with", "into", "to"}] if substitution else []
    label_tokens = [words(x) for x in labels]
    hits = [spans(x, tokens) for x in label_tokens]
    first = min(
        (s[0] for name, s in zip(label_tokens, hits, strict=True) if s and name not in source_names), default=-1
    )
    lf = []
    for name, hit in zip(label_tokens, hits, strict=True):
        mentioned = bool(hit)
        existing = name in source_names
        lf.append(
            [
                float(mentioned),
                float(not mentioned and bool(name) and name[-1] in bag),
                float(mentioned and not existing),
                float(mentioned and not existing and hit[0] == first),
                float(mentioned and any(i < c for i in connectors for c in hit)),
                float(mentioned and existing),
            ]
        )
    nf = []
    for i, (_name, attrs) in enumerate(node_payloads):
        # Scoped attributes such as "shirt blue" can match "blue shirt".
        attr = set(w for a in attrs for w in words(a)) - {"position", "left", "right", "upper", "lower", "center"}
        overlap = len(attr & bag) / max(len(attr), 1)
        x, h = float(exact[i]), float(head[i])
        nf.append(
            [
                x,
                h,
                overlap,
                float(exact[i] and sum(exact) == 1),
                0.0,
                remove * x,
                change * x,
                add * x,
                remove * h,
                change * h,
                remove * overlap,
                change * overlap,
            ]
        )
    ef = []
    for subject, predicate, obj in edge_payloads:
        a, b = float(exact[subject]), float(exact[obj])
        ef.append(
            [a, b, a * b, float(bool(spans(words(predicate), tokens))), remove * a, remove * b, change * a, change * b]
        )
    return lf, nf, ef, float(substitution)


class GroundingStore:
    def __init__(self, store, vocabulary):
        self.base = store
        self.vocabulary = vocabulary
        self.dim = store.dim

    def __getitem__(self, text):
        return self.base[text]


def collate_inputs(items, store, *, device="cpu"):
    batch = base_collate(items, store, device="cpu")
    n, m = batch["nodes"].shape[1], batch["edges"].shape[1]
    labels = tuple(store.vocabulary["labels"])
    lf = torch.zeros(len(items), len(labels), len(LABEL_FEATURES))
    nf = torch.zeros(len(items), n, len(NODE_FEATURES))
    ef = torch.zeros(len(items), m, len(EDGE_FEATURES))
    substitution = torch.zeros(len(items), 1, 1)
    for b, item in enumerate(items):
        # Accepting only these keys makes accidental target-conditioned calls fail.
        if set(item) != {"source", "instruction"}:
            raise ValueError("Only source and instruction allowed")
        graph = item["source"]
        ids = {x.id: i for i, x in enumerate(graph.nodes)}
        ls, ns, es, sub = lexical(
            item["instruction"],
            tuple((x.label, x.attributes) for x in graph.nodes),
            tuple((ids[e.subject], e.predicate, ids[e.object]) for e in graph.edges),
            labels,
        )
        lf[b] = torch.tensor(ls)
        if ns:
            nf[b, : len(ns)] = torch.tensor(ns)
        if es:
            ef[b, : len(es)] = torch.tensor(es)
        substitution[b] = sub
    nf[:, :, 4] = (
        torch.nn.functional.cosine_similarity(batch["nodes"], batch["instruction"][:, None], dim=-1)
        * batch["node_mask"]
    )
    batch.update(label_grounding=lf, node_grounding=nf, edge_grounding=ef, substitution=substitution)
    return {k: v.to(device) for k, v in batch.items()}
