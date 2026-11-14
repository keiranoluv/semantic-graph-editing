"""Frozen candidate vectors and lexical features from inference inputs only."""
import torch

from src.editing.operations import EditGraph, Node
from tools.edit_model_v1.data import feature_texts
from tools.magicbrush_experiment_v3.vocabulary import PrototypeInput
from tools.magicbrush_lexical_pilot_v1.features import GroundingStore
from tools.magicbrush_lexical_pilot_v1.features import collate_inputs as base_collate
from tools.magicbrush_lexical_pilot_v1.features import lexical

from .candidates import candidates


def candidate_text(label):
    return feature_texts(EditGraph((Node("candidate", label),)), "")["nodes"][0]


def cache_inputs(examples):
    # Source-only synthetic nodes encode instruction spans, not target vocabulary.
    names = sorted({s for e in examples for s in candidates(e.instruction)})
    return [
        PrototypeInput("instruction-span:" + str(i), EditGraph((Node("candidate", s),))) for i, s in enumerate(names)
    ]


def collate_inputs(items, store, *, device="cpu"):
    batch = base_collate(items, store, device="cpu")
    names = [candidates(item["instruction"]) for item in items]
    size = max(1, max(map(len, names)))
    vectors = torch.zeros(len(items), size, store.dim)
    mask = torch.zeros(len(items), size, dtype=torch.bool)
    matches = torch.full((len(items), size), -1, dtype=torch.long)
    features = torch.zeros(len(items), size, 6)
    labels = store.vocabulary["labels"]
    for b, (item, spans) in enumerate(zip(items, names, strict=True)):
        graph = item["source"]
        payloads = tuple((n.label, n.attributes) for n in graph.nodes)
        lf, _, _, _ = lexical(item["instruction"], payloads, (), spans)
        for i, value in enumerate(spans):
            vectors[b, i] = store[candidate_text(value)]
            mask[b, i] = True
            if value in labels:
                matches[b, i] = labels.index(value) + 1
            features[b, i] = torch.tensor(lf[i])
    batch.update(copy_vectors=vectors, copy_mask=mask, copy_matches=matches, copy_grounding=features)
    return {k: v.to(device) for k, v in batch.items()}


__all__ = ["GroundingStore", "collate_inputs", "cache_inputs", "candidate_text"]
