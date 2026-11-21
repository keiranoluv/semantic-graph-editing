"""Candidate name features reused; attribute sequence memory uses input words only."""
import torch

from src.editing.operations import EditGraph
from tools.magicbrush_easy_copy_v1.features import GroundingStore
from tools.magicbrush_easy_copy_v1.features import cache_inputs as name_inputs
from tools.magicbrush_easy_copy_v1.features import collate_inputs as name_collate
from tools.magicbrush_experiment_v3.vocabulary import PrototypeInput

from .sequence import input_words


class Store(GroundingStore):
    def __init__(self, base, vocabulary, tokens):
        super().__init__(base, vocabulary)
        self.tokens = tokens


def cache_inputs(examples, tokens):
    words = sorted({w for e in examples for w in input_words(e.source, e.instruction)} | set(tokens))
    return name_inputs(examples) + [
        PrototypeInput("attribute-token:" + str(i), EditGraph(), w) for i, w in enumerate(words)
    ]


def collate_inputs(items, store, *, device="cpu"):
    batch = name_collate(items, store, device="cpu")
    memories = [input_words(i["source"], i["instruction"]) for i in items]
    size = max(1, max(map(len, memories)))
    vectors = torch.zeros(len(items), size, store.dim)
    mask = torch.zeros(len(items), size, dtype=torch.bool)
    matches = torch.full((len(items), size), -1, dtype=torch.long)
    for b, words in enumerate(memories):
        for j, w in enumerate(words):
            vectors[b, j] = store[w]
            mask[b, j] = True
            if w in store.tokens:
                matches[b, j] = store.tokens.index(w)
    batch.update(attr_memory=vectors, attr_mask=mask, attr_matches=matches)
    return {k: v.to(device) for k, v in batch.items()}
