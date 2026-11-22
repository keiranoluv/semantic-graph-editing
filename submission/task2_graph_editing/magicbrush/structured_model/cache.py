"""Frozen input caches plus explicitly train-derived semantic prototypes."""

from task2_graph_editing.magicbrush.sequence_model.data import feature_texts
from task2_graph_editing.magicbrush.sequence_model.text import text_key


def input_texts(examples):
    texts = set()
    for example in examples:
        item = feature_texts(**example.model_input())
        texts.add(item["instruction"])
        texts.update(item["nodes"])
        texts.update(item["edges"])
    return sorted(texts, key=lambda t: (text_key(t), t))
