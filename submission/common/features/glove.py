"""Train-vocabulary GloVe initialization with explicit, deterministic OOV policies."""

import hashlib
from collections import Counter

import numpy as np


def needed_tokens(vocabularies):
    return {token for vocab in vocabularies for label in vocab for token in [label, *label.split()]}


def read_glove_subset(path, tokens, dimension=300):
    found, lines = {}, 0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            lines += 1
            token, _, values = line.rstrip().partition(" ")
            if token not in tokens:
                continue
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.shape != (dimension,) or not np.isfinite(vector).all() or token in found:
                raise ValueError(f"Invalid/duplicate GloVe token: {token}")
            found[token] = vector
    return found, lines


def map_vocabulary(vocabulary, vectors, kind, *, dimension=300, seed=42):
    if kind not in ("object", "predicate"):
        raise ValueError("Unknown vocabulary kind")
    if set(vocabulary.values()) != set(range(1, len(vocabulary) + 1)):
        raise ValueError("Expected contiguous foreground IDs from 1")
    table = np.empty((len(vocabulary) + 1, dimension), dtype=np.float32)
    rows, counts = [], Counter()
    for label, index in [("__background__", 0), *sorted(vocabulary.items(), key=lambda x: x[1])]:
        digest = hashlib.sha256(f"glove-fallback-v1:{seed}:{kind}:{label}".encode()).digest()
        vector = np.random.default_rng(int.from_bytes(digest[:8], "big")).standard_normal(dimension).astype(np.float32)
        used, status = [], "background" if index == 0 else "random_fallback"
        if index:
            if label in vectors:
                used, status = [label], "exact"
            elif kind == "object":
                word = sorted(label.split(), key=len, reverse=True)[0]
                if word in vectors:
                    used, status = [word], "longest_word"
            else:
                used = [word for word in label.split() if word in vectors]
                if used:
                    status = "word_mean"
            if used:
                vector = np.stack([vectors[word] for word in used]).mean(0)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid mapped vector")
        table[index] = vector
        counts[status] += 1
        rows.append({"id": index, "label": label, "status": status, "used_tokens": used})
    return table, {"counts": dict(counts), "rows": rows}
