"""Frozen Task B encoders/GRU, with train-only lexical JSON action tokens."""


import json

import torch
from common.editing.codec import BOS, EOS, PAD, encode_actions
from common.editing.encoders import EditPredictor
from common.editing.operations import KINDS
from torch import nn


def lexemes(operations):
    data = bytes(t - 3 for t in encode_actions(operations)[1:-1]).decode("utf-8")
    rows = json.loads(data)
    return [json.dumps(field, ensure_ascii=False, separators=(",", ":")) for row in rows for field in row]


class ActionVocabulary:
    def __init__(self, tokens):
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate_vocabulary_token")
        self.tokens = list(tokens)
        self.ids = {t: i + 4 for i, t in enumerate(tokens)}
        self.size = len(tokens) + 4  # PAD, BOS, EOS, UNK

    @classmethod
    def fit(cls, train):
        # All six operation names and bounded identity tokens are schema, not
        # fabricated training labels. Lexical payloads come only from train.
        tokens = {"[]"} | {json.dumps(k) for k in KINDS}
        tokens.update(
            json.dumps(f"{prefix}{i:03d}")
            for prefix, count in (("n", 20), ("e", 60), ("newn", 20), ("newe", 60))
            for i in range(1, count + 1)
        )
        for example in train:
            tokens.update(lexemes(example.operations))
        return cls(sorted(tokens))

    def encode(self, operations):
        return [BOS, *(self.ids.get(t, 3) for t in lexemes(operations)), EOS]

    def to_bytes(self, tokens):
        tokens = list(tokens)
        if not tokens or tokens[-1] != EOS or any(t < 4 or t >= self.size for t in tokens[:-1]):
            return []  # Existing evaluator records a parse failure, never drops it.
        fields = [json.loads(self.tokens[t - 4]) for t in tokens[:-1]]
        rows = []
        while fields:
            kind = fields[0]
            if kind not in KINDS:
                return []
            size = 2 if kind.startswith("delete_") else 4 if kind.endswith("_node") else 5
            if len(fields) < size or len(rows) >= 12:
                return []
            rows.append(fields[:size])
            fields = fields[size:]
        text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        return [*(b + 3 for b in text.encode("utf-8")), EOS]

    def collate(self, examples):
        sequences = [self.encode(e.operations) for e in examples]
        tensor = torch.full((len(sequences), max(map(len, sequences))), PAD, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            tensor[i, : len(sequence)] = torch.tensor(sequence)
        return tensor[:, :-1], tensor[:, 1:]


class LexicalPredictor(EditPredictor):
    def __init__(self, *, vocabulary_size, **kwargs):
        super().__init__(**kwargs)
        hidden = self.decoder.hidden_size
        self.byte_embedding = nn.Embedding(vocabulary_size, 48, padding_idx=PAD)
        # Parent configured with byte_dim=48; all encoders/attention unchanged.
        self.output = nn.Linear(hidden * 2, vocabulary_size)

    def encode(self, batch):
        # Same frozen encoder computation, with padding sized to the current
        # split rather than paying projection cost for 42 absent edges.
        instruction = self.text_projection(batch["instruction"])
        nodes = self.text_projection(batch["nodes"]) + self.node_identity.weight[: batch["nodes"].shape[1]]
        edges = self.text_projection(batch["edges"]) + self.edge_identity.weight[: batch["edges"].shape[1]]
        nodes = nodes * batch["node_mask"].unsqueeze(-1)
        for layer in self.layers:
            nodes = layer(nodes, edges, batch)
        subject = batch["subject"].unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
        obj = batch["object"].unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
        edges = self.edge_projection(torch.cat((nodes.gather(1, subject), edges, nodes.gather(1, obj)), -1))
        memory = torch.cat((instruction.unsqueeze(1), nodes, edges), 1)
        mask = torch.cat(
            (torch.ones(len(nodes), 1, dtype=torch.bool, device=nodes.device), batch["node_mask"], batch["edge_mask"]),
            1,
        )
        pooled = (memory * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        initial = torch.tanh(self.initial(torch.cat((pooled, instruction), -1)))
        return memory, mask, pooled, initial.unsqueeze(0)
