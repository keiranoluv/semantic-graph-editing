"""MLP/GNN source encoders with the same attentive GRU byte-action decoder."""

import math

import torch
from torch import nn

from tools.edit_model_v1.codec import BOS, EOS, PAD, VOCAB_SIZE
from tools.edit_model_v1.data import feature_texts


def collate_inputs(inputs, embeddings, *, max_nodes=20, max_edges=60, device="cpu"):
    batch, dim = len(inputs), embeddings.dim
    tensors = {
        "instruction": torch.zeros(batch, dim),
        "nodes": torch.zeros(batch, max_nodes, dim),
        "edges": torch.zeros(batch, max_edges, dim),
        "node_mask": torch.zeros(batch, max_nodes, dtype=torch.bool),
        "edge_mask": torch.zeros(batch, max_edges, dtype=torch.bool),
        "subject": torch.zeros(batch, max_edges, dtype=torch.long),
        "object": torch.zeros(batch, max_edges, dtype=torch.long),
    }
    for i, item in enumerate(inputs):
        source = item["source"]
        if len(source.nodes) > max_nodes or len(source.edges) > max_edges:
            raise ValueError("Source graph exceeds the predeclared model budget")
        texts = feature_texts(source, item["instruction"])
        tensors["instruction"][i] = embeddings[texts["instruction"]]
        for key, mask in (("nodes", "node_mask"), ("edges", "edge_mask")):
            for j, text in enumerate(texts[key]):
                tensors[key][i, j] = embeddings[text]
                tensors[mask][i, j] = True
        ids = {node.id: j for j, node in enumerate(source.nodes)}
        for j, edge in enumerate(source.edges):
            tensors["subject"][i, j], tensors["object"][i, j] = ids[edge.subject], ids[edge.object]
    return {k: v.to(device) for k, v in tensors.items()}


def collate_targets(examples, device="cpu"):
    from tools.edit_model_v1.codec import encode_actions

    sequences = [encode_actions(example.operations) for example in examples]
    tokens = torch.full((len(sequences), max(map(len, sequences))), PAD, dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        tokens[i, : len(sequence)] = torch.tensor(sequence, device=device)
    return tokens[:, :-1], tokens[:, 1:]


class MessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.forward_message = nn.Linear(2 * hidden, hidden)
        self.reverse_message = nn.Linear(2 * hidden, hidden)
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))

    def forward(self, nodes, edges, batch):
        hidden = nodes.shape[-1]
        subject = batch["subject"].unsqueeze(-1).expand(-1, -1, hidden)
        obj = batch["object"].unsqueeze(-1).expand(-1, -1, hidden)
        mask = batch["edge_mask"].unsqueeze(-1)
        incoming = self.forward_message(torch.cat((nodes.gather(1, subject), edges), -1)) * mask
        outgoing = self.reverse_message(torch.cat((nodes.gather(1, obj), edges), -1)) * mask
        sums = torch.zeros_like(nodes).scatter_add(1, obj, incoming).scatter_add(1, subject, outgoing)
        degree = torch.zeros_like(nodes).scatter_add(1, obj, mask.expand_as(incoming).float())
        degree = degree.scatter_add(1, subject, mask.expand_as(outgoing).float()).clamp_min(1)
        return self.update(torch.cat((nodes, sums / degree), -1)) * batch["node_mask"].unsqueeze(-1)


class EditPredictor(nn.Module):
    def __init__(self, *, kind, text_dim=512, hidden=192, byte_dim=64, max_nodes=20, max_edges=60, layers=2):
        super().__init__()
        if kind not in ("mlp", "gnn"):
            raise ValueError("Expected mlp or gnn")
        self.kind = kind
        self.text_projection = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.node_identity = nn.Embedding(max_nodes, hidden)
        self.edge_identity = nn.Embedding(max_edges, hidden)
        self.layers = nn.ModuleList(MessageLayer(hidden) for _ in range(layers if kind == "gnn" else 0))
        self.edge_projection = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.initial = nn.Linear(2 * hidden, hidden)
        self.byte_embedding = nn.Embedding(VOCAB_SIZE, byte_dim, padding_idx=PAD)
        self.decoder = nn.GRU(byte_dim + hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden * 2, VOCAB_SIZE)

    def encode(self, batch):
        instruction = self.text_projection(batch["instruction"])
        nodes = self.text_projection(batch["nodes"]) + self.node_identity.weight.unsqueeze(0)
        edges = self.text_projection(batch["edges"]) + self.edge_identity.weight.unsqueeze(0)
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

    def logits(self, states, memory, mask):
        attention = torch.bmm(states, memory.transpose(1, 2)) / math.sqrt(states.shape[-1])
        attention = attention.masked_fill(~mask.unsqueeze(1), -1e4).softmax(-1)
        context = torch.bmm(attention, memory)
        logits = self.output(torch.cat((states, context), -1))
        # These are input/padding symbols, never valid generated action bytes.
        logits[..., PAD] = -1e4
        logits[..., BOS] = -1e4
        return logits

    def forward(self, batch, previous_tokens):
        memory, mask, pooled, initial = self.encode(batch)
        embedded = self.byte_embedding(previous_tokens)
        states, _ = self.decoder(
            torch.cat((embedded, pooled.unsqueeze(1).expand(-1, embedded.shape[1], -1)), -1), initial
        )
        return self.logits(states, memory, mask)

    def predict(self, batch, max_tokens):
        memory, mask, pooled, state = self.encode(batch)
        previous = torch.full((len(memory), 1), BOS, dtype=torch.long, device=memory.device)
        result, finished = [[] for _ in range(len(memory))], [False] * len(memory)
        for _ in range(max_tokens):
            embedded = self.byte_embedding(previous)
            states, state = self.decoder(torch.cat((embedded, pooled.unsqueeze(1)), -1), state)
            previous = self.logits(states, memory, mask).argmax(-1)
            for i, token in enumerate(previous[:, 0].tolist()):
                if not finished[i]:
                    result[i].append(token)
                    finished[i] = token == EOS
            if all(finished):
                break
        return result
