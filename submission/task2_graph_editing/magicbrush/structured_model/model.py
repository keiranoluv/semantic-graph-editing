"""Matched MLP/GNN encoders and categorical, copy-aware graph-slot heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from task2_graph_editing.magicbrush.sequence_model.model import MessageLayer


class SemanticHead(nn.Module):
    def __init__(self, hidden, vectors):
        super().__init__()
        self.register_buffer("prototypes", F.normalize(vectors.detach().clone().float(), dim=-1))
        self.query = nn.Linear(hidden, vectors.shape[-1])
        self.copy = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.zeros(len(vectors)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, values):
        query = F.normalize(self.query(values), dim=-1)
        scores = self.log_scale.exp().clamp(max=100) * query @ self.prototypes.T + self.bias
        return torch.cat((self.copy(values), scores), -1)


class StructuredPredictor(nn.Module):
    def __init__(
        self,
        *,
        kind,
        vocabulary,
        prototypes,
        text_dim=512,
        hidden=192,
        max_nodes=20,
        max_edges=60,
        layers=2,
        add_slots=12,
        dropout=0.1,
        max_attributes=32,
    ):
        super().__init__()
        if kind not in ("mlp", "gnn"):
            raise ValueError("Expected mlp or gnn")
        self.vocabulary, self.max_nodes, self.add_slots = vocabulary, max_nodes, add_slots
        self.project = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.node_identity = nn.Embedding(max_nodes, hidden)
        self.edge_identity = nn.Embedding(max_edges, hidden)
        self.new_nodes = nn.Embedding(add_slots, hidden)
        self.new_edges = nn.Embedding(add_slots, hidden)
        self.layers = nn.ModuleList(MessageLayer(hidden) for _ in range(layers if kind == "gnn" else 0))
        self.edge_project = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.fuse = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.node_action, self.edge_action = nn.Linear(hidden, 3), nn.Linear(hidden, 3)
        self.node_present, self.edge_present = nn.Linear(hidden, 2), nn.Linear(hidden, 2)
        self.label = SemanticHead(hidden, prototypes["labels"])
        self.attributes = SemanticHead(hidden, prototypes["attributes"])
        self.remove_attributes = nn.Linear(hidden, max_attributes)
        self.attribute_count = nn.Linear(hidden, max_attributes + 1)
        self.predicate = SemanticHead(hidden, prototypes["predicates"])
        self.subject_query, self.object_query = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.subject_copy, self.object_copy = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def forward(self, batch):
        instruction = self.project(batch["instruction"])
        nodes = (self.project(batch["nodes"]) + self.node_identity.weight) * batch["node_mask"].unsqueeze(-1)
        predicates = self.project(batch["edges"]) + self.edge_identity.weight
        for layer in self.layers:
            nodes = layer(nodes, predicates, batch)
        pooled = nodes.sum(1) / batch["node_mask"].sum(1, keepdim=True).clamp_min(1)

        def fuse(values):
            return self.fuse(
                torch.cat((values, instruction[:, None].expand_as(values), pooled[:, None].expand_as(values)), -1)
            )

        nodes = fuse(nodes)
        endpoint_shape = (-1, -1, nodes.shape[-1])
        edges = self.edge_project(
            torch.cat(
                (
                    nodes.gather(1, batch["subject"].unsqueeze(-1).expand(*endpoint_shape)),
                    predicates,
                    nodes.gather(1, batch["object"].unsqueeze(-1).expand(*endpoint_shape)),
                ),
                -1,
            )
        )
        edges = fuse(edges)
        new_nodes = fuse(self.new_nodes.weight[None].expand(len(nodes), -1, -1))
        new_edges = fuse(self.new_edges.weight[None].expand(len(nodes), -1, -1))
        candidates = torch.cat((nodes, new_nodes), 1)
        candidate_mask = torch.cat(
            (batch["node_mask"], torch.ones(len(nodes), self.add_slots, dtype=torch.bool, device=nodes.device)), 1
        )
        result = {}
        for prefix, values in (("node", nodes), ("new_node", new_nodes), ("edge", edges), ("new_edge", new_edges)):
            is_node, added = prefix.endswith("node"), prefix.startswith("new_")
            if added:
                result[prefix + "_present"] = (self.node_present if is_node else self.edge_present)(values)
            else:
                result[prefix + "_action"] = (self.node_action if is_node else self.edge_action)(values)
            heads = (("label", self.label),) if is_node else (("predicate", self.predicate),)
            for field, head in heads:
                scores = head(values)
                if added:
                    scores = scores.clone()
                    scores[..., 0] = -1e4  # A new entity has no source payload to copy.
                result[prefix + "_" + field] = scores
            if is_node:
                result[prefix + "_remove_attributes"] = self.remove_attributes(values)
                result[prefix + "_add_attributes"] = self.attributes(values)[..., 1:]
                result[prefix + "_attribute_count"] = self.attribute_count(values)
            if not is_node:
                for field, query, copy in (
                    ("subject", self.subject_query, self.subject_copy),
                    ("object", self.object_query, self.object_copy),
                ):
                    scores = torch.bmm(query(values), candidates.transpose(1, 2)) / math.sqrt(values.shape[-1])
                    scores = scores.masked_fill(~candidate_mask[:, None], -1e4)
                    copy_scores = copy(values) if not added else values.new_full((*values.shape[:2], 1), -1e4)
                    result[prefix + "_" + field] = torch.cat((copy_scores, scores), -1)
        return result
