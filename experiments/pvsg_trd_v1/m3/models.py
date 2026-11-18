"""MLP, spatial GAT and alternating spatial/temporal GAT for 57-label TRD.

Residual GAT pattern follows src/models/gat.py; task dimensions/head/topology
are versioned here. No legacy source or weights are changed/imported.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class RelationModel(nn.Module):
    def __init__(self, kind, cfg):
        super().__init__()
        if kind not in ("mlp", "static_gat", "st_gat"):
            raise ValueError("Unknown M3 model")
        self.kind, self.dropout = kind, cfg["dropout"]
        h, heads = cfg["hidden_dim"], cfg["heads"]
        self.embedding = nn.Embedding(126, cfg["class_embedding_dim"])
        self.projection = nn.Sequential(nn.Linear(512 + 15 + cfg["class_embedding_dim"], h), nn.LayerNorm(h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(4 * h + 9, h), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(h, 57))
        self.spatial = nn.ModuleList()
        self.temporal = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        for _ in range(cfg["blocks"]):
            self.spatial.append(
                nn.Linear(h, h)
                if kind == "mlp"
                else GATConv(h, h // heads, heads=heads, dropout=self.dropout, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(h))
            if kind == "st_gat":
                self.temporal.append(
                    GATConv(
                        h,
                        h // heads,
                        heads=heads,
                        dropout=self.dropout,
                        edge_dim=1,
                        fill_value=0.0,
                        add_self_loops=True,
                    )
                )
                self.temporal_norms.append(nn.LayerNorm(h))

    def encode(self, batch):
        h = self.projection(torch.cat((batch["visual"], batch["numeric"], self.embedding(batch["nodes"][:, 2])), dim=1))
        for index, (layer, norm) in enumerate(zip(self.spatial, self.norms, strict=True)):
            message = layer(h) if self.kind == "mlp" else layer(h, batch["spatial_edges"])
            h = F.dropout(F.elu(norm(h + message)), self.dropout, self.training)
            if self.kind == "st_gat":
                delta = batch["temporal_delta_frames"].float().unsqueeze(1)
                message = self.temporal[index](h, batch["temporal_edges"], edge_attr=delta)
                h = F.dropout(F.elu(self.temporal_norms[index](h + message)), self.dropout, self.training)
        return h

    def score(self, encoded, batch, low=0, high=None):
        q = batch["query_edges"][:, low:high]
        s, o = encoded[q[0]], encoded[q[1]]
        return self.head(torch.cat((s, o, s * o, s - o, batch["pair_geometry"][low:high]), dim=1))

    def forward(self, batch):
        return self.score(self.encode(batch), batch)
