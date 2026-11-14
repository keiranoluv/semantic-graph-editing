# fmt: off
# isort: skip_file
# Exact SGGLayer excerpt; upstream ba2dd471fb3c3bb39030660dcb7f6f105c52446c.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2026 NamGyu Jung, Chang Choi. See experiments/aligng_adapted_v1/LICENSE-AlignG.txt.
import math

import torch
from torch import nn
from torch.nn import functional as F


class SGGLayer(nn.Module):
    """One block of AlignG's prototype-feedback module.

    Updates per-image predicate prototypes from edge (relation) features and
    then recalibrates edge features with the adapted prototypes.
    """

    def __init__(self, num_rel_cls: int, hidden_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.R, self.D, self.H = num_rel_cls, hidden_dim, num_heads
        self.d = hidden_dim // num_heads
        self.sqrt_d = math.sqrt(self.d)

        self.dropout = nn.Dropout(dropout)

        # Proto update
        self.proto_norm = nn.LayerNorm(self.D)
        self.proto_q = nn.Linear(self.D, self.D)
        self.edge_k = nn.Linear(self.D, self.D)
        self.edge_v = nn.Linear(self.D, self.D)
        self.proto_gru = nn.GRUCell(self.D, self.D)
        self.update_norm_p = nn.LayerNorm(self.D)

        # Edge update
        self.edge_norm = nn.LayerNorm(self.D)
        self.edge_q = nn.Linear(self.D, self.D)
        self.proto_k = nn.Linear(self.D, self.D)
        self.proto_v = nn.Linear(self.D, self.D)
        self.edge_ffn = nn.Linear(self.D * 2, self.D)

    def forward(self, protos, edge_feat, rel_pair_idxs, offset=0):
        edge_outputs = []
        for pair_idx in rel_pair_idxs:
            num_pairs = pair_idx.size(0)
            edge_feat_img = edge_feat[offset: offset + num_pairs]

            # --- Proto update ---
            protos_img_norm = self.proto_norm(protos)
            q_s = self.proto_q(protos_img_norm).view(self.R, self.H, self.d)
            k_e = self.edge_k(edge_feat_img).view(num_pairs, self.H, self.d)
            v_e = self.edge_v(edge_feat_img).view(num_pairs, self.H, self.d)

            q_s_b = q_s.permute(1, 0, 2)  # [H, R, d]
            k_e_b = k_e.permute(1, 0, 2)  # [H, P, d]
            v_e_b = v_e.permute(1, 0, 2)  # [H, P, d]

            attn_logits = torch.bmm(q_s_b, k_e_b.transpose(1, 2)) / self.sqrt_d
            attn = F.softmax(attn_logits, dim=2)
            attn = self.dropout(attn)

            updates = torch.bmm(attn, v_e_b)  # [H, R, d]
            updates = updates.permute(1, 0, 2).contiguous().view(self.R, self.D)
            updates = self.update_norm_p(updates)

            protos_img = self.proto_gru(updates, protos_img_norm)

            # --- Edge update ---
            edges_img_norm = self.edge_norm(edge_feat_img)
            q_e = self.edge_q(edges_img_norm).view(num_pairs, self.H, self.d)
            k_s = self.proto_k(protos_img).view(self.R, self.H, self.d)
            v_s = self.proto_v(protos_img).view(self.R, self.H, self.d)

            q_e_b = q_e.permute(1, 0, 2)  # [H, P, d]
            k_s_b = k_s.permute(1, 0, 2)  # [H, R, d]
            v_s_b = v_s.permute(1, 0, 2)  # [H, R, d]

            attn2_logits = torch.bmm(q_e_b, k_s_b.transpose(1, 2)) / self.sqrt_d
            attn2 = F.softmax(attn2_logits, dim=2)
            attn2 = self.dropout(attn2)

            e_updates = torch.bmm(attn2, v_s_b)  # [H, P, d]
            e_updates = e_updates.permute(1, 0, 2).contiguous().view(num_pairs, self.D)
            edges_img = self.edge_ffn(torch.cat([edges_img_norm, e_updates], dim=-1))

            edge_outputs.append(edges_img)
            offset += num_pairs

        return torch.cat(edge_outputs, dim=0)
