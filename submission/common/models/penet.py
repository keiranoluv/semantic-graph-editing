"""PE-Net adapted to frozen CLIP PredCls, not the original detector/union model.

Architecture adapted from VL-Group/PENET@9c9f50777c66647799cb7a17dfb855aa98aecd1e.
Copyright (c) 2018 Facebook. MIT notice: licenses/PENET-MIT.txt.
See docs/LEVEL2_METHOD_DECISION.md for all feature/dimension/evaluator deviations.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def mlp(input_dim, hidden, output):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output))


class PENetPredicateClassifier(nn.Module):
    def __init__(self, object_vectors, predicate_vectors, *, feature_dim=517, semantic_dim=256, dropout=0.2):
        super().__init__()
        if feature_dim <= 0 or semantic_dim < 2 or not 0 <= dropout < 1:
            raise ValueError("Invalid PE-Net dimensions/dropout")
        if object_vectors.ndim != 2 or predicate_vectors.ndim != 2:
            raise ValueError("Expected two embedding matrices")
        if (
            object_vectors.shape[1] != predicate_vectors.shape[1]
            or min(len(object_vectors), len(predicate_vectors)) < 2
        ):
            raise ValueError("Incompatible vocabulary embeddings")
        if not torch.isfinite(object_vectors).all() or not torch.isfinite(predicate_vectors).all():
            raise ValueError("Non-finite initial embeddings")
        self.feature_dim = feature_dim
        self.num_classes = len(predicate_vectors)
        self.object_embedding = nn.Embedding.from_pretrained(object_vectors.detach().float().clone(), freeze=False)
        self.predicate_embedding = nn.Embedding.from_pretrained(
            predicate_vectors.detach().float().clone(), freeze=False
        )
        d, text_dim = semantic_dim, object_vectors.shape[1]
        self.post_emb = nn.Linear(feature_dim, 2 * d)
        self.W_sub, self.W_obj, self.W_pred = [mlp(text_dim, d // 2, d) for _ in range(3)]
        self.gate_sub, self.gate_obj, self.gate_pred = [nn.Linear(2 * d, d) for _ in range(3)]
        self.vis2sem = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.project_head = mlp(d, d, 2 * d)
        self.linear_sub, self.linear_obj, self.linear_rel = [nn.Linear(d, d) for _ in range(3)]
        self.norm_sub, self.norm_obj, self.norm_rel = [nn.LayerNorm(d) for _ in range(3)]
        self.dropout_sub, self.dropout_obj, self.dropout_rel = [nn.Dropout(dropout) for _ in range(3)]
        self.dropout_rep, self.dropout_proto = nn.Dropout(dropout), nn.Dropout(dropout)
        # Deliberate adapter: features of the ordered object pair, NOT a union crop.
        self.pair_proxy = mlp(2 * feature_dim, d, d)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, x, edge_label_index, object_labels, *, return_embeddings=False):
        if x.ndim != 2 or x.shape[1] != self.feature_dim or not torch.isfinite(x).all():
            raise ValueError("Invalid node features")
        if object_labels.dtype != torch.long or object_labels.shape != (len(x),):
            raise ValueError("Invalid object labels")
        if object_labels.numel() and (
            (object_labels < 0).any() or (object_labels >= self.object_embedding.num_embeddings).any()
        ):
            raise ValueError("Object labels outside vocabulary")
        if edge_label_index.dtype != torch.long or edge_label_index.ndim != 2 or edge_label_index.shape[0] != 2:
            raise ValueError("Expected long [2,E] pairs")
        if edge_label_index.numel() and ((edge_label_index < 0).any() or (edge_label_index >= len(x)).any()):
            raise ValueError("Pair index outside graph")
        s, o = edge_label_index
        nodes = self.post_emb(x).reshape(len(x), 2, self.post_emb.out_features // 2)
        words = self.object_embedding(object_labels)
        sub, obj = self.W_sub(words[s]), self.W_obj(words[o])
        visual_sub, visual_obj = self.vis2sem(nodes[s, 1]), self.vis2sem(nodes[o, 0])
        sub = sub + visual_sub * self.gate_sub(torch.cat((sub, visual_sub), -1)).sigmoid()
        obj = obj + visual_obj * self.gate_obj(torch.cat((obj, visual_obj), -1)).sigmoid()
        sub = self.norm_sub(sub + self.dropout_sub(self.linear_sub(sub).relu()))
        obj = self.norm_obj(obj + self.dropout_obj(self.linear_obj(obj).relu()))
        fusion = F.relu(sub + obj) - (sub - obj).square()
        visual_pair = self.vis2sem(self.pair_proxy(torch.cat((x[s], x[o]), -1)))
        relation = fusion - visual_pair * self.gate_pred(torch.cat((fusion, visual_pair), -1)).sigmoid()
        relation = self.norm_rel(relation + self.dropout_rel(self.linear_rel(relation).relu()))
        prototypes = self.W_pred(self.predicate_embedding.weight)
        relation = self.project_head(self.dropout_rep(relation.relu()))
        prototypes = self.project_head(self.dropout_proto(prototypes.relu()))
        logits = F.normalize(relation, dim=-1, eps=1e-8) @ F.normalize(prototypes, dim=-1, eps=1e-8).T
        logits = logits * self.logit_scale.exp()
        return (logits, relation, prototypes) if return_embeddings else logits
