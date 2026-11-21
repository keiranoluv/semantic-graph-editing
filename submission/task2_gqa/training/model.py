"""Vectorized PE-Net encoder and masked, bucketed AlignG feedback.

Same parameter schema/formulas as P0, no per-image GPU validation or sort.
Dropout draws differ from P0; a new experiment identity is mandatory.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from task2_gqa.models.model import AlignGPredicateClassifier
from common.models.penet import PENetPredicateClassifier

from task2_gqa.training.batching import ModelBatch


@dataclass(frozen=True)
class FastConfig:
    encoder_chunk_size: int = 16384
    checkpoint_encoder: bool = True
    checkpoint_attention: bool = False

    def __post_init__(self):
        if type(self.encoder_chunk_size) is not int or self.encoder_chunk_size < 1:
            raise ValueError("Positive encoder chunk required")
        for name in ("checkpoint_encoder", "checkpoint_attention"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


class FastAlignG(AlignGPredicateClassifier):
    def __init__(self, *args, fast_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_config = fast_config or FastConfig()
        if self.config.num_attention_heads != 2:
            raise ValueError("Fast v1 CPU grouping is pinned to two heads")

    def _encode_all(self, x, labels, pairs):
        d = self.post_emb.out_features // 2
        nodes = self.post_emb(x).reshape(len(x), 2, d)
        words = self.object_embedding(labels)
        parts = []
        for indices in pairs.split(self.fast_config.encoder_chunk_size, dim=1):
            if self.fast_config.checkpoint_encoder and self.training and torch.is_grad_enabled():
                part = checkpoint(
                    self._encode_pairs, x, nodes, words, indices, use_reentrant=False, preserve_rng_state=True
                )
            else:
                part = self._encode_pairs(x, nodes, words, indices)
            parts.append(part)
        return torch.cat(parts)

    def _attention(self, q, k, v, mask):
        f = self.feedback
        b = k.shape[0]
        q = q.reshape(b, -1, f.H, f.d).transpose(1, 2)
        k = k.reshape(b, -1, f.H, f.d).transpose(1, 2)
        v = v.reshape(b, -1, f.H, f.d).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / f.sqrt_d
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        weights = f.dropout(scores.softmax(-1))
        return (weights @ v).transpose(1, 2).reshape(b, -1, f.D)

    def attend(self, q, k, v, mask=None):
        if self.fast_config.checkpoint_attention and self.training and torch.is_grad_enabled():
            return checkpoint(self._attention, q, k, v, mask, use_reentrant=False, preserve_rng_state=True)
        return self._attention(q, k, v, mask)

    def forward(self, batch, *, return_embeddings=False):
        if not isinstance(batch, ModelBatch):
            raise TypeError("Use CPU-prepared graphs, collate and model_inputs")
        x = batch.x.detach()
        if not self.config.feedback_enabled or not self.config.reverse_feedback_enabled:
            # Exact historical baseline, including its small number of batch-level checks.
            return PENetPredicateClassifier.forward(
                self, x, batch.query_pairs, batch.object_labels, return_embeddings=return_embeddings
            )
        prototypes = self.W_pred(self.predicate_embedding.weight)
        f = self.feedback
        if not batch.query_pairs.shape[1]:
            relation = prototypes.new_empty((0, f.D))
        else:
            encoded = self._encode_all(x, batch.object_labels, batch.context_pairs)
            queries = encoded[batch.query_context]
            normalized_p = f.proto_norm(prototypes)
            q_proto = f.proto_q(normalized_p)
            k_context, v_context = f.edge_k(encoded), f.edge_v(encoded)
            adapted_groups = []
            # ALL prototype updates precede query-dependent reverse attention.
            # Group membership uses only context lengths, so P_i cannot depend on sampling.
            for group in batch.groups:
                b = group.context_index.shape[0]
                normalized = normalized_p.unsqueeze(0).expand(b, -1, -1)
                if self.config.prototype_update_enabled:
                    updates = self.attend(
                        q_proto.unsqueeze(0).expand(b, -1, -1),
                        k_context[group.context_index],
                        v_context[group.context_index],
                        group.context_mask,
                    )
                    updates = f.update_norm_p(updates)
                    adapted = f.proto_gru(updates.reshape(-1, f.D), normalized.reshape(-1, f.D)).reshape(b, -1, f.D)
                else:
                    adapted = normalized
                adapted_groups.append(adapted)
            normalized_q = f.edge_norm(queries)
            q_edges = f.edge_q(normalized_q)
            values, order = [], []
            for group, adapted in zip(batch.groups, adapted_groups, strict=True):
                if not group.query_rows.numel():
                    continue
                feedback = self.attend(q_edges[group.query_index], f.proto_k(adapted), f.proto_v(adapted))
                feedback = feedback.reshape(-1, f.D)[group.query_slots]
                values.append(f.edge_ffn(torch.cat((normalized_q[group.query_rows], feedback), -1)))
                order.append(group.query_rows)
            relation = queries.new_zeros(queries.shape).index_copy(0, torch.cat(order), torch.cat(values))
        relation = self.project_head(self.dropout_rep(relation.relu()))
        prototypes = self.project_head(self.dropout_proto(prototypes.relu()))
        logits = F.normalize(relation, dim=-1, eps=1e-8) @ F.normalize(prototypes, dim=-1, eps=1e-8).T
        logits = logits * self.logit_scale.exp()
        return (logits, relation, prototypes) if return_embeddings else logits
