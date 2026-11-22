"""PE-Net relation encoder with image-specific AlignG feedback before projection.

Encoder formulas derived from common/models/penet.py (baseline 4e149ce).
Copyright (c) 2018 Facebook; MIT notice: ../../licenses/PENET-MIT.txt.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from common.models.penet import PENetPredicateClassifier
from task2_gqa.models.batching import validate_pairs
from task2_gqa.models.feedback import PrototypeFeedback
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class AlignGConfig:
    feedback_enabled: bool = True
    prototype_update_enabled: bool = True
    reverse_feedback_enabled: bool = True
    prototype_update_mode: str = "gru"
    loss_mode: str = "plpr"
    num_attention_heads: int = 2
    attention_query_chunk_size: int = 64
    pair_chunk_size: int = 1024
    checkpoint_chunks: bool = True

    def __post_init__(self):
        for name in ("feedback_enabled", "prototype_update_enabled", "reverse_feedback_enabled", "checkpoint_chunks"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.prototype_update_mode != "gru":
            raise ValueError("P0 supports only GRU; residual/EMA require a separately versioned implementation")
        if self.loss_mode not in ("ce", "pl", "plpr"):
            raise ValueError("Invalid PE-Net loss mode")
        for name in ("num_attention_heads", "attention_query_chunk_size", "pair_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


class AlignGPredicateClassifier(PENetPredicateClassifier):
    def __init__(
        self, object_vectors, predicate_vectors, *, feature_dim=517, semantic_dim=256, dropout=0.2, config=None
    ):
        # Same initialization order/RNG prefix as baseline; no historical code edited.
        super().__init__(
            object_vectors, predicate_vectors, feature_dim=feature_dim, semantic_dim=semantic_dim, dropout=dropout
        )
        self.config = config if config is not None else AlignGConfig()
        self.feedback = PrototypeFeedback(
            semantic_dim,
            self.config.num_attention_heads,
            dropout,
            query_chunk_size=self.config.attention_query_chunk_size,
            checkpoint_chunks=self.config.checkpoint_chunks,
        )

    def load_baseline(self, baseline):
        """Strictly copy every PE-Net parameter without accepting missing baseline keys."""
        expected = {k for k in self.state_dict() if not k.startswith("feedback.")}
        state = baseline.state_dict()
        if set(state) != expected:
            raise ValueError("Expected a complete original PE-Net state")
        self.load_state_dict({**self.state_dict(), **state}, strict=True)

    def _validate(self, x, object_labels, node_ptr, context_pairs, context_ptr, query_pairs, query_ptr):
        if x.ndim != 2 or x.shape[1] != self.feature_dim or not torch.isfinite(x).all():
            raise ValueError("Invalid node features")
        if object_labels.dtype != torch.long or object_labels.shape != (len(x),):
            raise ValueError("Invalid object labels")
        if object_labels.numel() and (
            (object_labels < 0).any() or (object_labels >= self.object_embedding.num_embeddings).any()
        ):
            raise ValueError("Object labels outside vocabulary")
        for pairs in (context_pairs, query_pairs):
            if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[0] != 2:
                raise ValueError("Expected long [2,E] pairs")
        if any(t.device != x.device for t in (object_labels, context_pairs, query_pairs)):
            raise ValueError("Features, labels and pairs must share a device")
        pointers = []
        for ptr, end in ((node_ptr, len(x)), (context_ptr, context_pairs.shape[1]), (query_ptr, query_pairs.shape[1])):
            if ptr.dtype != torch.long or ptr.ndim != 1 or len(ptr) < 2:
                raise ValueError("Expected long [images+1] pointers")
            values = ptr.tolist()
            if values[0] != 0 or values[-1] != end or any(a > b for a, b in zip(values[:-1], values[1:], strict=True)):
                raise ValueError("Invalid pointer boundaries")
            pointers.append(values)
        if len({len(p) for p in pointers}) != 1:
            raise ValueError("Image counts disagree")
        nptr, cptr, qptr = pointers
        for i in range(len(nptr) - 1):
            n = nptr[i + 1] - nptr[i]
            validate_pairs(context_pairs[:, cptr[i] : cptr[i + 1]] - nptr[i], n, full=True)
            validate_pairs(query_pairs[:, qptr[i] : qptr[i + 1]] - nptr[i], n)
        return pointers

    def _encode_pairs(self, x, nodes, words, pairs):
        # Same operations/order as PE-Net up to its final projection.
        s, o = pairs
        sub, obj = self.W_sub(words[s]), self.W_obj(words[o])
        visual_sub, visual_obj = self.vis2sem(nodes[s, 1]), self.vis2sem(nodes[o, 0])
        sub = sub + visual_sub * self.gate_sub(torch.cat((sub, visual_sub), -1)).sigmoid()
        obj = obj + visual_obj * self.gate_obj(torch.cat((obj, visual_obj), -1)).sigmoid()
        sub = self.norm_sub(sub + self.dropout_sub(self.linear_sub(sub).relu()))
        obj = self.norm_obj(obj + self.dropout_obj(self.linear_obj(obj).relu()))
        fusion = F.relu(sub + obj) - (sub - obj).square()
        visual_pair = self.vis2sem(self.pair_proxy(torch.cat((x[s], x[o]), -1)))
        relation = fusion - visual_pair * self.gate_pred(torch.cat((fusion, visual_pair), -1)).sigmoid()
        return self.norm_rel(relation + self.dropout_rel(self.linear_rel(relation).relu()))

    def _encode_context(self, x, labels, pairs):
        d = self.post_emb.out_features // 2
        nodes = self.post_emb(x).reshape(len(x), 2, d)
        words = self.object_embedding(labels)
        rows = []
        for part in pairs.split(self.config.pair_chunk_size, dim=1):
            if self.config.checkpoint_chunks and self.training and torch.is_grad_enabled():
                rows.append(
                    checkpoint(self._encode_pairs, x, nodes, words, part, use_reentrant=False, preserve_rng_state=True)
                )
            else:
                rows.append(self._encode_pairs(x, nodes, words, part))
        return torch.cat(rows, dim=0)

    def forward(
        self, x, object_labels, node_ptr, context_pairs, context_ptr, query_pairs, query_ptr, *, return_embeddings=False
    ):
        nptr, cptr, qptr = self._validate(
            x, object_labels, node_ptr, context_pairs, context_ptr, query_pairs, query_ptr
        )
        x = x.detach()  # The input cache/CLIP encoder is frozen, even if caller enables input gradients.
        if not self.config.feedback_enabled or not self.config.reverse_feedback_enabled:
            # Updating prototypes alone has no classifier effect: classifier uses GLOBAL P.
            return super().forward(x, query_pairs, object_labels, return_embeddings=return_embeddings)
        prototypes = self.W_pred(self.predicate_embedding.weight)
        updated = []
        for i in range(len(nptr) - 1):
            start, end = nptr[i : i + 2]
            context = context_pairs[:, cptr[i] : cptr[i + 1]] - start
            query = query_pairs[:, qptr[i] : qptr[i + 1]] - start
            if not query.shape[1]:
                updated.append(prototypes.new_empty((0, prototypes.shape[1])))
                continue
            encoded = self._encode_context(x[start:end], object_labels[start:end], context)
            # Reuse context encodings: sampled labels/query order cannot change P_i,
            # including in training with dropout. Do not encode queries a second time.
            keys, permutation = (context[0] * (end - start) + context[1]).sort()
            positions = torch.searchsorted(keys, query[0] * (end - start) + query[1])
            queries = encoded[permutation[positions]]
            relations, _ = self.feedback(
                prototypes,
                encoded,
                queries,
                prototype_update_enabled=self.config.prototype_update_enabled,
            )
            updated.append(relations)
        relation = torch.cat(updated, dim=0)
        relation = self.project_head(self.dropout_rep(relation.relu()))
        # Adapted prototypes recalibrate relations only, exactly as upstream SGGHead.
        prototypes = self.project_head(self.dropout_proto(prototypes.relu()))
        logits = F.normalize(relation, dim=-1, eps=1e-8) @ F.normalize(prototypes, dim=-1, eps=1e-8).T
        logits = logits * self.logit_scale.exp()
        return (logits, relation, prototypes) if return_embeddings else logits
