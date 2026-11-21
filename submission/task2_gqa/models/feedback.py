"""Prototype feedback adapted from AlignG SGGLayer at ba2dd471fb3c3bb39030660dcb7f6f105c52446c.

Copyright (c) 2026 NamGyu Jung, Chang Choi; portions Facebook / VL-Group.
MIT notice: LICENSE-AlignG.txt. No detector dependencies or batch-level state.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PrototypeFeedback(nn.Module):
    """One image, two pair sets; parameter names match the upstream SGGLayer.

    Chunk only attention QUERY rows, never its key/softmax dimension. Optional
    recomputation prevents autograd retaining all attention maps in training.
    No [pairs, classes, hidden] broadcast is constructed.
    """

    def __init__(self, hidden_dim=256, num_heads=2, dropout=0.2, *, query_chunk_size=64, checkpoint_chunks=True):
        super().__init__()
        if hidden_dim < 2 or num_heads < 1 or hidden_dim % num_heads or not 0 <= dropout < 1:
            raise ValueError("Invalid attention dimensions/dropout")
        if type(query_chunk_size) is not int or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")
        self.D, self.H = hidden_dim, num_heads
        self.d = hidden_dim // num_heads
        self.sqrt_d = math.sqrt(self.d)
        self.query_chunk_size, self.checkpoint_chunks = query_chunk_size, checkpoint_chunks
        self.dropout = nn.Dropout(dropout)
        self.proto_norm = nn.LayerNorm(self.D)
        self.proto_q = nn.Linear(self.D, self.D)
        self.edge_k = nn.Linear(self.D, self.D)
        self.edge_v = nn.Linear(self.D, self.D)
        self.proto_gru = nn.GRUCell(self.D, self.D)
        self.update_norm_p = nn.LayerNorm(self.D)
        self.edge_norm = nn.LayerNorm(self.D)
        self.edge_q = nn.Linear(self.D, self.D)
        self.proto_k = nn.Linear(self.D, self.D)
        self.proto_v = nn.Linear(self.D, self.D)
        self.edge_ffn = nn.Linear(2 * self.D, self.D)

    def _attention_block(self, q, k, v):
        weights = (torch.bmm(q, k.transpose(1, 2)) / self.sqrt_d).softmax(dim=-1)
        return torch.bmm(self.dropout(weights), v)

    def attend(self, q, k, v):
        """Projected [rows,D] inputs; each row attends to ALL key rows."""
        if not len(q):
            return q
        if not len(k):
            raise ValueError("Attention requires at least one key")
        q, k, v = [t.reshape(len(t), self.H, self.d).permute(1, 0, 2) for t in (q, k, v)]
        outputs = []
        for part in q.split(self.query_chunk_size, dim=1):
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                out = checkpoint(self._attention_block, part, k, v, use_reentrant=False, preserve_rng_state=True)
            else:
                out = self._attention_block(part, k, v)
            outputs.append(out)
        output = torch.cat(outputs, dim=1)
        return output.permute(1, 0, 2).contiguous().reshape(-1, self.D)

    def forward(self, prototypes, context, queries, *, prototype_update_enabled=True, reverse_feedback_enabled=True):
        for name, tensor in (("prototypes", prototypes), ("context", context), ("queries", queries)):
            if tensor.ndim != 2 or tensor.shape[1] != self.D or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid {name}")
            if tensor.device != prototypes.device or tensor.dtype != prototypes.dtype:
                raise ValueError("Feedback tensors must share device/dtype")
        if len(prototypes) < 2:
            raise ValueError("At least background and one foreground class are required")
        normalized = self.proto_norm(prototypes)
        adapted = normalized
        # Empty-context extension: no invented evidence, no GRU update.
        if prototype_update_enabled and len(context):
            updates = self.attend(self.proto_q(normalized), self.edge_k(context), self.edge_v(context))
            adapted = self.proto_gru(self.update_norm_p(updates), normalized)
        if not reverse_feedback_enabled or not len(queries):
            return queries, adapted
        normalized_queries = self.edge_norm(queries)
        feedback = self.attend(self.edge_q(normalized_queries), self.proto_k(adapted), self.proto_v(adapted))
        return self.edge_ffn(torch.cat((normalized_queries, feedback), dim=-1)), adapted
