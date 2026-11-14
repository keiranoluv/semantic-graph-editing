"""Generate-or-copy labels with a shared learned semantic query.

Duplicate generate/copy spellings are marginalized into one label class. Training
on in-vocabulary mentions therefore trains the same query and copy bias used for
unseen names at inference. No separate untrained OOV classifier is introduced.
"""
import torch
from torch import nn
from torch.nn import functional as F

from tools.magicbrush_lexical_pilot_v1.model import StructuredPredictor as LexicalPredictor


class StructuredPredictor(LexicalPredictor):
    def __init__(self, *, use_copy=True, **kwargs):
        super().__init__(**kwargs)
        self.use_copy = use_copy
        self.copy_bias = nn.Parameter(torch.zeros(2))
        self._label_inputs = []
        self.label.register_forward_pre_hook(self._capture_label_input)

    def _capture_label_input(self, _module, args):
        self._label_inputs.append(args[0])

    def forward(self, batch):
        self._label_inputs = []
        result = super().forward(batch)
        values = self._label_inputs
        self._label_inputs = []
        if not self.use_copy:
            return result
        if len(values) != 2:
            raise AssertionError("Unexpected label head calls")
        prototypes = F.normalize(batch["copy_vectors"], dim=-1)
        for j, (prefix, hidden) in enumerate(zip(("node", "new_node"), values, strict=True)):
            query = F.normalize(self.label.query(hidden), dim=-1)
            scores = self.label.log_scale.exp().clamp(max=100) * torch.bmm(query, prototypes.transpose(1, 2))
            scores = scores + self.copy_bias[j]
            weights = self.replace_label_weights if prefix == "node" else self.new_label_weights
            bonus = (batch["copy_grounding"] @ weights)[:, None]
            if prefix == "node":
                bonus = bonus * batch["substitution"]
            scores = scores + bonus
            scores = scores.masked_fill(~batch["copy_mask"][:, None], -1e4)
            original = result[prefix + "_label"]
            matched = batch["copy_matches"]
            # Candidate strings are unique, hence positive scatter indices unique.
            extra = torch.full_like(original, -torch.inf).scatter(
                -1,
                matched.clamp_min(0)[:, None].expand(-1, scores.shape[1], -1),
                scores.masked_fill((matched < 1)[:, None], -torch.inf),
            )
            merged = torch.logaddexp(original, extra)
            dynamic = scores.masked_fill((matched >= 1)[:, None], -1e4)
            result[prefix + "_label"] = torch.cat((merged, dynamic), -1)
        return result
