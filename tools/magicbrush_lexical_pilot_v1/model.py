"""Soft trainable lexical residuals; unchanged graph encoder and six edit heads."""
import torch
from torch import nn

from tools.magicbrush_experiment_v3.model import StructuredPredictor as BasePredictor

from .features import EDGE_FEATURES, LABEL_FEATURES, NODE_FEATURES


class StructuredPredictor(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parameters allocated without random draws: base initialization matches
        # the seed-42 compact baseline before the new residuals are applied.
        self.new_label_weights = nn.Parameter(torch.tensor([4.0, 0.5, 1.0, 1.0, 1.0, -1.0]))
        self.replace_label_weights = nn.Parameter(torch.tensor([0.0, 0.0, 2.0, 1.0, 2.0, 0.0]))
        self.node_action_weights = nn.Parameter(torch.zeros(len(NODE_FEATURES), 3))
        self.edge_action_weights = nn.Parameter(torch.zeros(len(EDGE_FEATURES), 3))
        with torch.no_grad():
            self.node_action_weights[5, 1] = 2.0  # explicit remove + exact source-name mention
            self.node_action_weights[6, 2] = 2.0  # change cue + exact source-name mention
        assert len(self.new_label_weights) == len(LABEL_FEATURES)
        self.enable_labels = True
        self.enable_ids = True

    def forward(self, batch):
        result = super().forward(batch)
        if self.enable_labels:
            added = (batch["label_grounding"] @ self.new_label_weights)[:, None]
            replaced = (batch["label_grounding"] @ self.replace_label_weights)[:, None] * batch["substitution"]
            # Index zero is the existing source payload COPY symbol.
            result["new_node_label"] = torch.cat(
                (result["new_node_label"][..., :1], result["new_node_label"][..., 1:] + added), -1
            )
            result["node_label"] = torch.cat(
                (result["node_label"][..., :1], result["node_label"][..., 1:] + replaced), -1
            )
        if self.enable_ids:
            result["node_action"] = result["node_action"] + batch["node_grounding"] @ self.node_action_weights
            result["edge_action"] = result["edge_action"] + batch["edge_grounding"] @ self.edge_action_weights
        return result
