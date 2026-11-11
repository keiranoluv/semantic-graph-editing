"""GAT with a frozen, relation-specific KG branch for ordered object pairs."""

import torch
from torch import nn

from src.models.gat import GATPredicateClassifier


class PairwiseKnowledgeGAT(nn.Module):
    """Add a gated KG predicate correction to ordinary GAT logits."""

    def __init__(self, config: dict, features: torch.Tensor, valid: torch.Tensor):
        super().__init__()
        if features.shape != (config["num_object_classes"], config["kg_dim"]):
            raise ValueError("KG feature shape differs from config")
        if valid.shape != (config["num_object_classes"],):
            raise ValueError("KG valid mask shape differs from config")
        if config["kg_mode"] not in {"on", "off"}:
            raise ValueError("kg_mode must be on or off")
        if not torch.isfinite(features).all():
            raise ValueError("KG features contain nonfinite values")
        self.base = GATPredicateClassifier(
            geometry_dim=config["geometry_dim"],
            num_object_classes=config["num_object_classes"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        self.kg_mode = config["kg_mode"]
        self.register_buffer("kg_features", features.detach().clone().float())
        self.register_buffer("kg_valid", valid.detach().clone().bool())
        pair_dim = 4 * config["kg_dim"] + 2
        self.predicate_correction = nn.Sequential(
            nn.Linear(pair_dim, config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["num_classes"]),
        )
        self.gate = nn.Linear(pair_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, x, edge_index, edge_label_index, object_labels):
        logits = self.base(x, edge_index, edge_label_index, object_labels)
        if self.kg_mode == "off":
            return logits
        source, target = edge_label_index
        subject = self.kg_features[object_labels[source]]
        object_ = self.kg_features[object_labels[target]]
        subject_valid = self.kg_valid[object_labels[source]].to(x.dtype).unsqueeze(-1)
        object_valid = self.kg_valid[object_labels[target]].to(x.dtype).unsqueeze(-1)
        pair = torch.cat(
            (subject, object_, subject * object_, torch.abs(subject - object_), subject_valid, object_valid), dim=-1
        )
        gate = torch.sigmoid(self.gate(pair)) * subject_valid * object_valid
        return logits + gate * self.predicate_correction(pair)


def model_from_checkpoint(checkpoint):
    config = checkpoint["config"]
    if checkpoint.get("checkpoint_version") != 5 or config.get("model") != "gat_pair_kg":
        raise ValueError("Expected version 5 pairwise KG checkpoint")
    state = checkpoint["model"]
    model = PairwiseKnowledgeGAT(config, state["kg_features"], state["kg_valid"])
    model.load_state_dict(state)
    return model
