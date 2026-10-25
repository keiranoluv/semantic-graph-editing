"""Matched KG fusion controls for VG150 PredCls."""

import torch
from torch import nn

from src.models.gat import GATPredicateClassifier


class KnowledgeFusionGAT(nn.Module):
    """One GAT backbone; select an ablated, visual-gated, or relation-score residual."""

    MODES = {"off", "pair", "visual_gate", "relation_prior"}

    def __init__(self, config: dict, features: torch.Tensor, valid: torch.Tensor, relation_scores: torch.Tensor):
        super().__init__()
        classes, kg_dim = config["num_object_classes"], config["kg_dim"]
        if features.shape != (classes, kg_dim) or valid.shape != (classes,):
            raise ValueError("KG feature dimensions differ from config")
        if relation_scores.shape != (classes, classes, config["kg_relation_dim"]):
            raise ValueError("KG relation-score dimensions differ from config")
        if config["fusion_mode"] not in self.MODES:
            raise ValueError("Unsupported KG fusion mode")
        variant = config["knowledge_variant"]
        if variant not in {"none", "real", "shuffled"} or (variant == "none") != (config["fusion_mode"] == "off"):
            raise ValueError("KG variant must be none exactly when fusion is off")
        if not torch.isfinite(features).all() or not torch.isfinite(relation_scores).all():
            raise ValueError("Nonfinite KG features")
        if features.dtype != torch.float32 or relation_scores.dtype != torch.float32 or valid.dtype != torch.bool:
            raise ValueError("KG buffers require float32 features/scores and bool validity")
        if config.get("shuffle_seed") is not None and variant != "shuffled":
            raise ValueError("Shuffle seed is only valid for shuffled KG")
        if variant == "shuffled" and type(config.get("shuffle_seed")) is not int:
            raise ValueError("Shuffled KG requires a fixed integer permutation seed")

        # Construct every branch for every condition. Thus the same model seed
        # gives identical GAT and branch initialization before training.
        self.base = GATPredicateClassifier(
            geometry_dim=config["geometry_dim"],
            num_object_classes=classes,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        self.fusion_mode = config["fusion_mode"]
        permutation = torch.arange(classes)
        if variant == "shuffled":
            linked = torch.where(valid)[0]
            if len(linked) < 2:
                raise ValueError("Shuffled KG requires at least two linked classes")
            generator = torch.Generator().manual_seed(config["shuffle_seed"])
            identity = torch.arange(len(linked))
            order = torch.randperm(len(linked), generator=generator)
            while bool((order == identity).any()):
                order = torch.randperm(len(linked), generator=generator)
            permutation[linked] = linked[order]
        self.register_buffer("kg_class_permutation", permutation)
        self.register_buffer("kg_features", features[permutation].detach().clone())
        self.register_buffer("kg_valid", valid.detach().clone())
        self.register_buffer("kg_relation_scores", relation_scores[permutation][:, permutation].detach().clone())

        pair_dim = 4 * kg_dim + 2
        hidden = config["hidden_dim"]
        self.predicate_correction = nn.Sequential(
            nn.Linear(pair_dim, hidden), nn.ReLU(), nn.Linear(hidden, config["num_classes"])
        )
        self.pair_gate = nn.Linear(pair_dim, 1)
        nn.init.zeros_(self.pair_gate.weight)
        nn.init.constant_(self.pair_gate.bias, -3.0)
        self.visual_gate = nn.Sequential(
            nn.Linear(2 * hidden + pair_dim, hidden), nn.ReLU(), nn.Linear(hidden, config["num_classes"])
        )
        nn.init.zeros_(self.visual_gate[-1].weight)
        nn.init.constant_(self.visual_gate[-1].bias, -3.0)
        self.relation_correction = nn.Sequential(
            nn.Linear(config["kg_relation_dim"] + 2, hidden), nn.ReLU(), nn.Linear(hidden, config["num_classes"])
        )
        self.relation_gate = nn.Parameter(torch.full((config["num_classes"],), -3.0))

    def forward(self, x, edge_index, edge_label_index, object_labels):
        h = self.base.encode_nodes(x, edge_index, object_labels)
        logits = self.base.score_pairs(h, edge_label_index)
        if self.fusion_mode == "off":
            return logits
        source, target = edge_label_index
        subject_ids, object_ids = object_labels[source], object_labels[target]
        subject = self.kg_features[subject_ids]
        object_ = self.kg_features[object_ids]
        subject_valid = self.kg_valid[subject_ids].to(x.dtype).unsqueeze(-1)
        object_valid = self.kg_valid[object_ids].to(x.dtype).unsqueeze(-1)
        pair_valid = subject_valid * object_valid
        if self.fusion_mode == "relation_prior":
            relation_input = torch.cat(
                (self.kg_relation_scores[subject_ids, object_ids], subject_valid, object_valid), dim=-1
            )
            correction = 2.0 * torch.tanh(self.relation_correction(relation_input))
            return logits + pair_valid * torch.sigmoid(self.relation_gate) * correction

        pair_kg = torch.cat(
            (subject, object_, subject * object_, torch.abs(subject - object_), subject_valid, object_valid), dim=-1
        )
        correction = self.predicate_correction(pair_kg)
        if self.fusion_mode == "pair":
            gate = torch.sigmoid(self.pair_gate(pair_kg)) * pair_valid
            return logits + gate * correction
        visual_pair = torch.cat((h[source], h[target], pair_kg), dim=-1)
        gate = torch.sigmoid(self.visual_gate(visual_pair)) * pair_valid
        return logits + gate * (2.0 * torch.tanh(correction))


def model_from_checkpoint(checkpoint):
    config = checkpoint["config"]
    if checkpoint.get("checkpoint_version") != 6 or config.get("model") != "gat_kg_fusion":
        raise ValueError("Expected version 6 KG fusion checkpoint")
    state = checkpoint["model"]
    model = KnowledgeFusionGAT(config, state["kg_features"], state["kg_valid"], state["kg_relation_scores"])
    model.load_state_dict(state)
    return model
