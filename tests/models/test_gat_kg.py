import pytest
import torch

from src.models.gat_kg import KnowledgeGAT, model_from_checkpoint


def config(**changes):
    return {
        "model": "gat_kg",
        "num_object_classes": 151,
        "num_classes": 51,
        "kg_dim": 8,
        "embedding_dim": 4,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "use_knowledge": True,
        "message_edges": "full",
        **changes,
    }


def inputs():
    return dict(
        x=torch.rand(3, 5),
        object_labels=torch.tensor([1, 2, 150]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_label_index=torch.tensor([[0, 1], [1, 2]]),
    )


def test_frozen_kg_and_portable_checkpoint():
    c = config()
    m = KnowledgeGAT(c, torch.randn(151, 8), torch.ones(151, dtype=torch.bool))
    data = inputs()
    y = m(**data)
    assert y.shape == (2, 51)
    y.sum().backward()
    assert not m.kg_features.requires_grad and m.kg_features.grad is None
    assert m.object_embedding.weight.grad is not None
    restored = model_from_checkpoint({"checkpoint_version": 4, "config": c, "model": m.state_dict()})
    assert torch.equal(y, restored(**data))


def test_no_knowledge_ignores_features_and_mask():
    m = KnowledgeGAT(config(use_knowledge=False), torch.randn(151, 8), torch.ones(151, dtype=torch.bool))
    data = inputs()
    first = m(**data)
    m.kg_features.fill_(100)
    m.kg_valid.fill_(False)
    assert torch.equal(first, m(**data))


def test_edge_ablation_keeps_candidate_pairs():
    m = KnowledgeGAT(config(message_edges="none"), torch.randn(151, 8), torch.ones(151, dtype=torch.bool))
    data = inputs()
    first = m(**data)
    data["edge_index"] = torch.tensor([[0, 2], [2, 0]])
    assert torch.equal(first, m(**data))
    assert first.shape[0] == data["edge_label_index"].shape[1]
    data["object_labels"][0] = 151
    with pytest.raises(ValueError, match="vocabulary"):
        m(**data)
