"""Matched backbone, semantic shuffle, and KG fusion gradients."""

import torch

from src.models.gat_kg_fusion import KnowledgeFusionGAT, model_from_checkpoint


def config(mode="off", variant="none"):
    return {
        "model": "gat_kg_fusion",
        "geometry_dim": 5,
        "num_object_classes": 5,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "num_classes": 4,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "kg_dim": 3,
        "kg_relation_dim": 2,
        "fusion_mode": mode,
        "knowledge_variant": variant,
        "shuffle_seed": 17 if variant == "shuffled" else None,
    }


def data():
    features = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    valid = torch.tensor([False, True, True, True, True])
    scores = torch.arange(50, dtype=torch.float32).reshape(5, 5, 2) / 10
    return features, valid, scores


def graph():
    x = torch.randn(3, 5)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    pairs = torch.tensor([[0, 1, 0], [1, 0, 2]])
    labels = torch.tensor([1, 2, 0])
    return x, edges, pairs, labels


def test_baseline_is_exact_gat_and_conditions_share_initial_weights():
    features, valid, scores = data()
    args = graph()
    models = {}
    for mode, variant in (("off", "none"), ("pair", "real"), ("visual_gate", "real"), ("relation_prior", "real")):
        torch.manual_seed(5)
        models[mode] = KnowledgeFusionGAT(config(mode, variant), features, valid, scores).eval()
    base_weights = models["off"].base.state_dict()
    for mode, model in models.items():
        assert all(torch.equal(value, model.base.state_dict()[key]) for key, value in base_weights.items())
        output = model(*args)
        assert output.shape == (3, 4)
        if mode == "off":
            torch.testing.assert_close(output, model.base(*args), rtol=0, atol=0)
        else:
            # The third pair has an unlinked class: KG must have no effect.
            torch.testing.assert_close(output[2], models["off"](*args)[2], rtol=0, atol=0)
            assert not torch.equal(output[:2], models["off"](*args)[:2])


def test_shuffle_preserves_distribution_and_breaks_every_linked_mapping():
    features, valid, scores = data()
    torch.manual_seed(5)
    shuffled = KnowledgeFusionGAT(config("visual_gate", "shuffled"), features, valid, scores)
    permutation = shuffled.kg_class_permutation
    assert torch.equal(permutation[~valid], torch.where(~valid)[0])
    assert not torch.equal(permutation[valid], torch.where(valid)[0])
    assert bool((permutation[valid] != torch.where(valid)[0]).all())
    torch.testing.assert_close(shuffled.kg_features[valid].sort(dim=0).values, features[valid].sort(dim=0).values)
    torch.testing.assert_close(shuffled.kg_relation_scores, scores[permutation][:, permutation])
    assert torch.equal(shuffled.kg_valid, valid)
    restored = model_from_checkpoint(
        {"checkpoint_version": 6, "config": config("visual_gate", "shuffled"), "model": shuffled.state_dict()}
    )
    assert torch.equal(restored.kg_class_permutation, permutation)
    torch.testing.assert_close(restored.kg_features, shuffled.kg_features)
    torch.testing.assert_close(restored.kg_relation_scores, shuffled.kg_relation_scores)


def test_visual_and_relation_branches_train_and_restore():
    features, valid, scores = data()
    args = graph()
    for mode, last_layer in (("visual_gate", "visual_gate"), ("relation_prior", "relation_correction")):
        torch.manual_seed(6)
        model = KnowledgeFusionGAT(config(mode, "real"), features, valid, scores).eval()
        model(*args).sum().backward()
        branch = getattr(model, last_layer)
        assert branch[-1].weight.grad is not None
        assert branch[-1].weight.grad.abs().sum() > 0
        restored = model_from_checkpoint(
            {"checkpoint_version": 6, "config": config(mode, "real"), "model": model.state_dict()}
        )
        restored.eval()
        torch.testing.assert_close(restored(*args), model(*args), rtol=0, atol=0)
