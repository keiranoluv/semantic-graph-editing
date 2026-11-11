"""Controls and gradients for the pairwise KG correction."""

import torch

from src.models.gat_pair_kg import PairwiseKnowledgeGAT, model_from_checkpoint


def config(mode):
    return {
        "model": "gat_pair_kg",
        "geometry_dim": 5,
        "num_object_classes": 4,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "num_classes": 3,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "kg_dim": 3,
        "kg_mode": mode,
    }


def inputs():
    x = torch.randn(3, 5)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    pairs = torch.tensor([[0, 1, 0], [1, 2, 2]])
    labels = torch.tensor([1, 2, 3])
    return x, edges, pairs, labels


def test_off_is_exact_base_and_on_trains_pair_branch():
    torch.manual_seed(3)
    features = torch.randn(4, 3)
    valid = torch.tensor([False, True, True, False])
    off = PairwiseKnowledgeGAT(config("off"), features, valid)
    on = PairwiseKnowledgeGAT(config("on"), features, valid)
    on.load_state_dict(off.state_dict())
    args = inputs()
    off.eval()
    on.eval()
    torch.testing.assert_close(off(*args), off.base(*args), rtol=0, atol=0)
    # The third pair contains an unlinked object and must have no correction.
    off_logits, on_logits = off(*args), on(*args)
    torch.testing.assert_close(off_logits[2], on_logits[2], rtol=0, atol=0)
    assert not torch.equal(off_logits[:2], on_logits[:2])
    on(*args).sum().backward()
    assert on.predicate_correction[0].weight.grad is not None
    assert on.predicate_correction[0].weight.grad.abs().sum() > 0
    restored = model_from_checkpoint({"checkpoint_version": 5, "config": config("on"), "model": on.state_dict()})
    restored.eval()
    torch.testing.assert_close(restored(*args), on(*args), rtol=0, atol=0)
