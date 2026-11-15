import pytest
import torch
import torch.nn.functional as F

from experiments.aligng_adapted_v1.batching import collate, model_inputs
from src.evaluation.gqa_pilot import evaluate_gqa_image, summarize_gqa
from src.evaluation.reference_predcls import evaluate_reference_image, summarize_reference
from src.models.penet import PENetPredicateClassifier

from .conftest import graph, model
from .upstream_reference import SGGLayer


@pytest.mark.parametrize("classes", [51, 311])
@pytest.mark.parametrize("sampled", [True, False])
def test_disabled_feedback_exact_baseline_logits_relations_prototypes(classes, sampled):
    adapted = model(classes, feedback_enabled=False).eval()
    baseline = PENetPredicateClassifier(torch.randn(8, 12), torch.randn(classes, 12), semantic_dim=16, dropout=0).eval()
    adapted.load_baseline(baseline)
    batch = collate(
        [graph(5), graph(0, ()), graph(1, ()), graph(3, (), seed=81)],
        classes,
        torch.Generator().manual_seed(42) if sampled else None,
    )
    expected = baseline(batch["x"], batch["query_pairs"], batch["object_labels"], return_embeddings=True)
    actual = adapted(**model_inputs(batch), return_embeddings=True)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_same_seed_gives_exact_same_baseline_parameter_prefix():
    objects, predicates = torch.randn(8, 12), torch.randn(51, 12)
    from experiments.aligng_adapted_v1.model import AlignGPredicateClassifier

    torch.manual_seed(123)
    base = PENetPredicateClassifier(objects, predicates, semantic_dim=16)
    torch.manual_seed(123)
    adapted = AlignGPredicateClassifier(objects, predicates, semantic_dim=16)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, adapted.state_dict()[key], atol=0, rtol=0)


def test_active_encoder_and_global_classifier_match_independent_reference():
    adapted = model(pair_chunk_size=2, attention_query_chunk_size=3).double().eval()
    base = PENetPredicateClassifier(torch.randn(8, 12), torch.randn(51, 12), semantic_dim=16, dropout=0).double().eval()
    adapted.load_baseline(base)
    batch = collate([graph(4)], 51)
    batch["x"] = batch["x"].double()
    captured = []
    handle = base.norm_rel.register_forward_hook(lambda module, args, out: captured.append(out))
    baseline_out = base(batch["x"], batch["context_pairs"], batch["object_labels"], return_embeddings=True)
    handle.remove()
    torch.testing.assert_close(
        adapted._encode_context(batch["x"], batch["object_labels"], batch["context_pairs"]),
        captured[0],
        atol=1e-12,
        rtol=1e-12,
    )
    reference = SGGLayer(51, 16, 2, 0).double().eval()
    reference.load_state_dict(adapted.feedback.state_dict())
    global_raw = base.W_pred(base.predicate_embedding.weight)
    updated = reference(global_raw, captured[0], [batch["context_pairs"].T])
    projected = base.project_head(updated.relu())
    global_projected = baseline_out[2]
    expected = F.normalize(projected, dim=-1) @ F.normalize(global_projected, dim=-1).T * base.logit_scale.exp()
    actual = adapted(**model_inputs(batch), return_embeddings=True)
    for a, b in zip(actual, (expected, projected, global_projected), strict=True):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    assert not torch.allclose(actual[0], baseline_out[0])


def test_reverse_disabled_is_baseline_and_bad_baseline_load_rejected():
    adapted = model(reverse_feedback_enabled=False).eval()
    batch = collate([graph()], 51)
    expected = PENetPredicateClassifier.forward(adapted, batch["x"], batch["query_pairs"], batch["object_labels"])
    torch.testing.assert_close(adapted(**model_inputs(batch)), expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        adapted.load_baseline(torch.nn.Linear(3, 4))


@pytest.mark.parametrize("classes", [51, 311])
@pytest.mark.parametrize("graph_constraint", [True, False])
def test_full_query_order_wires_into_unchanged_evaluators(classes, graph_constraint):
    net = model(classes).eval()
    graphs = [graph(0, ()), graph(1, ()), graph(3, (1, 1, classes - 1)), graph(4, (2,), seed=49)]
    batch = collate(graphs, classes)
    logits = net(**model_inputs(batch))
    together, separate = [], []
    for i, g in enumerate(graphs):
        start, end = batch["query_ptr"][i : i + 2].tolist()
        single = net(**model_inputs(collate([g], classes)))
        for output, destination in ((logits[start:end], together), (single, separate)):
            if classes == 311:
                destination.append(evaluate_gqa_image(output, g, 310, graph_constraint=graph_constraint))
            else:
                destination.append(
                    evaluate_reference_image(
                        output,
                        g.edge_label_index,
                        g.gt_relation_index,
                        g.gt_predicate,
                        g.boxes,
                        g.object_labels,
                        graph_constraint=graph_constraint,
                    )
                )
    assert together == separate
    assert together[2]["support"] == 3  # duplicate GT rows retained for evaluation
    summary = (summarize_gqa if classes == 311 else summarize_reference)(
        together,
        num_predicates=classes - 1,
        graph_constraint=graph_constraint,
    )
    assert summary["num_predicates_for_mean"] == classes - 1
    assert summary["num_images"] == 4 and summary["num_images_with_gt"] == 2
