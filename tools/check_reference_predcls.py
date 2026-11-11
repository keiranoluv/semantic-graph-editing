"""Differential check against pinned upstream classes; downloads source to a temp dir.

Run: myenv/bin/python -m tools.check_reference_predcls
Only selected definitions are loaded; the detector framework is not installed.
"""

import ast
import argparse
import json
import tempfile
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from src.evaluation.reference_predcls import REFERENCE_COMMIT, evaluate_reference_image, summarize_reference
from src.knowledge.transe.core import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Also compare first 16 real validation images on CPU")
    args = parser.parse_args()
    torch.set_num_threads(1)
    scope = {"np": np, "torch": torch, "ABC": ABC, "abstractmethod": abstractmethod, "reduce": reduce}
    sources = [
        ("structures/bounding_box.py", {"BoxList"}),
        ("structures/boxlist_ops.py", {"boxlist_iou"}),
        ("utils/miscellaneous.py", {"intersect_2d", "bbox_overlaps", "argsort_desc"}),
        (
            "data/datasets/evaluation/vg/sgg_eval.py",
            {
                "SceneGraphEvaluation",
                "SGRecall",
                "SGMeanRecall",
                "SGNoGraphConstraintRecall",
                "SGNGMeanRecall",
                "_triplet",
                "_compute_pred_matches",
            },
        ),
    ]
    hashes = {}
    with tempfile.TemporaryDirectory(prefix="reference-predcls-") as temp:
        for path, names in sources:
            url = f"https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/{REFERENCE_COMMIT}/maskrcnn_benchmark/{path}"
            with urlopen(url, timeout=30) as response:
                source = response.read()
            local = Path(temp) / Path(path).name
            local.write_bytes(source)
            hashes[path] = sha256(local)
            nodes = [
                node
                for node in ast.parse(source).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
            ]
            assert len(nodes) == len(names)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), url, "exec"), scope)
    store = {}
    recall = scope["SGRecall"](store)
    recall.register_container("predcls")
    mean = scope["SGMeanRecall"](store, 51, ["background"] + [str(i) for i in range(1, 51)])
    mean.register_container("predcls")
    ng_recall = scope["SGNoGraphConstraintRecall"](store)
    ng_recall.register_container("predcls")
    ng_mean = scope["SGNGMeanRecall"](store, 51, ["background"] + [str(i) for i in range(1, 51)])
    ng_mean.register_container("predcls")
    results = []
    ng_results = []
    cases = []
    for seed in range(100):
        torch.manual_seed(seed)
        n = 3 + seed % 10
        pairs = torch.tensor([(s, o) for s in range(n) for o in range(n) if s != o]).T
        xy = torch.randint(0, 10, (n, 2)).float()
        boxes = torch.cat((xy, xy + torch.randint(1, 15, (n, 2))), dim=1)
        boxes[1] = boxes[0]  # Overlapping, different node IDs.
        classes = torch.randint(1, 4, (n,))
        classes[1] = classes[0]
        logits = torch.randn(pairs.shape[1], 51)
        if seed % 10 == 0:
            logits.zero_()  # Deterministic same-device torch.sort tie behavior.
        chosen = torch.randint(pairs.shape[1], (12,))
        gt_pairs = pairs[:, chosen]
        labels = logits.argmax(-1).clamp_min(1)[chosen]
        gt_pairs = torch.cat((gt_pairs, gt_pairs[:, :2]), dim=1)
        labels = torch.cat((labels, labels[:2]))
        cases.append((logits, pairs, gt_pairs, labels, boxes, classes))
    real_count = 0
    if args.checkpoint:
        from scripts.evaluate_level1 import load_model
        from scripts.extract_level1_examples import predict_graph
        from scripts.train_mlp import make_dataset

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        model = load_model(checkpoint).eval()
        dataset = make_dataset(Path("data/raw/vg150"), "val", config["box_policy"], config.get("visual_cache"))
        with torch.inference_mode():
            for i in range(min(16, len(dataset))):
                g = dataset[i]
                logits = predict_graph(model, config["model"], g, torch.device("cpu"))
                cases.append(
                    (logits, g.edge_label_index, g.gt_relation_index, g.gt_predicate, g.boxes, g.object_labels)
                )
                real_count += 1
    for case_id, (logits, pairs, gt_pairs, labels, boxes, classes) in enumerate(cases):
        n = len(boxes)
        scores = logits.softmax(-1)
        ranking = scores[:, 1:].max(-1).values.sort(descending=True).indices
        local = {
            "pred_rel_inds": pairs[:, ranking].T.numpy(),
            "rel_scores": scores[ranking].numpy(),
            "gt_rels": torch.cat((gt_pairs.T, labels[:, None]), dim=1).numpy(),
            "gt_classes": classes.numpy(),
            "pred_classes": classes.numpy(),
            "gt_boxes": boxes.numpy(),
            "pred_boxes": boxes.numpy(),
            "obj_scores": np.ones(n),
        }
        local = recall.calculate_recall({"iou_thres": 0.5}, local, "predcls")
        mean.collect_mean_recall_items({}, local, "predcls")
        result = evaluate_reference_image(logits, pairs, gt_pairs, labels, boxes, classes)
        for k in (50, 100):
            expected = set().union(*map(set, local["pred_to_gt"][:k]))
            assert result["hits"][k] == len(expected), (case_id, k)
            assert result["hits"][k] / result["support"] == store["predcls_recall"][k][-1]
        results.append(result)
        local = ng_recall.calculate_recall({"iou_thres": 0.5}, local, "predcls")
        ng_mean.collect_mean_recall_items({}, local, "predcls")
        ng_result = evaluate_reference_image(logits, pairs, gt_pairs, labels, boxes, classes, graph_constraint=False)
        for k in (50, 100):
            expected = set().union(*map(set, local["nogc_pred_to_gt"][:k]))
            assert ng_result["hits"][k] == len(expected), ("ng", case_id, k)
            assert ng_result["hits"][k] / ng_result["support"] == store["predcls_recall_nogc"][k][-1]
        ng_results.append(ng_result)
    mean.calculate_mean_recall("predcls")
    summary = summarize_reference(results)
    for k in (50, 100):
        assert np.isclose(summary[f"R@{k}"], np.mean(store["predcls_recall"][k]), rtol=0, atol=1e-12)
        assert np.isclose(summary[f"mR@{k}"], store["predcls_mean_recall"][k], rtol=0, atol=1e-12)
        for c in range(1, 51):
            assert np.isclose(
                summary["per_predicate"][str(c)][f"R@{k}"],
                store["predcls_mean_recall_list"][k][c - 1],
                rtol=0,
                atol=1e-12,
            )
    ng_mean.calculate_mean_recall("predcls")
    ng_summary = summarize_reference(ng_results, graph_constraint=False)
    for k in (50, 100):
        assert np.isclose(ng_summary[f"ng-R@{k}"], np.mean(store["predcls_recall_nogc"][k]), rtol=0, atol=1e-12)
        assert np.isclose(ng_summary[f"ng-mR@{k}"], store["predcls_ng_mean_recall"][k], rtol=0, atol=1e-12)
        for c in range(1, 51):
            assert np.isclose(
                ng_summary["per_predicate"][str(c)][f"ng-R@{k}"],
                store["predcls_ng_mean_recall_list"][k][c - 1],
                rtol=0,
                atol=1e-12,
            )
    print(
        json.dumps(
            {
                "status": "PASS",
                "synthetic_images": 100,
                "real_validation_images": real_count,
                "real_checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
                "interpretation": "differential_test_metrics_on_combined_cases_not_model_benchmark",
                "reference_commit": REFERENCE_COMMIT,
                "source_sha256": hashes,
                "metrics": {k: v for k, v in summary.items() if k != "per_predicate"},
                "no_graph_constraint_metrics": {k: v for k, v in ng_summary.items() if k != "per_predicate"},
                "numpy_version": np.__version__,
                "torch_version": torch.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
