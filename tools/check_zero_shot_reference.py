"""Compare zero-shot scoring against actual pinned upstream evaluators on synthetic and real images."""

import argparse
import ast
import hashlib
import json
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from src.evaluation.reference_predcls import REFERENCE_COMMIT
from src.evaluation.zero_shot import class_triplets, evaluate_zero_shot_image, summarize_zero_shot
from src.knowledge.transe.core import sha256
from tools.analyze_level1_long_tail import encode, write_fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
                "SGNoGraphConstraintRecall",
                "SGZeroShotRecall",
                "SGNGZeroShotRecall",
                "_triplet",
                "_compute_pred_matches",
            },
        ),
    ]
    hashes = {}
    for path, names in sources:
        url = f"https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/{REFERENCE_COMMIT}/maskrcnn_benchmark/{path}"
        with urlopen(url, timeout=30) as response:
            source = response.read()
        hashes[path] = hashlib.sha256(source).hexdigest()
        nodes = [
            n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ]
        if len(nodes) != len(names):
            raise ValueError("Missing upstream definitions")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), url, "exec"), scope)
    store = {}
    metrics = {
        name: scope[cls](store)
        for name, cls in (
            ("GC", "SGRecall"),
            ("NG", "SGNoGraphConstraintRecall"),
            ("zGC", "SGZeroShotRecall"),
            ("zNG", "SGNGZeroShotRecall"),
        )
    }
    for metric in metrics.values():
        metric.register_container("predcls")
    cases = []
    for seed in range(100):
        torch.manual_seed(seed)
        n = 3 + seed % 10
        pairs = torch.tensor([(s, o) for s in range(n) for o in range(n) if s != o]).T
        xy = torch.randint(0, 10, (n, 2)).float()
        boxes = torch.cat((xy, xy + torch.randint(1, 15, (n, 2))), dim=1)
        boxes[1] = boxes[0]
        labels = torch.randint(1, 4, (n,))
        labels[1] = labels[0]
        logits = torch.randn(pairs.shape[1], 51)
        if seed % 10 == 0:
            logits.zero_()
        chosen = torch.randint(pairs.shape[1], (12,))
        gt_pairs = pairs[:, chosen]
        predicates = logits.argmax(-1).clamp_min(1)[chosen]
        gt_pairs = torch.cat((gt_pairs, gt_pairs[:, :2]), dim=1)
        predicates = torch.cat((predicates, predicates[:2]))
        unique = sorted(set(class_triplets(gt_pairs, predicates, labels)))
        seen = set(unique if seed % 10 == 1 else [] if seed % 10 == 2 else unique[::2])
        cases.append((logits, pairs, gt_pairs, predicates, boxes, labels, seen))
    from scripts.evaluate_level1 import load_model
    from scripts.extract_level1_examples import predict_graph
    from scripts.train_mlp import make_dataset

    vocabulary = json.loads(args.vocabulary.read_text())
    real_seen = {tuple(t) for t in vocabulary["triples"]}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["data_sha256"] != vocabulary["data_sha256"]:
        raise ValueError("Vocabulary and checkpoint inputs differ")
    model = load_model(checkpoint).eval()
    dataset = make_dataset(Path("data/raw/vg150"), "val", config["box_policy"], config["visual_cache"])
    with torch.inference_mode():
        for i in range(32):
            g = dataset[i]
            logits = predict_graph(model, config["model"], g, torch.device("cpu"))
            cases.append(
                (logits, g.edge_label_index, g.gt_relation_index, g.gt_predicate, g.boxes, g.object_labels, real_seen)
            )
    results = {"GC": [], "NG": []}
    for case_id, (logits, pairs, gt_pairs, predicates, boxes, labels, seen) in enumerate(cases):
        scores = logits.softmax(-1)
        ranking = scores[:, 1:].max(-1).values.sort(descending=True).indices
        triples = class_triplets(gt_pairs, predicates, labels)
        unseen = set(triples) - seen
        # Upstream expects (subject class, object class, predicate) column order.
        global_container = {
            "iou_thres": 0.5,
            "zeroshot_triplet": np.array([(s, o, p) for s, p, o in sorted(unseen)], dtype=np.int64).reshape(-1, 3),
        }
        local = {
            "pred_rel_inds": pairs[:, ranking].T.numpy(),
            "rel_scores": scores[ranking].numpy(),
            "gt_rels": torch.cat((gt_pairs.T, predicates[:, None]), dim=1).numpy(),
            "gt_classes": labels.numpy(),
            "pred_classes": labels.numpy(),
            "gt_boxes": boxes.numpy(),
            "pred_boxes": boxes.numpy(),
            "obj_scores": np.ones(len(boxes)),
        }
        local = metrics["GC"].calculate_recall(global_container, local, "predcls")
        local = metrics["NG"].calculate_recall(global_container, local, "predcls")
        for mode, key in (("GC", "pred_to_gt"), ("NG", "nogc_pred_to_gt")):
            metric = metrics["z" + mode]
            metric.prepare_zeroshot(global_container, local)
            metric.calculate_recall(global_container, local, "predcls")
            result = evaluate_zero_shot_image(
                logits, pairs, gt_pairs, predicates, boxes, labels, seen, graph_constraint=mode == "GC"
            )
            assert result["support"] == len(metric.zeroshot_idx), case_id
            for k in (50, 100):
                matched = set().union(*map(set, local[key][:k]))
                assert result["hits"][k] == len(matched & set(metric.zeroshot_idx)), (case_id, mode, k)
            results[mode].append(result)
    summaries = {}
    for mode, prefix, container in (("GC", "", "predcls_zeroshot_recall"), ("NG", "ng-", "predcls_ng_zeroshot_recall")):
        summary = summarize_zero_shot(results[mode], graph_constraint=mode == "GC")
        for k in (50, 100):
            assert len(store[container][k]) == summary["images_with_zero_shot_gt"]
            assert np.isclose(np.mean(store[container][k]), summary[f"{prefix}zR@{k}"], atol=1e-12, rtol=0)
        summaries[mode] = summary
    write_fixed(
        args.output,
        encode(
            {
                "status": "PASS",
                "reference_commit": REFERENCE_COMMIT,
                "synthetic_images": 100,
                "real_validation_images": 32,
                "real_images_with_zero_shot_gt": sum(r["support"] > 0 for r in results["GC"][100:]),
                "interpretation": "differential_test_on_mixed_cases_not_model_benchmark",
                "source_sha256": hashes,
                "checkpoint_sha256": sha256(args.checkpoint),
                "train_vocabulary_sha256": sha256(args.vocabulary),
                "checker_sha256": sha256(Path(__file__)),
                "zero_shot_evaluator_sha256": sha256(Path("src/evaluation/zero_shot.py")),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "metrics": summaries,
            }
        ),
    )
    print("PASS: exact zero-shot supports/hits, image-macro zR/ng-zR; 100 synthetic + 32 real images")


if __name__ == "__main__":
    main()
