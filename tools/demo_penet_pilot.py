"""Show PE-Net predictions on a fixed GQA pilot validation image (CLI/JSON)."""

import argparse
import json
from pathlib import Path

import torch

from src.datasets.gqa_pilot import GQAPilotDataset
from src.evaluation.gqa_pilot import evaluate_gqa_image, summarize_gqa
from src.knowledge.transe.core import sha256
from src.models.penet import PENetPredicateClassifier
from src.training.gqa_pilot import collate, forward
from tools.prepare_level2_gqa import write_fixed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/gqa"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    parser.add_argument("--cohort", type=Path, default=Path("reports/level2/gqa-pilot-v1-cohort.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/gqa-pilot-clip-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/level2/penet-pilot-v1/plpr.json"))
    parser.add_argument(
        "--image-index", type=int, default=0, help="Index in pinned validation cohort, not project test"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("top-k must be positive")
    torch.set_num_threads(1)
    report = json.loads(args.report.read_text())
    checkpoint_path = Path(report["checkpoint"])
    if sha256(checkpoint_path) != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config, kind = checkpoint["config"], "penet"
    if config != report["config"] or checkpoint["variant"] != report["variant"]:
        raise ValueError("Checkpoint/report mismatch")
    for key, path in (
        ("manifest_sha256", args.manifest),
        ("cohort_sha256", args.cohort),
        ("cache_manifest_sha256", args.cache / "manifest.json"),
    ):
        if sha256(path) != config[key]:
            raise ValueError(f"Input hash mismatch: {key}")
    dataset = GQAPilotDataset(args.root, args.manifest, args.cohort, args.cache, "val")
    if not 0 <= args.image_index < len(dataset):
        parser.error("image-index outside pinned validation cohort")
    graph = dataset[args.image_index]
    state = checkpoint["model"]
    model = PENetPredicateClassifier(
        state["object_embedding.weight"],
        state["predicate_embedding.weight"],
        semantic_dim=config["semantic_dim"],
        dropout=config["dropout"],
    ).eval()
    model.load_state_dict(checkpoint["model"])
    batch, _ = collate([graph], config["num_classes"])
    with torch.inference_mode():
        logits = forward(model, kind, batch, "cpu")
    scores, labels = logits.softmax(-1)[:, 1:].max(-1)
    labels += 1
    order = scores.sort(descending=True).indices[: args.top_k]
    objects = {i: s for s, i in dataset.vocabulary["label_to_idx"].items()}
    predicates = {i: s for s, i in dataset.vocabulary["predicate_to_idx"].items()}
    gt = set(
        zip(
            graph.gt_relation_index[0].tolist(),
            graph.gt_predicate.tolist(),
            graph.gt_relation_index[1].tolist(),
            strict=True,
        )
    )

    def node(index):
        return {"instance_id": int(graph.object_ids[index]), "label": objects[int(graph.object_labels[index])]}

    predictions = []
    for idx in order.tolist():
        s, o = graph.edge_label_index[:, idx].tolist()
        predicate = int(labels[idx])
        predictions.append(
            {
                "subject": node(s),
                "predicate": predicates[predicate],
                "object": node(o),
                "score": float(scores[idx]),
                "exact_instance_triplet_in_gt": (s, predicate, o) in gt,
            }
        )
    summary = summarize_gqa([evaluate_gqa_image(logits, graph, config["num_classes"] - 1)], config["num_classes"] - 1)
    payload = {
        "synthetic": False,
        "trained_prediction": True,
        "human_reviewed": False,
        "model": kind,
        "variant": report["variant"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "image_id": graph.image_id,
        "selection": f"pinned validation cohort index {args.image_index}",
        "input": "real cached frozen CLIP crops + GT geometry and object labels; no GT relation input",
        "objects": graph.num_nodes,
        "candidate_pairs": len(logits),
        "gt_rows": len(graph.gt_predicate),
        "top_predictions_gc": predictions,
        "image_R@50": summary["R@50"],
        "image_R@100": summary["R@100"],
        "note": "Exact-instance flags are automatic annotation checks, not human review; official pilot metric allows class/IoU matching. This demo predicts relations, not graph-edit instructions.",
    }
    if args.output:
        write_fixed(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
