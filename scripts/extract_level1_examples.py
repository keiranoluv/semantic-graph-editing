"""Find review candidates: five R@50 hits and five missed GT triplets."""

import argparse
import json
import random
from pathlib import Path

import torch

from scripts.evaluate_level1 import load_model
from scripts.pilot_mlp import resolve_device
from scripts.train_mlp import make_dataset
from src.features.visual import load_visual_manifest
from src.knowledge.transe.core import sha256


def predict_graph(model, name, graph, device):
    kwargs = {
        "x": graph.x.to(device),
        "edge_label_index": graph.edge_label_index.to(device),
    }
    if name in {"gat", "gat_kg", "gat_pair_kg"}:
        kwargs["edge_index"] = graph.edge_index.to(device)
        kwargs["object_labels"] = graph.object_labels.to(device)
    elif name == "mlp_semantic":
        kwargs["object_labels"] = graph.object_labels.to(device)
    return model(**kwargs).cpu()


def examples_from_graph(graph, logits, object_names, predicate_names, image_url):
    """Use exactly evaluator ranking: max foreground probability per ordered pair."""
    scores, classes = logits.softmax(dim=-1)[:, 1:].max(dim=-1)
    classes = classes + 1
    ranking = torch.argsort(scores, descending=True, stable=True).tolist()
    candidates = graph.edge_label_index
    pair_info = {}
    for rank, i in enumerate(ranking, 1):
        pair = (int(candidates[0, i]), int(candidates[1, i]))
        pair_info[pair] = (int(classes[i]), rank, float(scores[i]))
    chosen = {(int(candidates[0, i]), int(classes[i]), int(candidates[1, i])) for i in ranking[:50]}
    gt = {
        (int(graph.gt_relation_index[0, i]), int(graph.gt_predicate[i]), int(graph.gt_relation_index[1, i]))
        for i in range(graph.gt_predicate.numel())
    }
    groups = {"hit_at_50": [], "missed_at_50": []}
    for source, predicate, target in sorted(gt):
        predicted, rank, score = pair_info[(source, target)]
        kind = "hit_at_50" if (source, predicate, target) in chosen else "missed_at_50"
        groups[kind].append(
            {
                "image_id": str(graph.image_id),
                "image_url": image_url,
                "subject_node": source,
                "subject_object_id": int(graph.object_ids[source]),
                "subject_class": object_names[int(graph.object_labels[source])],
                "subject_box_xyxy": graph.boxes[source].tolist(),
                "object_node": target,
                "object_object_id": int(graph.object_ids[target]),
                "object_class": object_names[int(graph.object_labels[target])],
                "object_box_xyxy": graph.boxes[target].tolist(),
                "gt_predicate": predicate_names[predicate],
                "predicted_predicate": predicate_names[predicted],
                "predicted_rank": rank,
                "predicted_pair_score": score,
                "reason": "correct_predicate_in_top50"
                if kind == "hit_at_50"
                else "different_predicate"
                if predicted != predicate
                else "correct_predicate_below_top50",
                "human_analysis": "",
            }
        )
    return groups


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    p.add_argument("--visual-cache", type=Path, help="Override cache path stored in a visual checkpoint")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--max-images", type=int)
    p.add_argument("--selection-seed", type=int, default=2026)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        p.error("Held-out test requires --allow-test after protocol is frozen")
    if args.count <= 0 or (args.max_images is not None and args.max_images <= 0):
        p.error("Count and optional image limit must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    data_hashes = {
        name: sha256(args.root / name)
        for name in ["VG-SGG-with-attri.h5", "VG-SGG-dicts-with-attri.json", "image_data.json", "splits.json"]
    }
    if config.get("data_sha256") is not None and config["data_sha256"] != data_hashes:
        raise ValueError("Dataset differs from checkpoint input hashes")
    visual_cache = args.visual_cache or config.get("visual_cache")
    if config.get("visual_cache_manifest_sha256"):
        if visual_cache is None:
            raise ValueError("Visual checkpoint requires its frozen region cache")
        visual_cache = Path(visual_cache)
        load_visual_manifest(visual_cache, args.root, config["box_policy"])
        if sha256(visual_cache / "manifest.json") != config["visual_cache_manifest_sha256"]:
            raise ValueError("Visual cache differs from training checkpoint")
    elif args.visual_cache is not None:
        raise ValueError("Cannot add visual features to an annotation-only checkpoint")
    device = resolve_device(args.device)
    model = load_model(checkpoint).to(device).eval()
    dataset = make_dataset(args.root, args.split, config["box_policy"], visual_cache)
    limit = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
    metadata = json.loads((args.root / "image_data.json").read_text(encoding="utf-8"))
    image_urls = {str(item["image_id"]): item.get("url") for item in metadata}
    vocabulary = json.loads((args.root / "VG-SGG-dicts-with-attri.json").read_text(encoding="utf-8"))
    object_names = {v: k for k, v in vocabulary["label_to_idx"].items()}
    predicate_names = {v: k for k, v in vocabulary["predicate_to_idx"].items()}
    indices = list(range(limit))
    random.Random(args.selection_seed).shuffle(indices)
    selected = {"hit_at_50": [], "missed_at_50": []}
    for index in indices:
        graph = dataset[index]
        logits = predict_graph(model, config["model"], graph, device)
        groups = examples_from_graph(graph, logits, object_names, predicate_names, image_urls.get(str(graph.image_id)))
        for kind in selected:
            if len(selected[kind]) < args.count and groups[kind]:
                selected[kind].append(groups[kind][0])  # At most one per image and group.
        if all(len(selected[kind]) == args.count for kind in selected):
            break
    if any(len(selected[kind]) < args.count for kind in selected):
        raise ValueError("Could not find enough hit/missed examples; widen the image limit")
    report = {
        "schema_version": 1,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "visual_cache_manifest_sha256": config.get("visual_cache_manifest_sha256"),
        "model": config["model"],
        "selection_seed": args.selection_seed,
        "image_pool_size": limit,
        "selection_rule": "shuffle_images_seeded_take_first_gt_triplet_per_category_at_most_one_per_image",
        "protocol": "PredCls_graph_constrained_top1_per_pair_R@50",
        "examples": selected,
        "warning": "Missed means not matched to annotated GT at top50; human_analysis must be completed by a person",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[CANDIDATES] {args.count} hit and {args.count} missed GT triplets saved to {args.output}")


if __name__ == "__main__":
    main()
