"""Freeze GQA project splits/vocabulary and audit normalized box-based graphs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.datasets.gqa import build_vocabulary, image_archive_index, load_archive_image, make_splits, normalize_gqa
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.knowledge.transe.core import sha256


def write_fixed(path, payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Changed output; use a new path: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/gqa"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/level2/gqa-preparation-v1.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000)
    args = parser.parse_args()
    train_path = args.root / "sceneGraphs/train_sceneGraphs.json"
    test_path = args.root / "sceneGraphs/val_sceneGraphs.json"
    train, test = json.loads(train_path.read_text()), json.loads(test_path.read_text())
    splits = make_splits(train, test, seed=args.seed, val_size=args.val_size)
    vocabulary = build_vocabulary(train, splits["train"])
    manifest = {
        "schema_version": 1,
        "dataset": "GQA_project_full_train_vocabulary_box_PredCls_v1",
        "seed": args.seed,
        "official_split_policy": "official train -> project train/val; official val -> project held-out test",
        "split_order": "SHA256(seed:image_id), internal val first; stored IDs lexicographic",
        "annotation_sha256": {p.name: sha256(p) for p in (train_path, test_path)},
        "splits": splits,
        "vocabulary": vocabulary,
        "normalization": "clip_positive_area_boxes_drop_invalid_OOV_nodes_and_incident_relations; keep_duplicate_relation_rows",
    }
    # Freeze before any validation/test support audit. No held-out labels enter vocab.
    write_fixed(args.manifest, manifest)
    archive = args.root / "archives/images.zip"
    image_index = image_archive_index(archive)
    totals = {}
    for split, ids in splits.items():
        counts = Counter()
        annotations = test if split == "test" else train
        for image_id in ids:
            if image_id not in image_index:
                raise ValueError(f"Missing image: {image_id}")
            _, stats = normalize_gqa(image_id, annotations[image_id], vocabulary)
            counts.update(stats)
        totals[split] = dict(sorted(counts.items()))
        print(split, totals[split], flush=True)
    image_id = splits["train"][0]
    record, _ = normalize_gqa(image_id, train[image_id], vocabulary)
    image = load_archive_image(archive, image_id, image_index)
    if image.size != (record.width, record.height):
        raise ValueError("Smoke image dimensions do not match annotation")
    graph = SceneGraphBuilder().build(record, GeometryFeatureExtractor().extract(record))
    assert graph.edge_label_index.shape[1] == len(record.objects) * (len(record.objects) - 1)
    assert graph.gt_predicate.numel() == len(record.relations)
    report = {
        "status": "PASS",
        "protocol": manifest["dataset"],
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "annotation_sha256": manifest["annotation_sha256"],
        "split_policy": manifest["official_split_policy"],
        "seed": args.seed,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "vocabulary_sizes": {k: len(v) for k, v in vocabulary.items() if k.endswith("_to_idx")},
        "statistics": totals,
        "image_archive": str(archive),
        "image_archive_coverage": "all split image IDs found; one training image decoded",
        "smoke_image": {"image_id": image_id, "width": record.width, "height": record.height},
        "smoke_graph": {
            "nodes": graph.num_nodes,
            "candidate_pairs": graph.edge_label_index.shape[1],
            "gt_annotation_rows": graph.gt_predicate.numel(),
            "feature_dim": graph.x.shape[1],
        },
        "source_sha256": {p: sha256(Path(p)) for p in ("src/datasets/gqa.py", "tools/prepare_level2_gqa.py")},
        "training_performed": False,
        "model_evaluation_performed": False,
        "limitations": [
            "GQA shares Visual Genome images: separate training is not cross-dataset generalization",
            "Custom full-train vocabulary; not GQA200 or a directly comparable paper benchmark",
            "OOV objects/predicates are dropped and counted; held-out GT support is filtered",
            "Image archive contents were indexed; only the smoke image was decoded in this command",
            "CLIP cache, KG mapping and model runs on GQA are not implemented by this command",
        ],
    }
    write_fixed(args.report, report)
    print(f"PASS: {args.report}")


if __name__ == "__main__":
    main()
