"""CPU provenance, train-only coverage and GloVe preparation for the VG150 study."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.datasets.vg150 import VG150AnnotationLoader
from src.features.glove import map_vocabulary, needed_tokens, read_glove_subset
from src.features.visual import load_visual_manifest
from src.knowledge.export import load_class_features
from src.knowledge.transe.core import sha256
from tools.prepare_magicbrush import digest, encode, fixed
from tools.vg150_penet_v1.data import cohort, normalized_record, read_splits

MODELS = ("mlp", "gat", "gat-kg", "pe-ce", "pe-pl", "pe-plpr")


def validate(config):
    if (
        config["schema_version"] != 1
        or config["device"] != "cuda"
        or config["models"] != list(MODELS)
        or config["seeds"] != [42, 123, 3407]
    ):
        raise ValueError("Expected VG150 v1 six-model/three-seed CUDA handoff config")
    for name in ("epochs", "train_batch_size", "val_batch_size", "checkpoint_every_batches", "num_threads"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError("Invalid positive integer budget")
    if config["lr"] <= 0 or config["weight_decay"] < 0 or config["gradient_clip_norm"] <= 0:
        raise ValueError("Invalid optimization settings")
    if config["early_stopping"] != {"enabled": True, "patience": 15, "min_delta": 0.0}:
        raise ValueError("v1 early stopping is fixed to patience 15, min_delta 0")
    if config["pilot"] != {"train_images": 512, "val_images": 128, "epochs": 5, "seed": 42}:
        raise ValueError("v1 pilot must use the fixed 512/128 cohort, 5 epochs, seed 42")
    from tools.prepare_magicbrush import ROOT

    for key, folder in (("prepared", "data"), ("report", "reports"), ("output", "runs")):
        if not Path(config[key]).resolve().is_relative_to(ROOT / folder):
            raise ValueError(f"{key} must be under {folder}/")


def assets(config):
    validate(config)
    root, cache = Path(config["root"]), Path(config["visual_cache"])
    for filename, expected in config["expected_data_sha256"].items():
        if sha256(root / filename) != expected:
            raise ValueError(f"Frozen VG150 input changed: {filename}")
    if sha256(cache / "manifest.json") != config["expected_visual_manifest_sha256"]:
        raise ValueError("Frozen visual manifest changed")
    visual = load_visual_manifest(cache, root, "clip")
    _, valid, knowledge = load_class_features(Path(config["kg_features"]))
    if (
        knowledge["kg_source"] != "wordnet"
        or knowledge["input_sha256"]["vocabulary"] != config["expected_data_sha256"]["VG-SGG-dicts-with-attri.json"]
        or int(valid.sum()) != 150
    ):
        raise ValueError("Expected reviewed VG150 WordNet alignment with all 150 foreground classes")
    return {
        "data_sha256": config["expected_data_sha256"],
        "visual_manifest_sha256": sha256(cache / "manifest.json"),
        "visual_features_sha256": visual["features_sha256"],
        "kg_manifest_sha256": sha256(Path(config["kg_features"]) / "manifest.json"),
        "kg_tensor_sha256": knowledge["tensor_sha256"],
        "kg_review_sha256": knowledge["review_manifest_sha256"],
    }


def train_statistics(loader):
    if loader.split != "train":
        raise ValueError("Coverage/prior statistics must only inspect training annotations")
    counts, objects, predicates, node_counts = Counter(), Counter(), Counter(), []
    for i in range(len(loader)):
        original = loader.load(i)
        record = normalized_record(original)
        if any(not 1 <= n.label_id <= 150 for n in record.objects) or any(
            not 1 <= r.predicate_id <= 50 for r in record.relations
        ):
            raise ValueError("Out-of-taxonomy annotation")
        pairs = {}
        triples = set()
        for relation in record.relations:
            pairs.setdefault((relation.subject_id, relation.object_id), set()).add(relation.predicate_id)
            triples.add((relation.subject_id, relation.object_id, relation.predicate_id))
        n = len(record.objects)
        counts.update(
            images=1,
            objects=n,
            candidate_pairs=n * (n - 1),
            positive_pairs=len(pairs),
            multi_predicate_pairs=sum(len(v) > 1 for v in pairs.values()),
            GT_rows=len(record.relations),
            unique_GT_triplets=len(triples),
            self_GT_rows_excluded=len(original.relations) - len(record.relations),
            images_without_GT=int(not record.relations),
        )
        objects.update(n.label_id for n in record.objects)
        predicates.update(r.predicate_id for r in record.relations)
        node_counts.append(n)
        if (i + 1) % 5000 == 0:
            print(f"[CPU TRAIN AUDIT] {i+1}/{len(loader)}", flush=True)
    return {
        **dict(counts),
        "objects_per_image": {str(p): float(np.percentile(node_counts, p)) for p in (50, 90, 99, 100)},
        "object_support": {str(i): objects[i] for i in range(1, 151)},
        "predicate_support": {str(i): predicates[i] for i in range(1, 51)},
        "absent_object_ids": [i for i in range(1, 151) if not objects[i]],
        "absent_predicate_ids": [i for i in range(1, 51) if not predicates[i]],
        "statistics_source": "train only; annotation rows retained for evaluation, distinct positive pairs for sampling",
    }


def prepare(config):
    fingerprint = assets(config)
    root, output, report = (Path(config[k]) for k in ("root", "prepared", "report"))
    splits = read_splits(config)
    cohort(splits)  # Validate row disjointness before reading any annotation labels.
    metadata = json.loads((root / "image_data.json").read_text())
    image_membership = {}
    for split in ("train", "val", "test"):
        for row in splits[split]:
            image_id = str(metadata[row]["image_id"])
            if image_id in image_membership:
                raise ValueError("Duplicate original image ID inside/across VG150 splits")
            image_membership[image_id] = split
    vocabulary = json.loads((root / "VG-SGG-dicts-with-attri.json").read_text())
    labels, predicates = vocabulary["label_to_idx"], vocabulary["predicate_to_idx"]
    if set(labels.values()) != set(range(1, 151)) or set(predicates.values()) != set(range(1, 51)):
        raise ValueError("Expected fixed VG150 taxonomy")
    loader = VG150AnnotationLoader(root, "train", root / "splits.json", box_policy="clip")
    try:
        statistics = train_statistics(loader)
    finally:
        loader._annotations.close()
    provenance = json.loads(Path(config["glove_provenance"]).read_text())
    if sha256(Path(config["glove_text"])) != provenance["text_sha256"]:
        raise ValueError("Pinned GloVe text changed")
    vectors, lines = read_glove_subset(config["glove_text"], needed_tokens([labels, predicates]))
    objects, object_info = map_vocabulary(labels, vectors, "object")
    relations, predicate_info = map_vocabulary(predicates, vectors, "predicate")
    arrays = output / "glove-vectors.npz"
    arrays.parent.mkdir(parents=True, exist_ok=True)
    if arrays.exists():
        with np.load(arrays, allow_pickle=False) as saved:
            if not np.array_equal(saved["objects"], objects) or not np.array_equal(saved["predicates"], relations):
                raise ValueError("Existing VG150 GloVe arrays changed; use a new version")
    else:
        with arrays.open("xb") as stream:
            np.savez(stream, objects=objects, predicates=relations)
    pilot = cohort(splits, pilot=True)
    sources = [
        Path(__file__),
        Path(__file__).with_name("data.py"),
        Path("src/features/glove.py"),
        Path("src/datasets/vg150.py"),
        Path("src/features/visual.py"),
        Path("src/graph/builder.py"),
        Path("tools/run_gqa_full_prepared.py"),
    ]
    result = {
        "schema_version": 1,
        "dataset": "VG150 official frozen Level1 splits; new multi-positive supervision",
        "artifacts": fingerprint,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_rows_sha256": {k: digest(v) for k, v in splits.items()},
        "pilot": pilot,
        "vocabulary": {"label_to_idx": labels, "predicate_to_idx": predicates},
        "train_statistics": statistics,
        "glove": {
            "source": provenance,
            "rows_scanned": lines,
            "arrays_sha256": sha256(arrays),
            "dimension": 300,
            "objects": object_info,
            "predicates": predicate_info,
            "policy": "fixed taxonomy; exact / object-longest-word / predicate-word-mean / label-keyed seeded fallback",
            "fallback_seed": 42,
        },
        "source_sha256": {str(p): sha256(p) for p in sources},
        "val_annotation_labels_used_for_statistics_or_mapping": False,
        "test_annotation_labels_opened": False,
        "GPU_compute_started": False,
        "new_training_started": False,
    }
    fixed(output / "manifest.json", encode(result))
    fixed(report / "preparation.json", encode(result))
    fixed(report / "pilot-cohort.json", encode(pilot))
    print(
        json.dumps(
            {
                "status": "PASS",
                "splits": result["split_sizes"],
                "glove_objects": object_info["counts"],
                "glove_predicates": predicate_info["counts"],
                "training_started": False,
            },
            indent=2,
        )
    )
    return result


def load_prepared(config, *, verify_assets=True):
    path = Path(config["prepared"])
    metadata = json.loads((path / "manifest.json").read_text())
    if verify_assets and assets(config) != metadata["artifacts"]:
        raise ValueError("Prepared source assets changed")
    for name, expected in metadata["source_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Prepared implementation changed: {name}")
    if sha256(path / "glove-vectors.npz") != metadata["glove"]["arrays_sha256"]:
        raise ValueError("Prepared GloVe arrays changed")
    with np.load(path / "glove-vectors.npz", allow_pickle=False) as saved:
        arrays = [saved[k].copy() for k in ("objects", "predicates")]
    if [a.shape for a in arrays] != [(151, 300), (51, 300)] or any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid VG150 GloVe shape/values")
    return metadata, arrays
