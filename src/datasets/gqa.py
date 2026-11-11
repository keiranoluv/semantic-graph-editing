"""GQA box-based PredCls adapter, with project splits and train-only vocabulary.

Not GQA question answering and not a claim of reproducing GQA200/GQA100.
Original strings are retained; no synonym or singular/plural normalization.
"""

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.datasets.vg150 import _clip_box_to_image
from src.knowledge.transe.core import sha256


def make_splits(train_ids, official_val_ids, *, seed=42, val_size=5000):
    train_ids, official_val_ids = list(train_ids), list(official_val_ids)
    if len(set(train_ids)) != len(train_ids) or len(set(official_val_ids)) != len(official_val_ids):
        raise ValueError("Duplicate GQA image IDs")
    if set(train_ids) & set(official_val_ids):
        raise ValueError("Overlapping official GQA splits")
    if not 0 < val_size < len(train_ids):
        raise ValueError("Internal validation size must leave training images")
    ordered = sorted(train_ids, key=lambda x: (hashlib.sha256(f"{seed}:{x}".encode()).hexdigest(), x))
    return {"train": sorted(ordered[val_size:]), "val": sorted(ordered[:val_size]), "test": sorted(official_val_ids)}


def build_vocabulary(annotations, train_ids):
    counts = {"label": Counter(), "predicate": Counter(), "attribute": Counter()}
    for image_id in train_ids:
        for obj in annotations[image_id]["objects"].values():
            counts["label"][obj["name"]] += 1
            counts["attribute"].update(obj["attributes"])
            counts["predicate"].update(r["name"] for r in obj["relations"])
    return {
        "policy": "exact_strings_sorted_lexicographically_from_project_train_raw_annotations_only",
        **{f"{kind}_to_idx": {name: i + 1 for i, name in enumerate(sorted(c))} for kind, c in counts.items()},
    }


def normalize_gqa(image_id, annotation, vocabulary):
    """Drop invalid/OOV objects and incident relations; report each reason.

    Clip finite positive-area boxes to the image, then reject degenerate results.
    Preserve duplicate relation rows and multiple predicates on an ordered pair.
    OOV attributes are omitted and counted, not learned from validation/test."""
    width, height = annotation["width"], annotation["height"]
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Invalid GQA image dimensions")
    raw = annotation["objects"]
    ids = [int(k) for k in raw]
    if len(ids) != len(set(ids)):
        raise ValueError("Object IDs collide after integer conversion")
    counts = Counter(raw_objects=len(raw), raw_relations=sum(len(o["relations"]) for o in raw.values()))
    objects = []
    for object_id in sorted(raw, key=int):
        obj = raw[object_id]
        label = vocabulary["label_to_idx"].get(obj["name"])
        if label is None:
            counts["dropped_oov_objects"] += 1
            continue
        x, y, w, h = (float(obj[k]) for k in ("x", "y", "w", "h"))
        box = (x, y, x + w, y + h)
        try:
            clipped = _clip_box_to_image(box, width, height)
        except ValueError:
            counts["dropped_invalid_box_objects"] += 1
            continue
        counts["clipped_objects"] += clipped != box
        attrs = []
        for name in obj["attributes"]:
            attr = vocabulary["attribute_to_idx"].get(name)
            if attr is None:
                counts["dropped_oov_attributes"] += 1
            else:
                attrs.append(attr)
        objects.append(SceneObject(int(object_id), label, clipped, tuple(sorted(set(attrs)))))
    known = {o.object_id for o in objects}
    relations = []
    for subject in sorted(raw, key=int):
        for r in raw[subject]["relations"]:
            if str(r["object"]) not in raw:
                raise ValueError("Dangling endpoint in raw GQA annotation")
            if int(subject) not in known or int(r["object"]) not in known:
                counts["dropped_relations_filtered_endpoint"] += 1
            elif int(subject) == int(r["object"]):
                counts["dropped_self_relations"] += 1
            elif r["name"] not in vocabulary["predicate_to_idx"]:
                counts["dropped_oov_relations"] += 1
            else:
                relations.append(
                    SceneRelation(int(subject), int(r["object"]), vocabulary["predicate_to_idx"][r["name"]])
                )
    record = SceneGraphRecord(
        str(image_id),
        width,
        height,
        tuple(objects),
        tuple(relations),
        {"dataset": "gqa", "box_policy": "clip_drop_invalid"},
    )
    record.validate()
    counts.update(objects=len(objects), relations=len(relations), images=1)
    counts["images_without_relations"] += not bool(relations)
    counts["images_fewer_than_two_objects"] += len(objects) < 2
    return record, counts


class GQAAnnotationLoader:
    def __init__(self, root, split, manifest_path):
        self.root = Path(root)
        manifest = json.loads(Path(manifest_path).read_text())
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        groups = [manifest["splits"][k] for k in ("train", "val", "test")]
        if any(len(g) != len(set(g)) for g in groups) or any(
            set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)
        ):
            raise ValueError("Non-disjoint image split")
        for name, digest in manifest["annotation_sha256"].items():
            if sha256(self.root / "sceneGraphs" / name) != digest:
                raise ValueError("GQA annotations differ from frozen manifest")
        filename = "val_sceneGraphs.json" if split == "test" else "train_sceneGraphs.json"
        self.annotations = json.loads((self.root / "sceneGraphs" / filename).read_text())
        self.indices = manifest["splits"][split]
        if not set(self.indices) <= self.annotations.keys():
            raise ValueError("Split contains images outside its official annotation file")
        self.vocabulary = manifest["vocabulary"]

    def __len__(self):
        return len(self.indices)

    def load(self, index):
        image_id = self.indices[index]
        return normalize_gqa(image_id, self.annotations[image_id], self.vocabulary)[0]


def image_archive_index(archive):
    """Index images by original GQA image ID without extracting the 22 GB ZIP."""
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpg") and not n.startswith("__MACOSX/")]
    index = {Path(n).stem: n for n in names}
    if len(index) != len(names):
        raise ValueError("Duplicate JPEG basenames in GQA archive")
    return index


def load_archive_image(archive, image_id, index):
    with zipfile.ZipFile(archive) as z:
        data = z.read(index[str(image_id)])
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")
