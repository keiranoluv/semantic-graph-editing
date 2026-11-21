"""Read existing VG150 train/val and CLIP cache without changing historic assets."""

import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from src.datasets.vg150 import VG150AnnotationLoader
from src.features.visual import GeometryVisualFeatureExtractor, VisualFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from tools.prepare_magicbrush import digest
from tools.run_gqa_full_prepared import prepare_supervision


def normalized_record(record):
    # All-pairs candidate space excludes self edges. Retain duplicate and
    # multi-predicate annotation rows for reference evaluation.
    return replace(record, relations=tuple(r for r in record.relations if r.subject_id != r.object_id))


def cohort(splits, *, pilot=False, train_size=512, val_size=128, seed=42):
    sets = [set(splits[k]) for k in ("train", "val", "test")]
    if any(len(sets[i]) != len(splits[k]) for i, k in enumerate(("train", "val", "test"))) or any(
        sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Non-disjoint VG150 split rows")
    selected = {k: list(splits[k]) for k in ("train", "val")}
    if pilot:
        for split, size in (("train", train_size), ("val", val_size)):
            if not 0 < size <= len(selected[split]):
                raise ValueError("Pilot split budget exceeds available images")
            selected[split] = sorted(
                selected[split], key=lambda row: (digest(["vg150-penet-pilot-v1", seed, row]), row)
            )[:size]
    return selected


class Dataset:
    def __init__(self, loader, features, rows, *, cache_size=128):
        if loader.split not in ("train", "val") or not set(rows) <= set(loader.indices) or len(rows) != len(set(rows)):
            raise ValueError("Only distinct rows of the selected train/val split are supported")
        self.loader, self.features, self.rows = loader, features, list(rows)
        self.positions = {row: i for i, row in enumerate(loader.indices)}
        self.builder = SceneGraphBuilder()
        self.cache_size, self.cache = cache_size, OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        if row not in self.cache:
            record = normalized_record(self.loader.load(self.positions[row]))
            if any(not 1 <= n.label_id <= 150 for n in record.objects) or any(
                not 1 <= r.predicate_id <= 50 for r in record.relations
            ):
                raise ValueError("VG150 object/predicate outside 150/50 foreground taxonomy")
            graph = self.builder.build(record, self.features.extract(record))
            self.cache[row] = prepare_supervision(graph, 51)
        self.cache.move_to_end(row)
        value = self.cache[row]
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def close(self):
        self.cache.clear()
        self.loader._annotations.close()
        self.features.visual._features.close()


def datasets(config, selected):
    root, cache = Path(config["root"]), Path(config["visual_cache"])
    features = GeometryVisualFeatureExtractor(VisualFeatureExtractor(cache, root, "clip"))
    result = {}
    for split in ("train", "val"):
        loader = VG150AnnotationLoader(root, split, root / "splits.json", box_policy="clip")
        result[split] = Dataset(loader, features, selected[split])
    return result


def read_splits(config):
    return json.loads((Path(config["root"]) / "splits.json").read_text())
