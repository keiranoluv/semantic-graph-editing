"""Sparse, label-independent windows from frozen M2 feature/geometry shards."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from extension_pvsg.contracts import read_json
from extension_pvsg.graphs import Graph, batch_graphs
from extension_pvsg.m2.download import sha256
from torch.utils.data import Dataset, Sampler


class Store:
    def __init__(self, cache_root, feature_root, cache_size=2):
        self.cache_root, self.feature_root = Path(cache_root), Path(feature_root)
        self.manifest = read_json(self.cache_root / "dataset.lock.json")
        self.videos = {r["video_id"]: r for r in read_json(self.cache_root / "inputs.json")}
        self.targets = {r["video_id"]: r for r in read_json(self.cache_root / "targets.json")}
        self.ignored = read_json(self.cache_root / "annotation_adaptation.json")["ignore_negative_keys"]
        self.cache_size = cache_size
        self.loaded = OrderedDict()

    def ids(self, split):
        if split not in ("fit", "validation"):
            raise ValueError("M3 training/profiling is restricted to fit and internal validation")
        return self.manifest["internal_validation"]["fit_ids" if split == "fit" else "validation_ids"]

    def arrays(self, video_id):
        if video_id in self.loaded:
            self.loaded.move_to_end(video_id)
            return self.loaded[video_id]
        gp = self.cache_root / "geometry" / f"{video_id}.npz"
        fp = self.feature_root / f"{video_id}.npz"
        record = read_json(self.cache_root / "records" / f"{video_id}.json")
        meta = read_json(self.feature_root / f"{video_id}.json")
        if sha256(gp) != record["geometry_sha256"] or sha256(fp) != meta["feature_sha256"]:
            raise ValueError(f"M2 shard hash mismatch: {video_id}")
        if meta["geometry_sha256"] != record["geometry_sha256"]:
            raise ValueError("Visual/geometry cache provenance mismatch")
        with np.load(gp, allow_pickle=False) as z:
            geometry = {k: z[k] for k in z.files}
        with np.load(fp, allow_pickle=False) as z:
            visual = {k: z[k] for k in z.files}
        positions = np.argwhere(geometry["visible"].T)
        if (
            not np.array_equal(visual["frame_ids"], positions[:, 0])
            or not np.array_equal(visual["track_ids"], geometry["track_ids"][positions[:, 1]])
            or not np.array_equal(visual["visual_available"], geometry["rgb_available"][positions[:, 0]])
        ):
            raise ValueError("Visual rows do not match the exact GT visible frame/track order")
        if visual["features"].shape != (len(positions), 512) or not np.isfinite(visual["features"]).all():
            raise ValueError("Invalid visual feature shape/values")
        available = visual["visual_available"]
        if np.any(visual["features"][~available] != 0):
            raise ValueError("Missing RGB rows must have zero features")
        norms = np.linalg.norm(visual["features"][available].astype(np.float32), axis=1)
        if not np.allclose(norms, 1, atol=1e-3):
            raise ValueError("Available CLIP features must be unit normalized")
        self.loaded[video_id] = geometry, visual
        while len(self.loaded) > self.cache_size:
            self.loaded.popitem(last=False)
        return geometry, visual

    def window(self, video_id, start, size, *, supervision=True):
        video = self.videos[video_id]
        geometry, visual = self.arrays(video_id)
        return make_window(
            video,
            geometry,
            visual,
            start,
            min(start + size, video["num_frames"]),
            target=self.targets[video_id] if supervision else None,
            ignored=self.ignored[video_id] if supervision else (),
        )


def make_window(video, geometry, visual, start, stop, *, target=None, ignored=()):
    if not 0 <= start < stop <= video["num_frames"]:
        raise ValueError("Invalid window range")
    ids = geometry["track_ids"]
    visible = geometry["visible"][:, start:stop]
    local_frame, slots = np.nonzero(visible.T)
    frames = local_frame + start
    n = len(frames)
    grid = np.full(visible.T.shape, -1, dtype=np.int64)
    grid[local_frame, slots] = np.arange(n)
    spatial = []
    for t in range(stop - start):
        nodes = grid[t, grid[t] >= 0]
        a, b = np.meshgrid(nodes, nodes, indexing="ij")
        keep = a != b
        spatial.append(np.stack((a[keep], b[keep])))
    query = np.concatenate(spatial, axis=1) if spatial else np.empty((2, 0), dtype=np.int64)
    f, s = np.nonzero(visible[:, :-1].T & visible[:, 1:].T)
    a, b = grid[f, s], grid[f + 1, s]
    temporal = np.stack((np.column_stack((a, b)).ravel(), np.column_stack((b, a)).ravel()))
    metadata = np.column_stack((frames, ids[slots], geometry["class_ids"][slots]))
    graph = Graph(video["video_id"], metadata, query, temporal, np.tile(np.arange(n), (2, 1)), query)
    first, last = np.searchsorted(visual["frame_ids"], (start, stop))
    if not np.array_equal(visual["track_ids"][first:last], ids[slots]):
        raise ValueError("Window visual rows are misaligned")
    geom = geometry["geometry"][slots, frames]
    motion = geometry["motion"][slots, frames]
    # All models receive the same time/gap inputs; track IDs never enter x.
    phase = frames.astype(np.float32) / max(video["num_frames"] - 1, 1)
    numeric = np.column_stack(
        (
            geom,
            motion,
            visual["visual_available"][first:last],
            np.minimum(geometry["delta_frames"][slots, frames], 64) / 64,
            phase,
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
        )
    ).astype(np.float32)
    pair_geometry = pair_features(geom, motion, query)
    result = dict(
        graph=graph,
        visual=visual["features"][first:last].astype(np.float32),
        numeric=numeric,
        pair_geometry=pair_geometry,
        start=start,
        stop=stop,
    )
    if target is not None:
        labels, mask = sparse_targets(graph, target, ignored)
        result.update(labels=labels, supervision_mask=mask)
    return result


def pair_features(geometry, motion, query):
    s, o = query
    a, b = geometry[s], geometry[o]
    eps = 1e-6
    delta = (b[:, :2] - a[:, :2]) / np.maximum(a[:, 2:4], eps)
    ratios = np.log(np.maximum(b[:, 2:4], eps) / np.maximum(a[:, 2:4], eps))
    low = np.maximum(a[:, :2] - a[:, 2:4] / 2, b[:, :2] - b[:, 2:4] / 2)
    high = np.minimum(a[:, :2] + a[:, 2:4] / 2, b[:, :2] + b[:, 2:4] / 2)
    intersection = np.maximum(high - low, 0).prod(axis=1)
    iou = intersection / np.maximum(a[:, 4] + b[:, 4] - intersection, eps)
    return np.column_stack((np.clip(delta, -10, 10), np.clip(ratios, -10, 10), iou, motion[o] - motion[s])).astype(
        np.float32
    )


def sparse_targets(graph, target, ignored):
    q = graph.query_edges
    frames = graph.nodes[q[0], 0]
    subjects, objects = graph.nodes[q[0], 1], graph.nodes[q[1], 1]
    labels = np.zeros((q.shape[1], 57), dtype=np.float32)
    mask = np.ones_like(labels, dtype=bool)
    for row in target["relations"]:
        pair = (subjects == row["subject_id"]) & (objects == row["object_id"])
        for low, high in row["intervals"]:
            labels[pair & (frames >= low) & (frames <= high), row["predicate_id"]] = 1
    for row in ignored:
        pair = (subjects == row["subject_id"]) & (objects == row["object_id"])
        predicate = row["predicate_id"]
        mask[pair, predicate] = labels[pair, predicate] > 0
    return labels, mask


def collate(samples):
    graph = batch_graphs([s["graph"] for s in samples])
    result = {k: torch.from_numpy(v) for k, v in graph.items() if isinstance(v, np.ndarray)}
    for name in ("visual", "numeric", "pair_geometry", "labels", "supervision_mask"):
        if name in samples[0]:
            result[name] = torch.from_numpy(np.concatenate([s[name] for s in samples]))
    result["sample_count"] = len(samples)
    result["windows"] = [(s["graph"].video_id, s["start"], s["stop"]) for s in samples]
    return result


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class Windows(Dataset):
    def __init__(self, store, size):
        self.store, self.size = store, size

    def __getitem__(self, request):
        return self.store.window(request[0], request[1], self.size)

    def __len__(self):
        return len(self.store.ids("fit"))


class FitSampler(Sampler):
    def __init__(self, store, size, per_video, seed):
        self.videos, self.ids = store.videos, sorted(store.ids("fit"))
        self.size, self.per_video, self.seed, self.epoch = size, per_video, seed, 0

    def __len__(self):
        return len(self.ids) * self.per_video

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        requests = [
            (vid, int(rng.integers(max(1, self.videos[vid]["num_frames"] - self.size + 1))))
            for vid in self.ids
            for _ in range(self.per_video)
        ]
        rng.shuffle(requests)
        return iter(requests)
