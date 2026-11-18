"""Sparse streaming implementation of the frozen M1 decoder, checked against it."""

import numpy as np

from ..contracts import PROTOCOL_ID, normalize_predictions, visibility_masks
from ..intervals import mask_to_intervals


def query_keys(frames, subjects, objects, ids):
    n = len(ids)
    return (np.asarray(frames) * n + np.searchsorted(ids, subjects)) * n + np.searchsorted(ids, objects)


class Accumulator:
    def __init__(self, video):
        self.video = video
        self.ids = np.array(sorted(t["track_id"] for t in video["tracks"]), dtype=np.int64)
        masks = visibility_masks(video)
        frames, subjects, objects = [], [], []
        for t in range(video["num_frames"]):
            active = self.ids[[masks[int(track)][t] for track in self.ids]]
            a, b = np.meshgrid(active, active, indexing="ij")
            keep = a != b
            subjects.extend(a[keep])
            objects.extend(b[keep])
            frames.extend([t] * int(keep.sum()))
        self.frames = np.array(frames, dtype=np.int64)
        self.subjects = np.array(subjects, dtype=np.int64)
        self.objects = np.array(objects, dtype=np.int64)
        self.keys = query_keys(self.frames, self.subjects, self.objects, self.ids)
        self.sums = np.zeros((len(self.keys), 57), dtype=np.float64)
        self.counts = np.zeros(len(self.keys), dtype=np.int64)
        self.last_start = -1

    def add(self, graph, logits, start):
        if start <= self.last_start:
            raise ValueError("Merge windows once in strictly increasing start order")
        self.last_start = start
        q = graph.query_edges
        frames = graph.nodes[q[0], 0]
        keys = query_keys(frames, graph.nodes[q[0], 1], graph.nodes[q[1], 1], self.ids)
        locations = np.searchsorted(self.keys, keys)
        if (
            logits.shape != (len(keys), 57)
            or not np.isfinite(logits).all()
            or np.any(locations >= len(self.keys))
            or not np.array_equal(self.keys[locations], keys)
        ):
            raise ValueError("Invalid window prediction coverage or values")
        self.sums[locations] += logits.astype(np.float64)
        self.counts[locations] += 1

    def finish(self):
        if np.any(self.counts == 0):
            raise ValueError("Not every covisible query/frame was inferred")
        np.divide(self.sums, self.counts[:, None], out=self.sums, where=self.counts[:, None] > 0)
        rows = []
        order = np.lexsort((self.frames, self.objects, self.subjects))
        changed = (self.subjects[order][1:] != self.subjects[order][:-1]) | (
            self.objects[order][1:] != self.objects[order][:-1]
        )
        bounds = np.r_[0, np.flatnonzero(changed) + 1, len(order)]
        for low, high in zip(bounds[:-1], bounds[1:], strict=True):
            if low == high:
                continue
            keep = order[low:high]
            subject, obj = self.subjects[keep[0]], self.objects[keep[0]]
            frames, z = self.frames[keep], self.sums[keep]
            exp = np.exp(-np.abs(z))
            p = np.where(z >= 0, 1 / (1 + exp), exp / (1 + exp))
            for predicate in np.flatnonzero((p >= 0.5).any(axis=0)):
                active = p[:, predicate] >= 0.5
                timeline = np.zeros(self.video["num_frames"], dtype=bool)
                timeline[frames[active]] = True
                rows.append(
                    dict(
                        subject_id=int(subject),
                        object_id=int(obj),
                        predicate_id=int(predicate),
                        intervals=mask_to_intervals(timeline),
                        score=float(p[active, predicate].mean()),
                    )
                )
        return normalize_predictions(
            dict(
                schema_version=1,
                kind="relation_predictions",
                protocol_id=PROTOCOL_ID,
                video_id=self.video["video_id"],
                num_frames=self.video["num_frames"],
                predictions=rows,
            ),
            self.video,
        )
