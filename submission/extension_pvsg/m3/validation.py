"""Full internal-validation timelines; no test evaluation or pair filtering by GT."""

import numpy as np
from extension_pvsg.evaluation import evaluate
from extension_pvsg.intervals import window_bounds
from extension_pvsg.m3.data import collate, sparse_targets, to_device
from extension_pvsg.m3.decoding import Accumulator
from extension_pvsg.m3.engine import predict_batch


def validate(model, store, cfg, device, batch_size):
    model.eval()
    predictions = []
    video_ids = sorted(store.ids("validation"))
    window_losses = []
    for vid in video_ids:
        video = store.videos[vid]
        merge = Accumulator(video)
        bounds = window_bounds(video["num_frames"], size=cfg["window_frames"], stride=cfg["validation_stride"])
        for low in range(0, len(bounds), batch_size):
            samples = [
                store.window(vid, start, stop - start, supervision=False)
                for start, stop in bounds[low : low + batch_size]
            ]
            batch = to_device(collate(samples), device)
            scores = predict_batch(model, batch, cfg["pair_chunk_size"], cfg["precision"])
            offset = 0
            for sample in samples:
                count = sample["graph"].query_edges.shape[1]
                z = scores[offset : offset + count]
                merge.add(sample["graph"], z, sample["start"])
                # Targets are accessed only after inference, for loss/metrics.
                labels, mask = sparse_targets(sample["graph"], store.targets[vid], store.ignored[vid])
                bce = np.maximum(z, 0) - z * labels + np.log1p(np.exp(-np.abs(z)))
                window_losses.append(float((bce * mask).sum() / max(mask.sum(), 1)))
                offset += count
            del batch, scores
        predictions.append(merge.finish())
    report = evaluate(
        [store.videos[v] for v in video_ids],
        [store.targets[v] for v in video_ids],
        predictions,
        expected_video_ids=video_ids,
    )
    report["validation_bce"] = float(np.mean(window_losses))
    report["validation_loss_weighting"] = "mean_per_window; overlapping validation windows"
    return report
