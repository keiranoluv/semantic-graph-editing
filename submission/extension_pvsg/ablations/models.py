"""GT-track adaptations, NOT end-to-end PVSG benchmark reproductions.

Conv design reference: Yang et al., CVPR 2023, section 5.2 (k=5, 3 layers).
See provenance/sources.json and README for source version and changed heads.
"""

import numpy as np
import torch
from extension_pvsg.baselines.data import collate as collate_m3
from extension_pvsg.baselines.models import RelationModel as M3RelationModel
from torch import nn


class EdgeAblation(M3RelationModel):
    """Keep every ST-GAT parameter/self-loop; remove only message edges."""

    def __init__(self, kind, cfg):
        super().__init__("st_gat", cfg)
        self.variant = kind

    def encode(self, batch):
        changed = dict(batch)
        if self.variant in ("st_no_spatial", "st_no_edges"):
            changed["spatial_edges"] = batch["spatial_edges"][:, :0]
        if self.variant in ("st_no_temporal", "st_no_edges"):
            changed["temporal_edges"] = batch["temporal_edges"][:, :0]
            changed["temporal_delta_frames"] = batch["temporal_delta_frames"][:0]
        return super().encode(changed)


class PairTemporalConv(M3RelationModel):
    """M3 MLP pair projection -> temporal Conv/ReLU -> shared 57-label head.

    Convolve each contiguous co-visible pair segment independently. Re-mask the
    padding after EVERY layer, so artificial padded activations cannot leak back
    into real frames. Encoder outputs pair rows, compatible with M3's proxy-based
    chunked loss/backward. Pair chunks never truncate the temporal receptive field."""

    def __init__(self, kind, cfg):
        super().__init__("mlp", cfg)
        spec = cfg["variants"][kind]
        h = cfg["hidden_dim"]
        kernel, layers = spec["kernel_size"], spec["temporal_layers"]
        self.receptive_field = 1 + layers * (kernel - 1)
        self.pair_chunk_size = cfg["pair_chunk_size"]
        self.convs = nn.ModuleList([nn.Conv1d(h, h, kernel, padding=kernel // 2) for _ in range(layers)])

    def encode(self, batch):
        nodes = super().encode(batch)
        count = batch["query_edges"].shape[1]
        if not count:
            return nodes[:0]
        rows = []
        for low in range(0, count, self.pair_chunk_size):
            high = low + self.pair_chunk_size
            q = batch["query_edges"][:, low:high]
            s, o = nodes[q[0]], nodes[q[1]]
            pair = torch.cat((s, o, s * o, s - o, batch["pair_geometry"][low:high]), dim=1)
            rows.append(self.head[1](self.head[0](pair)))
        pairs = torch.cat(rows)
        padded = torch.cat((pairs, pairs.new_zeros(1, pairs.shape[1])))
        outputs = []
        for width in batch["sequence_widths"]:
            indices = batch[f"sequence_indices_{width}"]
            valid = indices != count
            x = padded[indices].transpose(1, 2)
            mask = valid.unsqueeze(1).to(x.dtype)
            for conv in self.convs:
                x = torch.relu(conv(x)) * mask
            outputs.append(x.transpose(1, 2)[valid])
        return torch.cat(outputs)[batch["sequence_inverse"]]

    def score(self, encoded, batch, low=0, high=None):
        return self.head[3](self.head[2](encoded[low:high]))


def RelationModel(kind, cfg):
    if kind not in cfg["models"]:
        raise ValueError(f"Unknown M4 model: {kind}")
    if kind.startswith("st_no_"):
        return EdgeAblation(kind, cfg)
    if kind in ("pvsg_conv_k5_l3", "mlp_conv_k3_l2"):
        return PairTemporalConv(kind, cfg)
    raise ValueError(f"No implementation for M4 model: {kind}")


def collate(samples):
    batch = collate_m3(samples)
    frames = batch["query_frame_indices"].numpy()
    pairs = batch["query_track_pairs"].numpy()
    windows = batch["query_batch"].numpy()
    count = len(frames)
    # A window ID is essential: even overlapping windows from the same video
    # must remain independent. Track IDs are grouping keys, never features.
    order = np.lexsort((frames, pairs[:, 1], pairs[:, 0], windows))
    groups = np.column_stack((windows, pairs))[order]
    cuts = np.flatnonzero(np.any(groups[1:] != groups[:-1], axis=1) | (np.diff(frames[order]) != 1)) + 1
    buckets = {}
    if count:
        for segment in np.split(order, cuts):
            width = 1 << (len(segment) - 1).bit_length()
            buckets.setdefault(width, []).append(segment)
    flattened = []
    for width, segments in sorted(buckets.items()):
        indices = np.full((len(segments), width), count, dtype=np.int64)
        for row, segment in enumerate(segments):
            indices[row, : len(segment)] = segment
            flattened.extend(segment)
        batch[f"sequence_indices_{width}"] = torch.from_numpy(indices)
    batch["sequence_widths"] = tuple(sorted(buckets))
    batch["sequence_inverse"] = torch.from_numpy(np.argsort(np.asarray(flattened, dtype=np.int64)))
    return batch
