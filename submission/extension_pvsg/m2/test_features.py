import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from extension_pvsg.contracts import read_json
from extension_pvsg.m2 import features
from extension_pvsg.m2.download import sha256


def test_crop_outside_mask_is_constant_and_preprocessing_is_deterministic():
    config = read_json(features.CONFIG_PATH)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 7
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[mask == 7] = [200, 50, 70]
    b = a.copy()
    b[mask != 7] = 255
    first = features.preprocess_crop(a, mask, 7, [0, 0, 4, 4], config)
    np.testing.assert_array_equal(first, features.preprocess_crop(b, mask, 7, [0, 0, 4, 4], config))
    assert first.shape == (3, 224, 224) and first.dtype == np.float32
    assert np.isfinite(first).all()


def test_cache_alignment_missing_rgb_and_resume_identity(tmp_path, monkeypatch):
    for name in ("records", "geometry", "output"):
        (tmp_path / name).mkdir()
    video_id = "tiny"
    source = tmp_path / "tiny.mp4"
    source.write_bytes(b"fake decoded by fixture")
    masks = tmp_path / "masks.zip"
    members = []
    with zipfile.ZipFile(masks, "w") as archive:
        for t in range(3):
            encoded = io.BytesIO()
            Image.fromarray(np.full((2, 2), 7, dtype=np.uint8)).save(encoded, format="PNG")
            members.append(f"{t:04d}.png")
            archive.writestr(members[-1], encoded.getvalue())
    geometry = tmp_path / "geometry/tiny.npz"
    np.savez(
        geometry,
        visible=np.ones((1, 3), dtype=bool),
        boxes=np.tile([0, 0, 2, 2], (1, 3, 1)),
        track_ids=np.array([7]),
        rgb_available=np.array([True, True, False]),
    )
    (tmp_path / "records/tiny.json").write_text(
        json.dumps({"geometry_sha256": sha256(geometry), "extra_unannotated_rgb_frames": 0})
    )

    class Frame:
        def to_ndarray(self, format):
            return np.full((2, 2, 3), 100, dtype=np.uint8)

    monkeypatch.setattr(features, "iter_video_timeline", lambda path: iter([Frame(), Frame()]))
    calls = []

    def encoder(crops):
        calls.append(len(crops))
        return np.tile(np.eye(1, 512, dtype=np.float32), (len(crops), 1))

    kwargs = {
        "media": {
            "video_path": "tiny.mp4",
            "video_sha256": sha256(source),
            "mask_archive": "masks.zip",
            "mask_members": members,
        },
        "media_root": tmp_path,
        "cache_root": tmp_path,
        "output": tmp_path / "output",
        "config": read_json(features.CONFIG_PATH),
        "identity": {"test": "v1"},
        "batch_size": 2,
    }
    result = features.extract_video(video_id, encoder, **kwargs)
    assert result["rows"] == 3 and result["visual_rows"] == 2 and result["missing_rgb_rows"] == 1
    with np.load(tmp_path / "output/tiny.npz") as cached:
        assert cached["frame_ids"].tolist() == [0, 1, 2]
        assert cached["track_ids"].tolist() == [7, 7, 7]
        assert not cached["features"][2].any()
        assert cached["visual_available"].tolist() == [True, True, False]
    assert features.extract_video(video_id, encoder, **kwargs) == result and calls == [2]
    kwargs["identity"] = {"test": "different"}
    with pytest.raises(ValueError, match="provenance"):
        features.extract_video(video_id, encoder, **kwargs)
