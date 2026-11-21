import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from extension_pvsg.m2.geometry import geometry_from_mask, motion_from_geometry
from extension_pvsg.m2.media import archive_members


@pytest.mark.parametrize("rgb_indices,missing,extra", [([0, 1], [2], 0), (list(range(5)), [], 2), ([0, 2], [1], 0)])
def test_audit_preserves_mask_timeline_when_rgb_tail_differs(tmp_path, monkeypatch, rgb_indices, missing, extra):
    from extension_pvsg.synthetic import synthetic_masks, synthetic_record
    from extension_pvsg.m2 import audit_media
    from extension_pvsg.m2 import media as media_module

    record = synthetic_record()
    record["meta"]["num_frames"] = 3
    path = tmp_path / "masks.zip"
    members = []
    with zipfile.ZipFile(path, "w") as archive:
        for i, mask in enumerate(synthetic_masks()[:3]):
            encoded = io.BytesIO()
            Image.fromarray(mask.astype(np.uint8)).save(encoded, format="PNG")
            name = f"video/{i:04d}.png"
            archive.writestr(name, encoded.getvalue())
            members.append(name)

    class Frame:
        width, height = 3, 2

        def __init__(self, index):
            self.time = index / 5

    monkeypatch.setattr(media_module, "iter_video_frames", lambda path: iter(Frame(i) for i in rgb_indices))
    for subdir in ("records", "geometry", "overlays"):
        (tmp_path / subdir).mkdir()
    media = {
        "header": {"frames": len(rgb_indices), "width": 3, "height": 2, "fps": 5},
        "mask_count": 3,
        "source": "vidor",
        "video_path": "fake.mp4",
        "mask_archive": "masks.zip",
        "mask_members": members,
    }
    with zipfile.ZipFile(path) as archive:
        result = audit_media.audit_one("synthetic_m1", record, media, tmp_path, tmp_path, archive, False, "test-code")
    assert result["missing_rgb_frame_ids"] == missing
    assert result["extra_unannotated_rgb_frames"] == extra
    assert result["input"]["num_frames"] == result["mask_frames"] == 3
    with np.load(tmp_path / "geometry/synthetic_m1.npz") as cache:
        assert np.flatnonzero(~cache["rgb_available"]).tolist() == missing


def test_geometry_pixel_edges_and_absent_track():
    mask = np.array([[0, 7, 7, 0], [0, 7, 0, 0]], dtype=np.uint8)
    boxes, geometry, visible, ignored = geometry_from_mask(mask, [7, 12])
    assert ignored == {}
    assert boxes.tolist() == [[1, 0, 3, 2], [0, 0, 0, 0]]
    assert visible.tolist() == [True, False]
    np.testing.assert_allclose(geometry[0], [0.5, 0.5, 0.5, 1, 0.5, 3 / 8])
    assert not geometry[1].any()
    with pytest.raises(ValueError, match="Unannotated"):
        geometry_from_mask(mask, [12])
    boxes, geometry, visible, ignored = geometry_from_mask(mask, [12], allow_unannotated=True)
    assert ignored == {7: 3} and not visible.any()


def test_motion_uses_valid_observations_and_elapsed_frames():
    geometry = np.zeros((1, 4, 6), dtype=np.float32)
    geometry[0, 0, 0] = 0.2
    geometry[0, 3, 0] = 0.8
    motion, delta = motion_from_geometry(geometry, np.array([[True, False, False, True]]))
    assert motion[0, 3, 0] == pytest.approx(0.2)
    assert delta.tolist() == [[0, 0, 0, 3]]
    assert not motion[0, :3].any()


def test_archive_paths_do_not_flatten_mask_video_identity(tmp_path):
    path = tmp_path / "masks.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("old/server/masks/a/0000.png", b"a")
        archive.writestr("old/server/masks/b/0000.png", b"b")
    indexed = archive_members(path, "masks")
    assert set(indexed) == {"a", "b"}
    assert indexed["a"][0]["member"] != indexed["b"][0]["member"]
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("another/masks/a/0000.png", b"duplicate")
    with pytest.raises(ValueError, match="collision"):
        archive_members(path, "masks")
