import numpy as np
import pytest
from PIL import Image

from tools import level3_clip_crop_v1 as crop


@pytest.mark.parametrize("size", [(448, 224), (224, 448), (224, 224)])
def test_letterbox_preserves_borders_and_center_crop_visibility(size):
    image = Image.new("RGB", size, (255, 0, 0))
    view = crop.letterbox(image)
    assert view.size == (224, 224)
    assert view.getpixel((112, 112)) == (255, 0, 0)
    if size[0] == size[1]:
        assert np.array_equal(np.asarray(image), np.asarray(view))
        assert crop.cropped_roi_fraction(size, [0, 0, 1, 1]) == 0
    else:
        assert view.getpixel((0, 0)) == crop.PAD_RGB
        assert crop.cropped_roi_fraction(size, [0, 0, 1, 1]) == 0.5
    assert crop.cropped_roi_fraction(size, [0.4, 0.4, 0.6, 0.6]) == pytest.approx(0)


def test_paired_inference_summary_handles_ties_and_constant_subgroup(monkeypatch):
    monkeypatch.setattr(crop, "BOOTSTRAPS", 60)
    rows = [{"image_id": i // 2, "overall_pass": i >= 3, "center": -i, "letterbox": i} for i in range(6)]
    result = crop.compare(rows, "overall_pass")
    assert result["images"] == 3 and result["pairs"] == 6 and result["passes"] == 3
    assert result["delta_rho"] > 0 and result["paired_cluster_ci95"][0] > 0
    assert crop.compare(rows[:2], "overall_pass")["center_rho"] is None
    assert crop.compare([], "overall_pass")["paired_cluster_ci95"] is None
