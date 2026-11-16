"""Keep frozen thumbnail/receipt helpers usable by existing graph and scoring workflows."""

import base64
import io

from PIL import Image

from tools.level3_image_graph_v1.core import sha256, write
from tools.level3_qwen21_v1.core import bundle
from tools.level3_shared.preview import thumb
from tools.level3_shared.receipts import finish_run


def test_receipt_excludes_mutable_tracking(tmp_path):
    write(tmp_path / "results.json", [])
    (tmp_path / "wandb").mkdir()
    (tmp_path / "wandb/transport").write_text("x")
    finish_run(tmp_path, True)
    (tmp_path / "wandb/transport").write_text("changed")
    assert bundle(tmp_path, "results.json") == []


def test_shared_helpers_still_import_and_preserve_native_alpha(tmp_path):
    from tools.level3_qwen_scoring_v1 import runner as scores
    from tools.level3_qwen_scoring_v1 import viewer as score_view

    assert scores.finish_run is finish_run
    assert score_view.thumb is thumb
    source = tmp_path / "alpha.png"
    Image.new("RGBA", (80, 60), (255, 0, 0, 0)).save(source)
    original = sha256(source)
    preview = thumb(source)
    with Image.open(io.BytesIO(base64.b64decode(preview.split(",", 1)[1]))) as im:
        assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)
    assert sha256(source) == original
