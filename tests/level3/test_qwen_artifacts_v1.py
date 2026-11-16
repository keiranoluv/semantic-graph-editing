"""Keep frozen thumbnail/receipt helpers usable by existing graph and scoring workflows."""

import base64
import io

from PIL import Image

from tools.level3_image_graph_v1.core import read, sha256, write
from tools.level3_qwen21_v1.core import bundle
from tools.level3_qwen100_v1.receipt import finish_run
from tools.level3_qwen100_v1.viewer import render, thumb


def test_receipt_excludes_mutable_tracking(tmp_path):
    write(tmp_path / "results.json", [])
    (tmp_path / "wandb").mkdir()
    (tmp_path / "wandb/transport").write_text("x")
    finish_run(tmp_path, True)
    (tmp_path / "wandb/transport").write_text("changed")
    assert bundle(tmp_path, "results.json") == []


def test_legacy_renderer_never_autolabels(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "green").save(source)
    render([{"task_id": "t", "source_path": str(source), "prompt": "<script>bad</script>", "status": "ok"}], tmp_path)
    assert "<script>bad</script>" not in (tmp_path / "comparison.html").read_text()
    template = read(tmp_path / "review-template.json")
    assert template["reviews"][0]["edit_fulfillment"] is None


def test_shared_helpers_still_import_and_preserve_native_alpha(tmp_path):
    from tools.level3_qwen_graphs_v1 import runner as graphs
    from tools.level3_qwen_graphs_v1 import viewer as graph_view
    from tools.level3_qwen_review_v2 import thumb as review_thumb
    from tools.level3_qwen_scoring_v1 import runner as scores
    from tools.level3_qwen_scoring_v1 import viewer as score_view

    assert graphs.finish_run is scores.finish_run is finish_run
    assert graph_view.thumb is score_view.thumb is review_thumb is thumb
    source = tmp_path / "alpha.png"
    Image.new("RGBA", (80, 60), (255, 0, 0, 0)).save(source)
    original = sha256(source)
    preview = thumb(source)
    with Image.open(io.BytesIO(base64.b64decode(preview.split(",", 1)[1]))) as im:
        assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)
    assert sha256(source) == original
