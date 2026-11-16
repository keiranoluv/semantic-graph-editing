"""One graph viewer for 200 pairs; image previews and native arrays shared by hash."""

import os
from pathlib import Path

from PIL import Image, ImageOps

from tools.level3_image_graph_v1.core import read, sha256
from tools.level3_qwen_unified.core import atomic


def render(pairs, images, out):
    for folder in ("media", "native", "native-output"):
        (out / folder).mkdir(exist_ok=True)
    cache = out / "preview-cache.json"
    hashes = read(cache) if cache.exists() else {}
    assets = {}
    for checksum, item in images.items():
        source = Path(item["native_path"])
        native = out / "native" / (checksum + source.suffix)
        if not native.exists():
            os.link(source, native)
        elif not os.path.samefile(source, native) and sha256(native) != checksum:
            raise ValueError("viewer_native_image_changed")
        preview = out / "media" / (checksum + ".jpg")
        if not preview.exists() or hashes.get(checksum) != sha256(preview):
            with Image.open(source) as im:
                im = ImageOps.exif_transpose(im).convert("RGBA")
                im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
                im.thumbnail((960, 720))
                im.save(preview, quality=88)
            hashes[checksum] = sha256(preview)
        with Image.open(source) as im:
            dimensions = {"width": im.width, "height": im.height}
        a = {
            "preview": str(preview.relative_to(out)),
            "native_image": str(native.relative_to(out)),
            "image": dimensions,
        }
        if item["status"] == "ok":
            pred = item["prediction"]
            raw = out / "native-output" / (checksum + ".npz")
            if not raw.exists():
                os.link(pred["raw_path"], raw)
            elif not os.path.samefile(pred["raw_path"], raw) and sha256(raw) != pred["raw_sha256"]:
                raise ValueError("viewer_native_output_changed")
            a["raw_npz"] = str(raw.relative_to(out))
        assets[checksum] = a
    cases = []
    for pair in pairs:
        case = {k: pair[k] for k in ("task_id", "image_id", "dataset", "prompt")}
        case["roles"] = {}
        for role, item in pair["roles"].items():
            rec = read(item["graph_path"]) if item.get("graph_path") else {}
            case["roles"][role] = {
                **assets[item["native_sha256"]],
                "status": item["status"],
                "image": rec.get("image", assets[item["native_sha256"]]["image"]),
                "graph": rec.get("source_graph"),
                "grounding": rec.get("grounding"),
                "graph_json": str(Path(item["graph_path"]).relative_to(out)) if item.get("graph_path") else None,
            }
        cases.append(case)
    atomic(cache, hashes)
    atomic(
        out / "review-data.json",
        {
            "cases": cases,
            "unique_images": len(images),
            "ready_images": sum(i["status"] == "ok" for i in images.values()),
            "ready_pairs": sum(all(r["status"] == "ok" for r in p["roles"].values()) for p in pairs),
        },
    )
    temp = out / ".comparison.tmp"
    temp.write_text(Path(__file__).with_name("viewer.html").read_text())
    os.replace(temp, out / "comparison.html")
