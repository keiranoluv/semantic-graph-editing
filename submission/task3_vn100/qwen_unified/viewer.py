"""One review/demo for 200 requests; cache one preview per image and hard-link native files."""

import io
import os
from pathlib import Path

from PIL import Image, ImageOps

from task3_vn100.image_graph.core import digest, read, sha256

from task3_vn100.qwen_unified.core import atomic


def render(out, cfg, tasks, state):
    cache_path = out / "preview-cache.json"
    cache = read(cache_path) if cache_path.exists() else {}

    def asset(path, checksum):
        path = Path(path)
        key = digest([checksum, cfg["preview_policy"]])
        preview = out / "media" / (key + ".jpg")
        native = out / "native" / (checksum + path.suffix.lower())
        preview.parent.mkdir(exist_ok=True)
        native.parent.mkdir(exist_ok=True)
        if not native.exists():
            os.link(path, native)  # no second allocation for native image data
        elif not os.path.samefile(path, native) and sha256(native) != checksum:
            raise ValueError("native_link_changed")
        if not preview.exists() or cache.get(key) != sha256(preview):
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGBA")
                im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
                im.thumbnail((960, 720))
                buffer = io.BytesIO()
                im.save(buffer, format="JPEG", quality=88)
            temp = preview.with_suffix(".tmp")
            temp.write_bytes(buffer.getvalue())
            os.replace(temp, preview)
            cache[key] = sha256(preview)
        return {"preview": str(preview.relative_to(out)), "native": str(native.relative_to(out))}

    cases = []
    known_sources = {
        e["image_id"]: (e["result"]["source_path"], e["result"]["source_sha256"])
        for e in state["entries"]
        if e["status"] == "ok"
    }
    for task, entry in zip(tasks, state["entries"], strict=True):
        if task["task_id"] != entry["task_id"]:
            raise ValueError("viewer_task_order_changed")
        ready = entry["status"] == "ok"
        source, source_hash = known_sources.get(task["image_id"], (task["source_path"], task["sha256"]))
        row = {
            "task_id": task["task_id"],
            "index": entry["index"],
            "image_id": task["image_id"],
            "prompt": task["instruction_en"],
            "kind": task["operation"]["kind"],
            "status": entry["status"],
            "binding": entry.get("binding"),
            "source": asset(source, source_hash),
            "target": None,
        }
        if ready:
            row["target"] = asset(entry["result"]["target_path"], entry["result"]["target"]["sha256"])
        cases.append(row)
    bindings = {c["task_id"]: c["binding"] for c in cases}
    legacy = [
        {"results_sha256": group["results_sha256"], "bindings": {tid: bindings[tid] for tid in group["task_ids"]}}
        for group in state["legacy_reviews"]
    ]
    atomic(cache_path, cache)
    atomic(
        out / "review-data.json",
        {
            "schema": "qwen-unified-view-v1",
            "cohort_sha256": cfg["manifest_sha256"],
            "successful": state["successful"],
            "cases": cases,
            "legacy": legacy,
        },
    )
    page = Path(__file__).with_name("viewer.html").read_text()
    temp = out / ".comparison.tmp"
    temp.write_text(page, encoding="utf-8")
    os.replace(temp, out / "comparison.html")
    atomic(
        out / "snapshot.json",
        {
            "schema": "qwen-unified-snapshot-v1",
            "execution_complete": state["execution_complete"],
            "requested": len(tasks),
            "successful": state["successful"],
            "files_sha256": {
                name: sha256(out / name)
                for name in ("run.json", "index.json", "results.json", "review-data.json", "comparison.html")
            },
            "case_receipts": {e["case_dir"]: e["case_receipt_sha256"] for e in state["entries"] if e["status"] == "ok"},
        },
    )
