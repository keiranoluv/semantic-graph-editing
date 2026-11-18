"""Verify frozen annotations and build the browser worksheet for a new human audit."""

import argparse
import base64
import html
import io
import json
import random
from pathlib import Path

from PIL import Image

from experiments.magicbrush_prompt_pilot_v1.run import checked_json, example
from src.editing.operations import EditGraph
from tools.edit_model_v1.codec import decode_actions, encode_actions
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256

from .dataset import MANIFEST, rows

SEED = 20261119


def verify():
    manifest = read(MANIFEST)
    config = checked_json(manifest["parent_config"], manifest["parent_config_sha256"])
    selected = []
    media = {}
    groups = {}
    pixels = {}

    def artifact(key):
        item = config["artifacts"][key]
        return checked_json(item["path"], item["sha256"])["records"]

    for shard, splits in [("trainval", ["train", "val"]), ("test", ["test"])]:
        original = {r["id"]: r for r in artifact(shard + "_edits")}
        media.update({r["id"]: r for r in artifact(shard + "_manifest")})
        for split in splits:
            keep = rows(split)
            held = rows(split, held=True)
            ids = [r["id"] for r in keep + held]
            assert len(ids) == len(set(ids))
            assert set(ids) == {rid for rid, r in original.items() if r["split"] == split}
            assert all(r["reason"].strip() for r in held)
            groups[split] = {media[rid]["source_group_id"] for rid in ids}
            pixels[split] = {
                media[rid][role]["pixel_sha256"] for rid in ids for role in ["source_image", "target_image"]
            }
            for r in keep:
                old = original[r["id"]]
                for key in ["source_graph", "source_group_id", "instruction", "split"]:
                    assert r[key] == old[key]
                for role, fingerprint in r["images"].items():
                    assert media[r["id"]][role]["sha256"] == fingerprint
                baseline = example(r, r["graph_diff_operations"])
                assert baseline.operations == example(old, old["operations"]).operations
                assert baseline.target == EditGraph.from_dict(old["target_graph"])
                prompt = example(r, r["operations"])
                assert prompt.operations and prompt.source != prompt.target
                assert tuple(decode_actions(encode_actions(prompt.operations)[1:], prompt.source)) == prompt.operations
            selected.extend(keep)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert groups[a].isdisjoint(groups[b]) and pixels[a].isdisjoint(pixels[b])
    return selected, media


def picture(item):
    path = Path(item["path"])
    if sha256(path) != item["sha256"]:
        raise ValueError(f"Image changed: {path}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((520, 400))
        packed = io.BytesIO()
        im.save(packed, "JPEG", quality=86)
    return base64.b64encode(packed.getvalue()).decode("ascii")


def entity_text(v):
    if "label" in v:
        attrs = ", ".join(v["attributes"])
        return f"{v['id']} {v['label']}" + (f" [{attrs}]" if attrs else "")
    return f"{v['id']} ({v['subject']} — {v['predicate']} → {v['object']})"


def lines(items):
    return "<pre>" + html.escape("\n".join(items)) + "</pre>"


def worksheet(rows_to_check, media, path):
    parts = [
        """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra 20 nhãn MagicBrush</title>
<style>body{font:17px system-ui;max-width:1100px;margin:auto;padding:1rem;color:#182333}
article{border-top:1px solid #888;padding:1rem 0}figure{display:inline-block;width:46%;margin:1%}
img{max-width:100%;height:auto}pre{white-space:pre-wrap}label,select,input,button{font:inherit;margin:.4rem}
button{padding:.5em}output{display:block}</style><h1>Kiểm tra 20 nhãn MagicBrush</h1>
<p>Đọc prompt và đồ thị nguồn. Dùng hai ảnh để đối chiếu. Chọn Đạt khi các thao tác đúng đối tượng,
đủ các thay đổi mà prompt yêu cầu và không thêm thay đổi ngoài ý. Ghi lý do nếu chọn Sai hoặc Chưa rõ.
Quy ước riêng của mẫu bò ăn cà rốt cho phép dùng cạnh near khi đống cà rốt sát mõm bò.</p>
<label>Người duyệt <input id="reviewer"></label>"""
    ]
    for index, r in enumerate(rows_to_check, 1):
        rid = html.escape(r["id"])
        parts.append(
            f'<article data-id="{rid}"><h2>{index}. {html.escape(r["instruction"])}</h2><small>{rid}</small><div>'
        )
        for role, title in [("source_image", "Ảnh nguồn"), ("target_image", "Ảnh đích")]:
            content = picture(media[r["id"]][role])
            parts.append(
                f'<figure><figcaption>{title}</figcaption><img alt="{title}" src="data:image/jpeg;base64,{content}"></figure>'
            )
        graph = r["source_graph"]
        parts.extend(
            [
                "</div><h3>Đồ thị nguồn</h3>",
                lines(entity_text(v) for v in graph["nodes"] + graph["edges"]),
                "<h3>Thao tác đích theo prompt</h3>",
                lines(
                    o["kind"] + "   " + " → ".join(entity_text(v) for v in [o["before"], o["after"]] if v)
                    for o in r["operations"]
                ),
                '<label>Kết luận <select><option></option><option>Đạt</option><option>Sai</option><option>Chưa rõ</option></select></label><label>Ghi chú <input class="note"></label></article>',
            ]
        )
    parts.append(
        """<button id="save">Tải kết quả JSON</button><output id="status"></output><script>
const articles=[...document.querySelectorAll("article")];
document.querySelector("#save").onclick=()=>{
 const records=articles.map(a=>({id:a.dataset.id,verdict:a.querySelector("select").value,note:a.querySelector("input").value}));
 const passed=records.filter(r=>r.verdict==="Đạt").length;
 const reviewed=records.filter(r=>r.verdict).length;
 const payload={dataset:"magicbrush-prompt-intent-v1",manifest_sha256:"HASH",seed:SEED,
 reviewer:document.querySelector("#reviewer").value,reviewed,passed,agreement:reviewed===20?passed/20:null,records};
 const link=document.createElement("a");
 link.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));
 link.download="magicbrush-prompt-intent-review20.json";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
 document.querySelector("#status").textContent=reviewed===20?"Đạt "+passed+"/20":"Đã xuất bản nháp "+reviewed+"/20";
};</script></html>""".replace(
            "HASH", sha256(MANIFEST)
        ).replace(
            "SEED", str(SEED)
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    kept, media = verify()
    sample = random.Random(SEED).sample(sorted(kept, key=lambda r: r["id"]), 20)
    if args.html:
        worksheet(sample, media, args.html)
    print(json.dumps({"verified": len(kept), "seed": SEED, "review_ids": [r["id"] for r in sample]}, indent=2))
