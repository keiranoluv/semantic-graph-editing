"""One CPU REACT++ qualitative pilot on the pre-existing 20 validation cases."""
import csv
import hashlib
import html
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from src.editing.operations import Edge, EditGraph, Node, apply_operations, diff_aligned_graphs
from tools.level3_qwen_graphs_v1.core import verify_react
from tools.level3_sgg_pretrained_v2.backends import React

OUT = Path("reports/level2/magicbrush-react-pilot20-v1")
CASES = Path("reports/level2/magicbrush-manual-v3-error-analysis-v1/cases.json")


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def overlap(a, b):
    xy = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    inter = np.prod(xy)
    return inter / max(1e-12, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter)


def alignment(a, b):
    def boxes(g):
        return [np.array(o["bbox_xyxy"]) / (g["size"] * 2) for o in g["objects"]]

    aa, bb = boxes(a), boxes(b)
    weights = np.zeros((len(aa), len(bb)))
    for i, x in enumerate(aa):
        for j, y in enumerate(bb):
            iou = overlap(x, y)
            if a["objects"][i]["label"] == b["objects"][j]["label"] and iou >= 0.5:
                weights[i, j] = iou
    ii, jj = linear_sum_assignment(-weights)
    return [(int(i), int(j), weights[i, j]) for i, j in zip(ii, jj, strict=False) if weights[i, j]]


def graph(g, ids, edge_ids):
    nodes = tuple(Node(i, o["label"]) for i, o in zip(ids, g["objects"], strict=True))
    edges = []
    for s, p, o in dict.fromkeys((ids[r["subject"]], r["predicate"], ids[r["object"]]) for r in g["relations"]):
        key = (s, p, o)
        ident = edge_ids.setdefault(key, f"e{len(edge_ids):03}")
        edges.append(Edge(ident, s, p, o))
    return EditGraph(nodes, tuple(edges))


def main():
    cases = json.loads(CASES.read_text())["cases"]
    assert len(cases) == len({r["id"] for r in cases}) == 20
    cfg = json.loads(Path("configs/level3/qwen-sgg-unified-v1.json").read_text())["react"]
    OUT.mkdir(exist_ok=True, parents=True)
    write(
        OUT / "cohort.json",
        dict(
            ids=[r["id"] for r in cases],
            source_sha256=hashlib.sha256(CASES.read_bytes()).hexdigest(),
            react=cfg,
            device="cpu",
            alignment="same label, maximum total normalized box IoU, eligible IoU >= 0.5; edges matched by aligned triple",
        ),
    )
    verify_react()
    esc = html.escape
    pages = [
        "<meta charset=utf-8><title>MagicBrush · REACT++</title><style>body{font:16px system-ui;max-width:1100px;margin:auto}img{max-width:48%;height:290px;object-fit:contain}pre{white-space:pre-wrap}</style><h1>20 cặp MagicBrush · kiểm tay REACT++</h1><p>Ảnh nguồn bên trái, ảnh đích bên phải. Ghi nhận khả năng biểu diễn yêu cầu, ghép sai hoặc bỏ sót, thao tác đúng và nhiễu vào review.csv theo ID. Các thao tác tự động là đề xuất để kiểm tra.</p>"
    ]
    images = {}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = React(Path("data/cache/level3-sgg-pretrained-v2"), "cpu", tmp, cfg)
        for case in cases:
            pair = []
            for role in ("source", "target"):
                m = case[role + "_image"]
                path = Path(m["path"])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"]
                if m["sha256"] not in images:
                    im = Image.open(path).convert("RGBA")
                    bg = Image.new("RGBA", im.size, "white")
                    bg.alpha_composite(im)
                    bg.convert("RGB").save(tmp / "image.png")
                    obj, rel, _ = model.predict(tmp / "image.png", tmp / "native.npz")
                    images[m["sha256"]] = dict(path=str(path), size=list(im.size), objects=obj, relations=rel)
                pair.append(images[m["sha256"]])
            a, b = pair
            matches = alignment(a, b)
            ids = [f"s{i:03}" for i in range(len(a["objects"]))]
            target = [f"t{i:03}" for i in range(len(b["objects"]))]
            for i, j, _ in matches:
                target[j] = ids[i]
            edge_ids = {}
            gs, gt = graph(a, ids, edge_ids), graph(b, target, edge_ids)
            ops = diff_aligned_graphs(gs, gt)
            assert apply_operations(gs, ops) == gt
            data = dict(
                id=case["id"],
                instruction=case["instruction"],
                images=[case[k + "_image"]["sha256"] for k in ("source", "target")],
                matches=matches,
                source=gs.to_dict(),
                target=gt.to_dict(),
                operations=[o.to_dict() for o in ops],
                prior_annotation=case["expected_operations"],
            )
            rows.append(data)
            pages.append(f'<h2>{esc(case["id"])} · {esc(case["instruction"])}</h2>')
            for g in pair:
                pages.append(f'<img src="../../../{esc(g["path"])}">')
            for title, key in [
                ("Thao tác từ annotation cũ", "prior_annotation"),
                ("Thao tác từ REACT++", "operations"),
                ("Ghép node (index nguồn, đích, IoU)", "matches"),
                ("Graph nguồn", "source"),
                ("Graph đích", "target"),
            ]:
                pages.append(
                    f"<details><summary>{title} ({len(data[key])})</summary><pre>{esc(json.dumps(data[key],ensure_ascii=False,indent=2))}</pre></details>"
                )
        model.close()
    write(OUT / "images.json", images)
    write(OUT / "pairs.json", rows)
    (OUT / "index.html").write_text("\n".join(pages))
    if not (OUT / "review.csv").exists():
        with (OUT / "review.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "instruction",
                    "edit_visible_in_graphs",
                    "identity_errors",
                    "supported_operations",
                    "noise_or_missing",
                    "reviewer",
                ]
            )
            for r in rows:
                w.writerow([r["id"], r["instruction"], "", "", "", "", ""])
    print(len(rows), "pairs,", len(images), "distinct images")


if __name__ == "__main__":
    main()
