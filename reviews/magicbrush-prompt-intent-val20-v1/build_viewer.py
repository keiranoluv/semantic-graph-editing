"""Build an offline viewer of the fixed validation cohort and proposed edits."""

import argparse
import base64
import hashlib
import io
import json
import math
from html import escape
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def photo(meta, title):
    raw = (ROOT / meta["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise ValueError(f"Ảnh sai checksum: {meta['path']}")
    with Image.open(io.BytesIO(raw)) as im:
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=90)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><figcaption>{title}</figcaption><img alt="{title}" src="data:image/jpeg;base64,{data}"></figure>'


def entity(obj):
    if "label" in obj:
        parts = [obj["id"], obj["label"], *obj["attributes"]]
    else:
        parts = [obj["id"], obj["subject"], obj["predicate"], "→", obj["object"]]
    return escape(" · ".join(parts))


def op_list(operations, decisions=()):
    if not operations:
        return "<p>Chưa có một tập thao tác duy nhất.</p>"
    rows = []
    for i, op in enumerate(operations):
        content = " ⇒ ".join(entity(v) for v in (op["before"], op["after"]) if v)
        decision = f" <em>{escape(decisions[i])}</em>" if decisions else ""
        rows.append(f'<li><b>{escape(op["kind"])}</b>{decision}<br>{content}</li>')
    return "<ol>" + "".join(rows) + "</ol>"


def graph(raw, case_nr):
    nodes, edges = raw["nodes"], raw["edges"]
    n = len(nodes)
    radius = max(95, n * 21)
    size = 2 * radius + 180
    center = size / 2
    points = {}
    out = [
        f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="Sơ đồ đồ thị nguồn">'
        f'<defs><marker id="arrow{case_nr}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#617588"/></marker></defs>'
    ]
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n - math.pi / 2
        r = radius if n > 1 else 0
        points[node["id"]] = (center + r * math.cos(angle), center + r * math.sin(angle))
    for e in edges:
        x, y = points[e["subject"]]
        end_x, end_y = points[e["object"]]
        angle = math.atan2(end_y - y, end_x - x)
        dx, dy = 23 * math.cos(angle), 23 * math.sin(angle)
        out.append(
            f'<line x1="{x+dx}" y1="{y+dy}" x2="{end_x-dx}" y2="{end_y-dy}" stroke="#8095a5" stroke-width="1.8" marker-end="url(#arrow{case_nr})"><title>{entity(e)}</title></line>'
        )
    for node in nodes:
        x, y = points[node["id"]]
        out.append(
            f'<g><title>{entity(node)}</title><circle cx="{x}" cy="{y}" r="21" fill="#e2f0ed" stroke="#22705d"/><text x="{x}" y="{y+4}">{escape(node["id"])}</text><text class="node-name" x="{x}" y="{y+34}">{escape(node["label"])}</text></g>'
        )
    out.append("</svg>")
    objects = "".join(f"<li>{entity(v)}</li>" for v in nodes)
    relations = "".join(f"<li>{entity(v)}</li>" for v in edges)
    return (
        "".join(out)
        + f"<div><h3>Đỉnh ({n})</h3><ul>{objects}</ul><h3>Cạnh ({len(edges)})</h3><ul>{relations}</ul></div>"
    )


def build(output):
    proposal = json.loads((HERE / "proposals.json").read_bytes())
    source = (ROOT / proposal["source_path"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != proposal["source_sha256"]:
        raise ValueError("Nguồn graph khác phiên bản đã duyệt")
    cases = {c["id"]: c for c in json.loads(source)["cases"]}
    cards, options = [], []
    assert len(cases) == len(proposal["records"]) == 20
    for i, record in enumerate(proposal["records"], 1):
        c = cases[record["id"]]
        assert record["prompt"] == c["instruction"]
        title = escape(record["prompt"])
        ident = escape(record["id"])
        options.append(f'<option value="{i}">{i:02} · {title}</option>')
        alternatives = "".join(
            f"<h4>Phương án khác {j}</h4>{op_list(ops)}" for j, ops in enumerate(record["alternative_operations"], 1)
        )
        cards.append(
            f"""<article id="case-{i}"><small>{ident} · Validation · {i}/20</small><h1>{title}</h1>
<div class="pair">{photo(c["source_image"],"Ảnh nguồn")}{photo(c["target_image"],"Ảnh sau chỉnh sửa")}</div>
<h2>Đồ thị nguồn</h2><div class="pair graph">{graph(c["source_graph"], i)}</div>
<h2>Thao tác đích</h2><div class="pair"><section><h3>Nhãn gốc và quyết định đề xuất</h3>{op_list(c["expected_operations"], record["original_operation_decisions"])}</section>
<section><h3>Đề xuất theo prompt</h3><p><b>{escape(record["status"])}</b></p><p>{escape(record["justification"])}</p>{op_list(record["candidate_operations"])}{alternatives}<p>{escape(record["scope_note"])}</p></section></div></article>"""
        )
    template = (HERE / "viewer-template.html").read_text()
    output.write_text(template.replace("<!--options-->", "".join(options)).replace("<!--cards-->", "".join(cards)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
