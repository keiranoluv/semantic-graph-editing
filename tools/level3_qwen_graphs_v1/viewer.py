"""Self-contained, paginated source/target images, boxes and directed graphs."""

import json
from pathlib import Path

from tools.level3_image_graph_v1.core import read
from tools.level3_qwen100_v1.viewer import thumb


def render(results, output):
    cases = []
    for pair in results:
        item = {k: pair[k] for k in ("task_id", "image_id", "dataset", "prompt")}
        item["roles"] = {}
        for role, r in pair["roles"].items():
            rec = read(r["graph_path"]) if r.get("graph_path") else {}
            item["roles"][role] = {
                "preview": thumb(r.get("input", {}).get("image_path", r["native_image_path"])),
                "status": r["status"],
                "error": rec.get("error", r.get("error")),
                "image": rec.get("image", r.get("input", {})),
                "graph": rec.get("source_graph"),
                "grounding": rec.get("grounding"),
            }
        cases.append(item)
    data = json.dumps(cases, ensure_ascii=False)
    for a, b in [("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")]:
        data = data.replace(a, b)
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    (output / "comparison.html").write_text(template.replace("__GRAPH_DATA__", data), encoding="utf-8")
