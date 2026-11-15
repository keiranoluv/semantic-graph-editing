"""Local diagnostic scores and separately attributed assistant observations."""

import json
from pathlib import Path

from tools.level3_image_graph_v1.core import read
from tools.level3_qwen100_v1.viewer import thumb


def render(results, out, review=None):
    cases = []
    reviews = {r["task_id"]: r for r in review["reviews"]} if review else {}
    for row in results:
        case = {k: row[k] for k in ("task_id", "image_id", "dataset", "prompt", "operation", "grounding", "scores")}
        case["roles"] = {}
        for role, item in row["roles"].items():
            record = read(item["graph_path"])
            case["roles"][role] = {
                "preview": thumb(item["input"]["image_path"]),
                "image": record["image"],
                "graph": record["source_graph"],
                "grounding": record["grounding"],
            }
        case["assistant_review"] = reviews.get(row["task_id"])
        cases.append(case)
    data = json.dumps(cases, ensure_ascii=False)
    for char, escaped in [("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")]:
        data = data.replace(char, escaped)
    template = Path(__file__).with_name("viewer.html").read_text()
    (out / "comparison.html").write_text(template.replace("__SCORE_DATA__", data), encoding="utf-8")
