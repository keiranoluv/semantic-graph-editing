"""CPU validation/alignment/diff of independently predicted MagicBrush graphs.

Bounding boxes here are MODEL PREDICTIONS in normalized xyxy_1000 coordinates,
not MagicBrush ground truth. Alignment and labels are pseudo annotations and
remain subject to human review. No editing instruction enters the matcher.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.editing.operations import Edge, EditGraph, Node, apply_operations, diff_aligned_graphs
from src.knowledge.transe.core import sha256
from tools.prepare_magicbrush import DATA, ROOT, digest, encode, fixed


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError("Invalid graph text field")
    return " ".join(value.casefold().split())


def validate_graph(raw):
    if not isinstance(raw, dict) or set(raw) != {"nodes", "edges"}:
        raise ValueError("Predicted graph requires only nodes and edges")
    if (
        not isinstance(raw["nodes"], list)
        or not isinstance(raw["edges"], list)
        or len(raw["nodes"]) > 20
        or len(raw["edges"]) > 60
    ):
        raise ValueError("Invalid or excessive graph elements")
    nodes, ids = [], set()
    for row in raw["nodes"]:
        if set(row) != {"id", "label", "attributes", "bbox"}:
            raise ValueError("Node requires id, label, attributes, bbox")
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 50 or identifier in ids:
            raise ValueError("Invalid/repeated predicted node ID")
        ids.add(identifier)
        box = row["bbox"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= 1000
            or not 0 <= box[1] < box[3] <= 1000
        ):
            raise ValueError("Predicted box must be positive-area normalized xyxy_1000")
        if not isinstance(row["attributes"], list) or len(row["attributes"]) > 8:
            raise ValueError("Attributes must be a short string list")
        nodes.append(
            {
                "id": identifier,
                "label": text(row["label"]),
                "attributes": sorted({text(a) for a in row["attributes"]}),
                "bbox": box,
            }
        )
    edges = set()
    for row in raw["edges"]:
        if (
            set(row) != {"subject", "predicate", "object"}
            or row["subject"] not in ids
            or row["object"] not in ids
            or row["subject"] == row["object"]
        ):
            raise ValueError("Invalid predicted edge endpoint/schema")
        edges.add((row["subject"], text(row["predicate"]), row["object"]))
    return {
        "nodes": sorted(nodes, key=lambda n: (n["bbox"], n["label"], n["attributes"], n["id"])),
        "edges": [{"subject": s, "predicate": p, "object": o} for s, p, o in sorted(edges)],
    }


def parse_graph(response):
    response = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.DOTALL)
    if fenced:
        response = fenced.group(1)
    return validate_graph(json.loads(response))


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / total if total else 0.0


def color_feature(image, box):
    w, h = image.size
    bounds = (
        math.floor(box[0] * w / 1000),
        math.floor(box[1] * h / 1000),
        math.ceil(box[2] * w / 1000),
        math.ceil(box[3] * h / 1000),
    )
    crop = np.asarray(image.crop(bounds).resize((24, 24)).convert("RGB"))
    histogram = np.concatenate([np.histogram(crop[:, :, c], bins=8, range=(0, 256))[0] for c in range(3)]).astype(
        np.float64
    )
    return histogram / max(float(np.linalg.norm(histogram)), 1e-12)


def align_graphs(source_raw, target_raw, source_image, target_image):
    """Hungarian match with unmatched dummies and ambiguity flags; instruction-free."""
    from scipy.optimize import linear_sum_assignment

    left, right = validate_graph(source_raw), validate_graph(target_raw)
    n, m = len(left["nodes"]), len(right["nodes"])
    costs = np.full((n, m), 5.0)
    lhs = [color_feature(source_image, node["bbox"]) for node in left["nodes"]]
    rhs = [color_feature(target_image, node["bbox"]) for node in right["nodes"]]
    for i, a in enumerate(left["nodes"]):
        for j, b in enumerate(right["nodes"]):
            overlap = iou(a["bbox"], b["bbox"])
            same = a["label"] == b["label"]
            similarity = float(np.clip(lhs[i] @ rhs[j], 0, 1))
            distance = math.hypot(
                (a["bbox"][0] + a["bbox"][2] - b["bbox"][0] - b["bbox"][2]) / 2000,
                (a["bbox"][1] + a["bbox"][3] - b["bbox"][1] - b["bbox"][3]) / 2000,
            )
            if overlap >= 0.2 or same and (distance <= 0.15 or similarity >= 0.85):
                cost = 0.6 * (1 - overlap) + 0.25 * (not same) + 0.15 * (1 - similarity) + 0.1 * min(distance, 1)
                if cost <= 0.62:
                    costs[i, j] = cost
    matrix = np.full((n + m, n + m), 5.0)
    matrix[:n, :m] = costs
    for i in range(n):
        matrix[i, m + i] = 0.45
    for j in range(m):
        matrix[n + j, j] = 0.45
    matrix[n:, m:] = 0
    assignments = list(zip(*linear_sum_assignment(matrix), strict=True)) if n + m else []
    matches, ambiguous = [], []
    for i, j in assignments:
        if i < n and j < m and costs[i, j] < 5:
            alternatives = [costs[i, k] for k in range(m) if k != j] + [costs[k, j] for k in range(n) if k != i]
            margin = min(alternatives, default=5.0) - costs[i, j]
            matches.append(
                {"source_index": int(i), "target_index": int(j), "cost": float(costs[i, j]), "margin": float(margin)}
            )
            if margin < 0.08:
                ambiguous.append(
                    {"source_id": left["nodes"][i]["id"], "target_id": right["nodes"][j]["id"], "margin": float(margin)}
                )
    source_ids = {node["id"]: f"n{i+1:03d}" for i, node in enumerate(left["nodes"])}
    target_ids = {
        right["nodes"][x["target_index"]]["id"]: source_ids[left["nodes"][x["source_index"]]["id"]] for x in matches
    }
    additions = [node for node in right["nodes"] if node["id"] not in target_ids]
    matched_source = {x["source_index"] for x in matches}
    unmatched_source = [node for i, node in enumerate(left["nodes"]) if i not in matched_source]
    possible_missed = sorted({node["label"] for node in unmatched_source} & {node["label"] for node in additions})
    for i, node in enumerate(additions):
        target_ids[node["id"]] = f"new{i+1:03d}"
    source_nodes = tuple(
        Node(source_ids[node["id"]], node["label"], tuple(node["attributes"])) for node in left["nodes"]
    )
    target_nodes = tuple(
        Node(target_ids[node["id"]], node["label"], tuple(node["attributes"])) for node in right["nodes"]
    )
    old_edges = sorted((source_ids[e["subject"]], e["predicate"], source_ids[e["object"]]) for e in left["edges"])
    source_edges = tuple(Edge(f"e{i+1:03d}", s, p, o) for i, (s, p, o) in enumerate(old_edges))
    new_triples = sorted((target_ids[e["subject"]], e["predicate"], target_ids[e["object"]]) for e in right["edges"])
    remaining_old = {(e.subject, e.predicate, e.object): e for e in source_edges}
    target_edges, remaining_new = [], []
    for triple in new_triples:
        if triple in remaining_old:
            target_edges.append(remaining_old.pop(triple))
        else:
            remaining_new.append(triple)
    for index, (s, p, o) in enumerate(remaining_new):
        candidates = [e for e in remaining_old.values() if (e.subject, e.object) == (s, o)]
        same_pair_count = sum((a, c) == (s, o) for a, _, c in remaining_new)
        if len(candidates) == same_pair_count == 1:
            edge = candidates[0]
            target_edges.append(Edge(edge.id, s, p, o))
            remaining_old.pop((edge.subject, edge.predicate, edge.object))
        else:
            target_edges.append(Edge(f"new-e{index+1:03d}", s, p, o))
    source = EditGraph(source_nodes, source_edges)
    target = EditGraph(target_nodes, tuple(target_edges))
    operations = diff_aligned_graphs(source, target)
    if apply_operations(source, operations) != target:
        raise AssertionError("Diff does not reconstruct target")
    source_boxes = {source_ids[node["id"]]: node["bbox"] for node in left["nodes"]}
    target_boxes = {target_ids[node["id"]]: node["bbox"] for node in right["nodes"]}
    return {
        "source_graph": source.to_dict(),
        "target_graph": target.to_dict(),
        "operations": [op.to_dict() for op in operations],
        "matches": matches,
        "ambiguous_matches": ambiguous,
        "possible_missed_correspondence_labels": possible_missed,
        "source_boxes": source_boxes,
        "target_boxes": target_boxes,
        "alignment_policy": "hungarian_geometry_0.6_label_0.25_RGB_0.15_center_0.1_threshold0.62_margin0.08_v1",
        "instruction_used_in_matching": False,
        "target_reconstruction_exact": True,
    }


def build(manifest_path, graph_dir, output):
    manifest = json.loads(manifest_path.read_text())
    records = []
    reasons = Counter()
    cache_manifest_path = graph_dir / "manifest.json"
    cache = json.loads(cache_manifest_path.read_text()) if cache_manifest_path.exists() else None
    if cache and cache["dataset_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Inference cache belongs to another dataset")
    for row in manifest["records"]:
        entry = {
            "id": row["id"],
            "split": row["split"],
            "source_group_id": row["source_group_id"],
            "instruction": row["instruction"],
            "label_origin": "frozen_image_only_model_pseudo_annotations",
            "human_review_status": "pending",
            "eligible_for_training": False,
        }
        predictions = []
        for role in ("source_image", "target_image"):
            path = graph_dir / f"{row[role]['sha256']}.json"
            if not path.exists():
                break
            prediction = json.loads(path.read_text())
            if (
                prediction.get("image_sha256") != row[role]["sha256"]
                or not cache
                or prediction.get("inference_identity_sha256") != digest(cache)
            ):
                raise ValueError("Prediction image/model identity mismatch")
            if prediction.get("status") != "PASS":
                break
            predictions.append(prediction)
        if len(predictions) != 2:
            entry.update(status="pending_graphs", reason="missing_or_invalid_image_graph")
            reasons[entry["reason"]] += 1
        else:
            images = []
            for role in ("source_image", "target_image"):
                path = ROOT / row[role]["path"]
                if sha256(path) != row[role]["sha256"]:
                    raise ValueError("Source image bytes changed")
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            aligned = align_graphs(predictions[0]["graph"], predictions[1]["graph"], *images)
            problems = []
            if aligned["ambiguous_matches"]:
                problems.append("ambiguous_instance_alignment")
            if aligned["possible_missed_correspondence_labels"]:
                problems.append("possible_missed_instance_correspondence")
            if not aligned["operations"]:
                problems.append("no_represented_change")
            if len(aligned["operations"]) > 12:
                problems.append("excessive_graph_drift")
            if re.search(
                r"\b(watercolou?r|oil painting|black and white|sepia|cartoon style|entire image.*style)\b",
                row["instruction"],
                re.I,
            ):
                problems.append("global_style_outside_graph_schema")
            entry.update(
                **aligned,
                status="candidate" if not problems else "needs_review",
                problems=problems,
                eligible_for_training=not problems,
                source_prediction_sha256=sha256(graph_dir / f"{row['source_image']['sha256']}.json"),
                target_prediction_sha256=sha256(graph_dir / f"{row['target_image']['sha256']}.json"),
            )
            for problem in problems:
                reasons[problem] += 1
        records.append(entry)
    result = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(manifest_path),
        "inference_manifest_sha256": sha256(cache_manifest_path) if cache else None,
        "records": records,
        "candidate_pairs": sum(r["eligible_for_training"] for r in records),
        "pending_graph_pairs": sum(r["status"] == "pending_graphs" for r in records),
        "problem_counts": dict(reasons),
        "human_reviews_completed": 0,
        "minimum_valid_pairs": 300,
        "ready_for_main_training": False,
        "interpretation": "Pseudo labels require random train human review; candidate counts are not verified ground-truth counts.",
    }
    if cache:
        source = Path(__file__).resolve().parents[1] / "src/editing/operations.py"
        result["builder_source_sha256"] = {str(p): sha256(p) for p in (Path(__file__), source)}
    fixed(output, encode(result))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    parser.add_argument("--graphs", type=Path, default=DATA / "graphs-qwen25vl3b-bs1")
    parser.add_argument("--output", type=Path, default=DATA / "edits-v1.json")
    args = parser.parse_args(argv)
    build(args.manifest, args.graphs, args.output)


if __name__ == "__main__":
    main()
