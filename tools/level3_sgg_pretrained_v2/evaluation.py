"""Hash-bound item review. Recall needs a human-confirmed complete inventory."""

from collections import Counter

from tools.level3_image_graph_v1.core import digest, validate_record

MODELS = ("robin-sam", "react++", "relate-anything")


def template(comparison):
    rows = []
    for case in comparison["cases"]:
        for model, result in case["models"].items():
            r = result["record"]
            graph = r.get("source_graph") or {"nodes": [], "edges": []}
            rows.append(
                {
                    "candidate_id": case["candidate_id"],
                    "model": model,
                    "record_sha256": digest(r),
                    "status": "pending",
                    "reviewer": "",
                    "notes": "",
                    "inventory_complete": False,
                    "missing_objects": [],
                    "missing_relations": [],
                    "objects": {
                        n["id"]: {"correct": None, "grounded": None, "hallucinated": None, "reference_instance": None}
                        for n in graph["nodes"]
                    },
                    "relations": {
                        e["id"]: {"correct": None, "hallucinated": None, "reference_relation": None}
                        for e in graph["edges"]
                    },
                }
            )
    return {
        "schema": "level3-sgg-item-review-v2",
        "comparison_sha256": digest(comparison),
        "reviews": rows,
        "instructions": "Review ALL predicted items. correct requires semantic label/instance and bbox IoU>=0.5; "
        "relation correct requires both grounded endpoints, directed predicate. Repeated reference IDs cannot "
        "both be TP. Hallucination = visually unsupported, not merely a wrong label. Record unique reference "
        "IDs, list ALL missed objects/relations, then inventory_complete=true only after exhaustive review. "
        "Use null when uncertain. ROBIN descriptions require human semantic matching. Never use another model as GT.",
    }


def prf(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def evaluate(comparison, review=None):
    expected = template(comparison)
    if review is None:
        review = expected
    if review["schema"] != expected["schema"] or review["comparison_sha256"] != expected["comparison_sha256"]:
        raise ValueError("stale_review")
    index = {(r["candidate_id"], r["model"]): r for r in expected["reviews"]}
    keys = [(r["candidate_id"], r["model"]) for r in review["reviews"]]
    if len(keys) != len(set(keys)) or set(keys) != set(index):
        raise ValueError("review_coverage_mismatch")
    records = {(c["candidate_id"], m): v["record"] for c in comparison["cases"] for m, v in c["models"].items()}
    totals = {
        m: {
            "reviewed_images": 0,
            "complete_inventory_images": 0,
            "objects": Counter(),
            "relations": Counter(),
            "grounding": [],
            "hallucination": {"objects": [], "relations": []},
        }
        for m in MODELS
    }
    inventories = {}
    for row in review["reviews"]:
        key = (row["candidate_id"], row["model"])
        exp = index[key]
        rec = records[key]
        if row["record_sha256"] != exp["record_sha256"] or row["status"] not in ("pending", "reviewed"):
            raise ValueError("invalid_review_identity_or_status")
        if row["status"] == "pending":
            continue
        if rec.get("status") != "ok" or not row["reviewer"].strip() or not row["notes"].strip():
            raise ValueError("review_requires_prediction_reviewer_and_notes")
        graph = validate_record(rec)
        if type(row["inventory_complete"]) is not bool:
            raise ValueError("invalid_inventory_complete")
        t = totals[row["model"]]
        t["reviewed_images"] += 1
        complete = row["inventory_complete"]
        refs_by_kind = {}
        for kind, ref_key in [("objects", "reference_instance"), ("relations", "reference_relation")]:
            if set(row[kind]) != set(exp[kind]):
                raise ValueError("item_review_coverage_mismatch")
            refs = set()
            decisions = []
            for identifier, item in row[kind].items():
                if set(item) != set(exp[kind][identifier]):
                    raise ValueError("item_fields_mismatch")
                for field in ("correct", "hallucinated") + (("grounded",) if kind == "objects" else ()):
                    if item[field] is not None and type(item[field]) is not bool:
                        raise ValueError("review_boolean_required")
                if item["correct"] is True:
                    ref = item[ref_key]
                    if not isinstance(ref, str) or not ref.strip() or ref in refs:
                        raise ValueError("correct_items_need_unique_reference_ids")
                    refs.add(ref)
                    if item["hallucinated"] is not False or (kind == "objects" and item["grounded"] is not True):
                        raise ValueError("correct_item_cannot_be_ungrounded_or_hallucinated")
                decisions.append(item["correct"])
                if item["hallucinated"] is not None:
                    t["hallucination"][kind].append(item["hallucinated"])
                if kind == "objects" and item["grounded"] is not None:
                    t["grounding"].append(item["grounded"])
            missing = row["missing_" + kind]
            if not isinstance(missing, list) or any(not isinstance(x, str) or not x.strip() for x in missing):
                raise ValueError("missing_inventory_requires_reference_ids")
            if len(missing) != len(set(missing)) or refs.intersection(missing):
                raise ValueError("duplicate_or_already_matched_missing_reference")
            if complete and any(x is None for x in decisions):
                raise ValueError("complete_inventory_requires_decisions")
            refs_by_kind[kind] = refs | set(missing)
            if complete:
                t[kind].update(
                    tp=sum(x is True for x in decisions), fp=sum(x is False for x in decisions), fn=len(missing)
                )
        for edge in graph.edges:
            if row["relations"][edge.id]["correct"] is True and any(
                row["objects"][n]["correct"] is not True for n in (edge.subject, edge.object)
            ):
                raise ValueError("correct_relation_requires_correct_grounded_endpoints")
        if complete:
            cid = row["candidate_id"]
            if cid in inventories and inventories[cid] != refs_by_kind:
                raise ValueError("models_must_share_one_human_reference_inventory")
            inventories[cid] = refs_by_kind
            t["complete_inventory_images"] += 1
    result = {}
    for m in MODELS:
        rs = [c["models"][m]["record"] for c in comparison["cases"]]
        statuses = Counter(r.get("status", "not_run") for r in rs)
        attempted = sum(v for k, v in statuses.items() if k != "not_run")
        t = totals[m]

        def rate(xs):
            return {"rate": sum(xs) / len(xs) if xs else None, "reviewed_items": len(xs)}

        result[m] = {
            "status_counts": dict(statuses),
            "attempted_images": attempted,
            "parse_success": statuses["ok"] / attempted if attempted else None,
            "reviewed_images": t["reviewed_images"],
            "complete_inventory_images": t["complete_inventory_images"],
            "objects": prf(*(t["objects"][k] for k in ("tp", "fp", "fn"))) if t["complete_inventory_images"] else None,
            "relations": prf(*(t["relations"][k] for k in ("tp", "fp", "fn")))
            if t["complete_inventory_images"]
            else None,
            "grounding": rate(t["grounding"]),
            "hallucination": {k: rate(v) for k, v in t["hallucination"].items()},
            "prf_scope": "micro_over_exhaustively_reviewed_successful_images_only",
            "source_target_stability": None,
        }
    return {
        "schema": "level3-sgg-evaluation-v2",
        "models": result,
        "comparison_sha256": digest(comparison),
        "review_sha256": digest(review),
        "warning": "Source-only pilot cannot establish edit/preservation consistency. "
        "Partial reviews are not full-cohort metrics; report failures and review denominators.",
    }
