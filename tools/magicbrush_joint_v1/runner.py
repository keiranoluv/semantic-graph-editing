"""Resumable joint-image annotation and content-addressed CPU compilation."""

import json
from collections import Counter
from pathlib import Path

from src.knowledge.transe.core import sha256
from tools.magicbrush_joint_v1.client import LocalClient
from tools.magicbrush_joint_v1.compiler import compile_annotation
from tools.magicbrush_joint_v1.prepare import check_run, image_check
from tools.magicbrush_joint_v1.schema import parse
from tools.magicbrush_joint_v1.storage import event, exclusive, read, sample_dir, write_once
from tools.prepare_magicbrush import ROOT, digest


def selected_ids(cfg, scope):
    run, manifest = check_run(cfg)
    if scope == "pilot":
        return run, manifest, run["pilot_ids"]
    expansion = read(ROOT / cfg["output"] / "expansion.json")
    if expansion["run_sha256"] != digest(run):
        raise ValueError("expansion_provenance_mismatch")
    from tools.magicbrush_joint_v1.review import audit

    gate = audit(cfg, Path(expansion["csv_path"]), Path(expansion["compilation_path"]))
    if not gate["ready"] or gate != expansion["review_gate"]:
        raise ValueError("expansion_human_gate_changed")
    expected = sorted(manifest["splits"]["train"] + manifest["splits"]["val"])
    if expansion["ids"] != expected:
        raise ValueError("expansion_cohort_changed")
    return run, manifest, expected


def parse_response(raw):
    if raw["error"]:
        raise ValueError(raw["error"])
    response = json.loads(raw["body"])
    choices = response.get("choices", []) if isinstance(response, dict) else []
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "stop"
    ):
        raise ValueError("incomplete_or_multiple_completion")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("missing_completion_content")
    return parse(content)


def attempt_result(raw):
    http_success = not raw["error"] and raw["http_status"] == 200
    successful = False
    json_valid = False
    try:
        # Report valid annotation JSON independently from its schema validity.
        response = json.loads(raw["body"]) if http_success else {}
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            content = message.get("content") if isinstance(message, dict) else None
            successful = (
                len(choices) == 1
                and choices[0].get("finish_reason") == "stop"
                and isinstance(content, str)
                and bool(content.strip())
            )
            json.loads(content)
            json_valid = True
        annotation = parse_response(raw)
        return {
            "http_success": http_success,
            "successful_inference": successful,
            "json_valid": json_valid,
            "schema_valid": True,
            "error": None,
        }, annotation
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return {
            "http_success": http_success,
            "successful_inference": successful,
            "json_valid": json_valid,
            "schema_valid": False,
            "error": str(error),
        }, None


def infer(cfg, *, scope="pilot", limit=None, client=None):
    run, manifest, ids = selected_ids(cfg, scope)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit_must_be_positive")
        ids = ids[:limit]
    out = ROOT / cfg["output"]
    index = {r["id"]: r for r in manifest["records"]}
    # A real server must have been started by the explicit launcher; mocks bypass transport only.
    receipt = read(out / "server.json") if client is None else {"mock_cpu_transport": True}
    if client is None and (receipt["config_sha256"] != digest(cfg) or receipt["run_sha256"] != digest(run)):
        raise ValueError("server_provenance_mismatch")
    client = client or LocalClient(cfg)
    checked = False
    with exclusive(out):
        write_once(out / "inference-backend.json", receipt)
        for identifier in ids:
            row = index[identifier]
            folder = sample_dir(out, identifier)
            folder.mkdir(parents=True, exist_ok=True)
            context = {"id": identifier, "record": row, "run_sha256": digest(run), "backend_sha256": digest(receipt)}
            write_once(folder / "input.json", context)
            if (folder / "final.json").exists():
                continue
            try:
                image_check(row)
            except (OSError, ValueError) as error:
                write_once(folder / "final.json", {"status": "input_error", "error": str(error), "attempts": 0})
                event(out / "events.jsonl", {"id": identifier, "event": "input_error", "error": str(error)})
                continue
            for number in range(1, cfg["http"]["max_attempts"] + 1):
                prefix = folder / f"attempt-{number:03d}"
                started = prefix.with_suffix(".started.json")
                raw_path = prefix.with_suffix(".raw.json")
                result_path = prefix.with_suffix(".result.json")
                if result_path.exists():
                    result = read(result_path)
                    annotation = parse_response(read(raw_path)) if result["schema_valid"] else None
                else:
                    if raw_path.exists():
                        raw = read(raw_path)  # Crash after response storage: parse it without another request.
                    elif started.exists():
                        raw = {
                            "http_status": None,
                            "body": "",
                            "error": "interrupted_inference_unknown_server_completion",
                        }
                        write_once(raw_path, raw)
                    else:
                        if not checked:
                            client.check_model()
                            checked = True
                        write_once(started, {"id": identifier, "attempt": number, "context_sha256": digest(context)})
                        # A KeyboardInterrupt leaves started evidence; the budget is still consumed.
                        raw = client.annotate(row)
                        write_once(raw_path, raw)
                    result, annotation = attempt_result(raw)
                    result["raw_sha256"] = sha256(raw_path)
                    write_once(result_path, result)
                    event(out / "events.jsonl", {"id": identifier, "attempt": number, **result})
                if sha256(raw_path) != result["raw_sha256"]:
                    raise ValueError("raw_response_evidence_changed")
                if annotation is not None:
                    write_once(folder / "annotation.json", annotation)
                    write_once(
                        folder / "final.json",
                        {
                            "status": "completed",
                            "attempts": number,
                            "annotation_sha256": digest(annotation),
                            "result": result,
                        },
                    )
                    break
            else:
                write_once(
                    folder / "final.json",
                    {
                        "status": "inference_failed",
                        "error": result["error"],
                        "attempts": cfg["http"]["max_attempts"],
                        "result": result,
                    },
                )
    return {"selected": len(ids), "completed_or_failed": sum((sample_dir(out, i) / "final.json").exists() for i in ids)}


def compile_run(cfg, *, scope="pilot"):
    run, manifest, ids = selected_ids(cfg, scope)
    out = ROOT / cfg["output"]
    index = {r["id"]: r for r in manifest["records"]}
    records, reason_counts, operations, states = [], Counter(), Counter(), Counter()
    attempted = inference_count = json_count = schema_count = graph_count = transaction_count = 0
    backend = read(out / "inference-backend.json") if (out / "inference-backend.json").exists() else None
    for identifier in ids:
        row, folder = index[identifier], sample_dir(out, identifier)
        entry = {
            **{k: row[k] for k in ("id", "split", "source_group_id", "instruction", "source_image", "target_image")},
            "label_origin": "joint_local_vlm_edit_focused_pseudo_annotation_v1",
            "human_review_status": "pending",
            "visual_reliability": "unverified",
            "eligible_for_training": False,
            "status": "pending",
            "classification": "needs_review",
            "problems": ["inference_pending"],
            "validation": {"schema_valid": False, "graph_compiled": False, "transaction_valid": False},
            "run_sha256": digest(run),
            "backend_sha256": digest(backend) if backend else None,
        }
        if (folder / "input.json").exists():
            if read(folder / "input.json") != {
                "id": identifier,
                "record": row,
                "run_sha256": digest(run),
                "backend_sha256": digest(backend),
            }:
                raise ValueError("sample_input_provenance_changed")
            attempts = list(folder.glob("attempt-*.started.json"))
            attempted += bool(attempts)
            for role in ("source_image", "target_image"):
                # Missing input remains a reported failure; changed bytes invalidate evidence.
                path = ROOT / row[role]["path"]
                if path.exists() and sha256(path) != row[role]["sha256"]:
                    raise ValueError("image_evidence_changed")
            attempt_rows = []
            for path in sorted(folder.glob("attempt-*.result.json")):
                result = read(path)
                raw_path = path.with_name(path.name.replace(".result.json", ".raw.json"))
                if sha256(raw_path) != result["raw_sha256"]:
                    raise ValueError("raw_response_evidence_changed")
                attempt_rows.append(result)
            inference_count += any(r["successful_inference"] for r in attempt_rows)
            json_count += any(r["json_valid"] for r in attempt_rows)
            schema_count += any(r["schema_valid"] for r in attempt_rows)
            entry["attempt_evidence"] = attempt_rows
            if (folder / "final.json").exists():
                final = read(folder / "final.json")
                entry["inference_result"] = final
                if final["status"] != "completed":
                    entry.update(
                        status="rejected", classification="rejected", problems=[final["status"] + ":" + final["error"]]
                    )
                else:
                    annotation = read(folder / "annotation.json")
                    raw_path = folder / f"attempt-{final['attempts']:03d}.raw.json"
                    if digest(annotation) != final["annotation_sha256"] or parse_response(read(raw_path)) != annotation:
                        raise ValueError("normalized_annotation_evidence_changed")
                    entry["validation"]["schema_valid"] = True
                    entry["annotation"] = annotation
                    try:
                        compiled = compile_annotation(annotation, cfg["limits"])
                    except ValueError as error:
                        entry.update(status="needs_review", problems=["graph_compilation:" + str(error)])
                    else:
                        entry.update(compiled)
                        graph_count += 1
                        transaction_count += 1
                        operations.update(op["kind"] for op in entry["operations"])
        records.append(entry)
        states[entry["classification"]] += 1
        reason_counts.update(entry["problems"])
    total = len(ids)
    summary = {
        "selected_pairs": total,
        "number_attempted": attempted,
        "successful_inference_count": inference_count,
        "json_valid_count": json_count,
        "schema_valid_count": schema_count,
        "graph_compilation_count": graph_count,
        "transaction_valid_count": transaction_count,
        "graph_compilation_rate_per_selected": graph_count / total if total else None,
        "transaction_valid_rate_per_selected": transaction_count / total if total else None,
        "classification_counts": dict(states),
        "operation_distribution_all_compiled": dict(operations),
        "failure_review_reasons": dict(reason_counts),
        "human_reviews_completed": 0,
        "human_verified_accuracy": None,
        "model_confidence_is_calibrated": False,
        "minimum_usable_pairs": 300,
        "dataset_requirement_complete": False,
        "interpretation": "Structural validation is not visual correctness. Candidate acceptance is not human approval. Counts include failures without replacement; human evidence is audited separately.",
    }
    export = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(out / "manifest.json"),
        "run_sha256": digest(run),
        "scope": scope,
        "records": records,
        "summary": summary,
        "ready_for_main_training": False,
    }
    key = digest(export)
    with exclusive(out):
        path = out / "compilations" / f"{key}.json"
        write_once(path, export)
        write_once(out / "reports" / f"{key}.json", summary)
    return path, summary
