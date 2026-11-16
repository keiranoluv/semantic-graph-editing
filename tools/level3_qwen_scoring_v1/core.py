"""Immutable inputs and source-only intent bindings; no human outcomes are loaded."""

import subprocess
from pathlib import Path

from tools.level3_image_graph_v1.core import ROOT, read, sha256, validate_record
from tools.level3_qwen21_v1.core import bundle

CONFIG = ROOT / "configs/level3/qwen-graph-score-v1.json"
RUNS = ROOT / "runs/level3/qwen-graph-score-v1"


def inputs():
    cfg = read(CONFIG)
    graph_run, qwen_run = ROOT / cfg["graph_run"], ROOT / cfg["qwen_run"]
    for root, key in [(graph_run, "graph_run_receipt_sha256"), (qwen_run, "qwen_run_receipt_sha256")]:
        if sha256(root / "receipt.json") != cfg[key]:
            raise ValueError("input_run_receipt_changed")
    for name in ("manifest", "grounding"):
        if sha256(ROOT / cfg[name]) != cfg[name + "_sha256"]:
            raise ValueError("frozen_" + name + "_changed")
    if cfg["device"] != "cpu" or cfg["training"] or cfg["read_human_outcomes"] or cfg["expected_pairs"] != 100:
        raise ValueError("fixed_scoring_protocol_changed")
    pairs = bundle(graph_run, "results.json")
    edited = bundle(qwen_run, "results.json")
    tasks = read(ROOT / cfg["manifest"])["tasks"]
    bindings = read(ROOT / cfg["grounding"])
    if (
        bindings["instruction_manifest_sha256"] != cfg["manifest_sha256"]
        or bindings["graph_run_receipt_sha256"] != cfg["graph_run_receipt_sha256"]
    ):
        raise ValueError("grounding_parent_mismatch")
    ids = [r["task_id"] for r in pairs]
    if (
        len(ids) != 100
        or len(set(ids)) != 100
        or any([r["task_id"] for r in x] != ids for x in [edited, tasks, bindings["tasks"]])
    ):
        raise ValueError("pair_task_binding_mismatch")
    records = []
    for pair, task, binding, image in zip(pairs, tasks, bindings["tasks"], edited, strict=True):
        if pair["prompt"] != task["instruction_en"] or pair["prompt"] != image["prompt"]:
            raise ValueError("prompt_changed")
        if binding["author"] != "assistant_source_visual_grounding" or binding["human_ground_truth"]:
            raise ValueError("incorrect_grounding_authorship")
        region = binding["region"]
        if len(region) != 4 or not all(0 <= x <= 1 for x in region) or region[0] >= region[2] or region[1] >= region[3]:
            raise ValueError("invalid_source_grounded_region")
        rec = {}
        for role in ("source", "target"):
            item = pair["roles"][role]
            path = Path(item["graph_path"])
            if graph_run.resolve() not in path.resolve().parents or item["status"] != "ok":
                raise ValueError("invalid_graph_pair")
            rec[role] = read(path)
            validate_record(rec[role])
            if rec[role]["image_role"] != role or rec[role]["pair_task_id"] != task["task_id"]:
                raise ValueError("graph_role_mismatch")
            if item["native_image_path"] != image[role + "_path"]:
                raise ValueError("graph_Qwen_image_mismatch")
            if role == "source" and sha256(path) != binding["source_graph_sha256"]:
                raise ValueError("source_binding_stale")
        records.append({"pair": pair, "task": task, "grounding": binding, **rec})
    return cfg, records


def new_output(path):
    path = Path(path).resolve()
    if RUNS.resolve() not in path.parents:
        raise ValueError("output_outside_scoring_runs")
    path.mkdir(parents=True, exist_ok=False)
    return path


def identity():
    cfg = read(CONFIG)
    files = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.html"))
    files += [
        CONFIG,
        ROOT / cfg["grounding"],
        ROOT / cfg["manifest"],
        ROOT / "scripts/run_level3_qwen_scoring_v1.sh",
        ROOT / "tools/level3_image_graph_v1/core.py",
        ROOT / "tools/level3_qwen21_v1/core.py",
        ROOT / "tools/level3_shared/preview.py",
        ROOT / "tools/level3_shared/receipts.py",
        ROOT / "tools/level3_shared/geometry.py",
        ROOT / "src/editing/operations.py",
        ROOT / "tools/level3_shared/__init__.py",
        ROOT / "requirements-level3-sgg-pretrained-v2.txt",
    ]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *map(str, files)], text=True, cwd=ROOT):
        raise ValueError("commit_scoring_code_before_running")
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
    }


def validate_assistant_review(pack, results, score_receipt):
    if (
        pack["author"] != "assistant"
        or pack["human_ground_truth"] is not False
        or pack["score_run_receipt_sha256"] != score_receipt
    ):
        raise ValueError("assistant_review_provenance_mismatch")
    expected = {r["task_id"] for r in results}
    ids = [r["task_id"] for r in pack["reviews"]]
    if len(set(ids)) != len(ids) or not set(ids) <= expected:
        raise ValueError("unknown_duplicate_review_case")
    for r in pack["reviews"]:
        if any(
            r[k] not in ("appears_satisfied", "appears_not_satisfied", "uncertain")
            for k in ("edit_assessment", "preservation_assessment")
        ):
            raise ValueError("invalid_assistant_assessment")
        if (
            r["evidence_basis"] != "source_and_target_images_plus_frozen_graphs"
            or not r["observation_vi"].strip()
            or not r["graph_comment_vi"].strip()
        ):
            raise ValueError("assistant_review_needs_visual_and_graph_evidence")
    return pack
