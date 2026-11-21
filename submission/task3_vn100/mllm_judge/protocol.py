"""Frozen inputs and independent scoring contract for the VN100 judge."""

import json
import re
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from task3_vn100.image_graph.core import ROOT, digest, read, sha256
from task3_vn100.qwen_unified.core import atomic

CONFIG = ROOT / "configs/level3/mllm-judge-v1.json"
RELEASE = CONFIG.with_name("mllm-judge-release-v1.json")
REQUIREMENTS = ROOT / "environments/vlm-judge.txt"
RUN = ROOT / "runs/level3/mllm-judge-v1/seed42"


def checked(ref):
    path = ROOT / ref["path"]
    if sha256(path) != ref["sha256"]:
        raise ValueError(f"Frozen input changed: {path}")
    return path


def by_id(rows):
    table = {r["task_id"]: r for r in rows}
    if len(table) != len(rows):
        raise ValueError("Duplicate task_id")
    return table


def project_vn100(cfg):
    """Reproduce the inference-only manifest from the historical run on CPU."""
    tasks = read(checked(cfg["cohort"]))["tasks"]
    entries = by_id(read(checked(cfg["images"]))["entries"])
    if entries.keys() != by_id(tasks).keys() or Counter(Counter(t["image_id"] for t in tasks).values()) != {2: 100}:
        raise ValueError("Expected 200 matching IDs from 100 two-edit source images")
    main = []
    for task in tasks:
        tid = task["task_id"]
        entry = entries[tid]
        im = entry["result"]
        if (entry["status"], im["task_id"], im["image_id"], im["prompt"]) != (
            "ok",
            tid,
            task["image_id"],
            task["instruction_en"],
        ):
            raise ValueError(f"Image / instruction mismatch: {tid}")
        main.append(
            dict(
                task_id=tid,
                image_id=task["image_id"],
                instruction=im["prompt"],
                source=str(Path(im["source_path"]).relative_to(ROOT)),
                target=str(Path(im["target_path"]).relative_to(ROOT)),
                source_sha256=im["source_sha256"],
                target_sha256=im["target"]["sha256"],
                binding=entry["binding"],
            )
        )
    return main


def datasets(cfg):
    main = read(checked(cfg["vn100"]))["cases"]
    main_ids = by_id(main)
    if len(main_ids) != 200 or Counter(Counter(c["image_id"] for c in main).values()) != {2: 100}:
        raise ValueError("Expected 200 unique tasks, two edits per source")
    smoke = read(checked(cfg["smoke"]))["cases"]
    if len(by_id(smoke)) != 5 or by_id(smoke).keys() & main_ids.keys():
        raise ValueError("Invalid smoke panel")
    seen = {}
    for case in main + smoke:
        for role in ("source", "target"):
            path = ROOT / case[role]
            if path not in seen:
                seen[path] = sha256(path)
            if seen[path] != case[role + "_sha256"]:
                raise ValueError(f"Image changed: {path}")

    def hashes(rows):
        return {r[k] for r in rows for k in ("source_sha256", "target_sha256")}

    if hashes(main) & hashes(smoke):
        raise ValueError("Smoke and VN100 images overlap")

    def square(path):
        with Image.open(ROOT / path) as im:
            return im.size == (1024, 1024)

    if not any(square(r["source"]) and square(r["target"]) for r in smoke):
        raise ValueError("Missing full-size two-image smoke case")
    return main, smoke


def versions():
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        if "==" in line:
            name, expected = line.split("==")
            pins[name] = version(name)
            if pins[name] != expected:
                raise ValueError(f"{name}=={expected} required, found {pins[name]}")
    return pins


def committed(paths):
    paths = [str(p.relative_to(ROOT)) for p in paths]
    state = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=ROOT)
    if state:
        raise ValueError("Commit frozen inputs and code before GPU execution: " + state.decode())
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", *paths], cwd=ROOT, text=True).strip()


def identity(cfg):
    files = [
        CONFIG,
        REQUIREMENTS,
        checked(cfg["prompt"]),
        checked(cfg["smoke"]),
        checked(cfg["vn100"]),
        ROOT / "task3_vn100/commands/run_level3_mllm_judge_v1.sh",
        ROOT / "task3_vn100/image_graph/core.py",
        ROOT / "task3_vn100/qwen_unified/core.py",
        ROOT / "task3_vn100/analysis/core.py",
    ]
    files.extend(sorted(Path(__file__).parent.glob("*.py")))
    return {
        "commit": committed(files),
        "files": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "packages": versions(),
    }


def messages(template, instruction):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": template.replace("{instruction}", instruction)},
            ],
        }
    ]


def unique_keys(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate_key")
    return obj


def parse(raw):
    try:
        obj = json.loads(raw, object_pairs_hook=unique_keys)
        if type(obj) is not dict or obj.keys() != {"fulfillment", "preservation", "evidence"}:
            raise ValueError("schema")
        for key in ("fulfillment", "preservation"):
            if type(obj[key]) is not int or not 0 <= obj[key] <= 4:
                raise ValueError("score_range_or_type")
        evidence = obj["evidence"]
        if (
            not isinstance(evidence, str)
            or not re.search(r"[A-Za-z]", evidence)
            or "\n" in evidence.strip()
            or re.search(r'[.!?]["\']?\s+[A-Z]', evidence.strip())
        ):
            raise ValueError("expected_one_evidence_sentence")
    except ValueError as exc:
        return {
            "status": "invalid_format",
            "error": str(exc),
            "fulfillment": None,
            "preservation": None,
            "evidence": None,
            "J": None,
        }
    return {"status": "ok", "error": None, **obj, "J": obj["fulfillment"] * obj["preservation"] / 16}


def snapshot(cfg, download=False, weights=True):
    from huggingface_hub import snapshot_download

    assets = {name: h for name, h in cfg["assets_sha256"].items() if weights or not name.endswith(".safetensors")}
    folder = Path(
        snapshot_download(
            cfg["model"],
            revision=cfg["revision"],
            allow_patterns=list(assets),
            local_files_only=not download,
            max_workers=2,
        )
    )
    for name, expected in assets.items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Checkpoint file changed: {name}")
    return folder


def save_bound(path, value):
    atomic(path, {"data": value, "sha256": digest(value)})


def load_bound(path):
    record = read(path)
    if digest(record["data"]) != record["sha256"]:
        raise ValueError(f"Changed output: {path}")
    return record["data"]
