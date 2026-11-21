import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path("runs/level3/evaluation-v1/seed42")
REPORT = EVAL / "reports/0fc28934ad71812a"
DEFAULT = Path("runs/level3/submission-v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or not p.parts or name != str(p) or ":" in p.parts[0]:
        raise ValueError("unsafe_archive_path:" + name)
    return str(p)


def inventory():
    from tools.level3_evaluation.core import inputs

    cfg, records = inputs()
    files = {}

    def add(path, expected=None):
        path = Path(path)
        path = path if path.is_absolute() else ROOT / path
        relative = str(path.relative_to(ROOT))
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing_or_symlink:" + relative)
        if relative not in files:
            files[relative] = {"size": path.stat().st_size, "sha256": sha(path)}
        if expected and files[relative]["sha256"] != expected:
            raise ValueError("artifact_hash_changed:" + relative)
        return relative

    qwen = read(ROOT / cfg["qwen_run"] / "index.json")
    sgg = read(ROOT / cfg["sgg_run"] / "index.json")
    labels = read(ROOT / REPORT / "labels.json")
    results = {r["task_id"]: r for r in read(ROOT / REPORT / "results.json")}
    sources, cases, predictions = {}, [], {}
    run_roots = {ROOT / cfg["qwen_run"], ROOT / cfg["sgg_run"]}
    for record, entry in zip(records, qwen["entries"], strict=True):
        task, image = record["task"], record["image"]
        tid, iid = task["task_id"], task["image_id"]
        sources[iid] = {
            "image_id": iid,
            "dataset": task["dataset"],
            "original": add(task["source_path"], task["sha256"]),
            "source": add(image["source_path"], image["source_sha256"]),
            "upstream_url": task["source_asset_url"],
            "revision": task["source_revision_sha"],
            "dataset_license": task["dataset_license_claim"],
        }
        graph_paths = {role: add(p["graph_path"], p["graph_sha256"]) for role, p in record["pair"]["roles"].items()}
        case_dir = Path(entry["case_dir"])
        for path in case_dir.glob("*.json"):
            add(path)
        run_roots.add(case_dir.parent.parent)
        cases.append(
            {
                "task_id": tid,
                "image_id": iid,
                "dataset": task["dataset"],
                "instruction": task["instruction_en"],
                "task": task,
                "source": add(image["source_path"], image["source_sha256"]),
                "target": add(image["target_path"], image["target"]["sha256"]),
                "graphs": graph_paths,
                "author": record["author"],
                "scores_and_labels": results[tid],
            }
        )
    for key, item in sgg["images"].items():
        prediction = item["prediction"]
        exported = {}
        for field, value in prediction.items():
            if field.endswith("_path"):
                exported[field] = add(value, prediction.get(field.replace("_path", "_sha256")))
        predictions[key] = {"files": exported, "uses": item["uses"], "prediction_key": item["key"]}
        folder = Path(prediction["record_path"]).parent
        for path in folder.glob("*.json"):
            add(path)
        receipt = Path(prediction["receipt_path"])
        for path in receipt.parent.glob("*.json"):
            # Profiling traces and transient viewer caches are not inference evidence.
            if not path.name.startswith("react-profile") and path.name not in (
                "preview-cache.json",
                "review-data.json",
            ):
                add(path)
    for root in run_roots:
        for path in root.glob("*.json"):
            if path.name not in ("preview-cache.json", "review-data.json"):
                add(path)
    for dataset in ("sidewalk", "travel"):
        add(ROOT / "data/raw/level3/qwen21-batch100-v1" / dataset / "source-receipt.json")
    add(ROOT / EVAL / "run.json")
    for sub in ("graph", "metrics"):
        for path in (ROOT / EVAL / sub).rglob("*.json"):
            if "wandb" not in path.relative_to(ROOT / EVAL / sub).parts:
                add(path)
    for sub in ("label-recheck-v1", "labels-final-v1"):
        for path in (ROOT / EVAL / sub).glob("*.json"):
            add(path)
    for path in (ROOT / REPORT).glob("*"):
        if path.is_file():
            add(path)
    for sub in ("media", "analysis-final-v1"):
        for path in (ROOT / REPORT / sub).rglob("*"):
            if path.is_file():
                add(path)
    add("qwen200-reviewed.json", labels["normalization"]["raw_sha256"])
    # Small, genuine Task 2 predictions, displayed separately from Task 3.
    task2_dir = Path("runs/level2/emnlp2020-task-b-wandb-v4-test-v1/gnn-seed42")
    receipt = read(ROOT / task2_dir / "completed-receipt.json")
    for name in ("cases.json", "result.json"):
        add(task2_dir / name, receipt["files"][name])
    add(task2_dir / "completed-receipt.json")
    manifest = {
        "schema": "level3-portable-evidence-v1",
        "original_workspace": str(ROOT),
        "counts": {"sources": len(sources), "edits": len(cases), "sgg_predictions": len(predictions)},
        "operations": dict(Counter(r["task"]["operation"]["kind"] for r in cases)),
        "sources": sources,
        "cases": cases,
        "predictions": predictions,
        "labels": str(REPORT / "labels.json"),
        "graph_scores": str(EVAL / "graph/results.json"),
        "summary": str(REPORT / "summary.json"),
        "analysis_html": str(REPORT / "analysis-final-v1/index.html"),
        "task2_cases": str(task2_dir / "cases.json"),
        "task2_result": str(task2_dir / "result.json"),
        "task2_scope": "stored GNN seed42 predictions on EMNLP; not predictions for Qwen/REACT++ images",
        "graph_config": cfg["graph_config"],
        "files": files,
        "excluded": ["model weights", "virtual environments", "W&B logs", "unrelated untracked files"],
    }
    if manifest["counts"] != {"sources": 100, "edits": 200, "sgg_predictions": 300}:
        raise ValueError("incomplete_cohort")
    return manifest


def prepare(output):
    manifest = inventory()
    path = Path(output) / "portable-manifest.json"
    payload = encoded(manifest)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("snapshot_changed_use_new_output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    return manifest


def build(output):
    output = Path(output)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scoped = [
        "tools/level3_submission_v1",
        "scripts/package_level3.sh",
        "notebooks/level3_task3_demo.ipynb",
        "docs/LEVEL3_DATA_CARD.md",
        "docs/LEVEL3_TASK3_REPORT.md",
        "docs/LEVEL3_SUBMISSION.md",
        "requirements-level3-demo-v1.txt",
        "tests/level3/test_submission_v1.py",
    ]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *scoped], cwd=ROOT):
        raise ValueError("commit_submission_code_and_docs_before_packaging")
    manifest = prepare(output)
    archive = output / f"level3-seed42-{commit[:7]}.zip"
    if archive.exists():
        return verify(archive)
    partial = archive.with_suffix(".zip.partial")
    included = {}
    with tempfile.TemporaryDirectory(prefix="level3-code-") as tmp:
        code_tar = Path(tmp) / "code.tar"
        subprocess.run(["git", "archive", "-o", str(code_tar), commit], cwd=ROOT, check=True)
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            with tarfile.open(code_tar) as tar:
                for member in tar:
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError("non_regular_tracked_file:" + member.name)
                    name = safe_name(member.name)
                    payload = tar.extractfile(member).read()
                    z.writestr(name, payload)
                    included[name] = {
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "origin": "git_HEAD",
                    }
            for i, (name, info) in enumerate(sorted(manifest["files"].items())):
                if name in included:
                    if included[name]["sha256"] != info["sha256"]:
                        raise ValueError("tracked_artifact_differs_from_HEAD:" + name)
                    continue
                name = safe_name(name)
                digest, size = hashlib.sha256(), 0
                with (ROOT / name).open("rb") as src, z.open(name, "w", force_zip64=True) as dest:
                    while chunk := src.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        dest.write(chunk)
                if digest.hexdigest() != info["sha256"] or size != info["size"]:
                    raise ValueError("file_changed_while_packaging:" + name)
                included[name] = {**info, "origin": "frozen_artifact"}
                if i % 400 == 0:
                    print(json.dumps({"packaged_artifacts": i, "total": len(manifest["files"])}), flush=True)
            extras = {
                "portable-manifest.json": encoded(manifest),
                "BUNDLE_README.md": (ROOT / "docs/LEVEL3_SUBMISSION.md").read_bytes(),
            }
            preview = output / "demo-executed.ipynb"
            if preview.exists():
                original = read(ROOT / "notebooks/level3_task3_demo.ipynb")
                executed = read(preview)
                if [c["source"] for c in original["cells"]] != [c["source"] for c in executed["cells"]]:
                    raise ValueError("notebook_preview_source_is_stale")
                extras["DEMO_PREVIEW.ipynb"] = preview.read_bytes()
                extras["DEMO_PREVIEW.html"] = preview.with_suffix(".html").read_bytes()
            for name, payload in extras.items():
                z.writestr(name, payload)
                included[name] = {
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "origin": "derived",
                }
            index = {
                "schema": "level3-backup-index-v1",
                "commit": commit,
                "counts": manifest["counts"],
                "files": included,
            }
            z.writestr("BACKUP_INDEX.json", encoded(index))
    # Atomic publication, without replacing an existing archive.
    os.link(partial, archive)
    partial.unlink()
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(sha(archive) + "  " + archive.name + "\n")
    return verify(archive)


def verify(archive):
    archive = Path(archive)
    sidecar = archive.with_suffix(".zip.sha256")
    if not sidecar.exists() or sha(archive) != sidecar.read_text().split()[0]:
        raise ValueError("archive_checksum_mismatch")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate_archive_paths")
        for name in names:
            safe_name(name)
        index = json.loads(z.read("BACKUP_INDEX.json"))
        if set(names) != set(index["files"]) | {"BACKUP_INDEX.json"}:
            raise ValueError("archive_membership_mismatch")
        for name, info in index["files"].items():
            with z.open(name) as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            if actual != info["sha256"] or z.getinfo(name).file_size != info["size"]:
                raise ValueError("archive_member_changed:" + name)
    result = {
        "archive": str(archive.resolve()),
        "sha256": sha(archive),
        "bytes": archive.stat().st_size,
        "files_verified": len(index["files"]),
        "counts": index["counts"],
        "code_commit": index["commit"],
    }
    archive.with_suffix(".verification.json").write_bytes(encoded(result))
    return result
