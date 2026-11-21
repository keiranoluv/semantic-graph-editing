"""Reclaim exact duplicate PNG allocations in completed Qwen/REACT runs, keeping every path and byte."""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from tools.level3_image_graph_v1.core import ROOT, sha256, write
from tools.level3_qwen21_v1.core import bundle

from .core import atomic

ROOTS = [ROOT / "runs/level3/qwen100-v1/seed42-v1", ROOT / "runs/level3/qwen-react100-v1/seed42-v1"]


def active_writers(roots):
    matches = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if "--output" in args and args.index("--output") + 1 < len(args):
            output = Path(args[args.index("--output") + 1])
            if not output.is_absolute():
                output = ROOT / output
            output = output.resolve()
            if any(
                output == r.resolve() or r.resolve() in output.parents or output in r.resolve().parents for r in roots
            ):
                matches.append({"pid": int(p.name), "output": str(output)})
    return matches


def plan(roots):
    if active_writers(roots):
        raise ValueError("do_not_clean_an_active_output")
    by_size = defaultdict(list)
    receipts = {}
    for root in roots:
        bundle(root, "results.json")
        receipts[str(root)] = sha256(root / "receipt.json")
        for p in root.rglob("*.png"):
            if not p.is_symlink() and root.resolve() in p.resolve().parents:
                by_size[p.stat().st_size].append(p)
    groups = defaultdict(list)
    for size, paths in by_size.items():
        if len(paths) > 1:
            for p in paths:
                groups[(size, sha256(p))].append(p)
    operations = []
    for (_size, checksum), paths in groups.items():
        # Prefer the original Qwen files as keepers, irrespective of inode count.
        paths.sort(key=lambda p: (ROOTS[0].resolve() not in p.resolve().parents, str(p)))
        keeper = paths[0]
        for p in paths[1:]:
            a, b = keeper.stat(), p.stat()
            if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
                continue
            if a.st_dev != b.st_dev or (a.st_mode, a.st_uid, a.st_gid) != (b.st_mode, b.st_uid, b.st_gid):
                continue
            operations.append(
                {
                    "keeper": str(keeper),
                    "duplicate": str(p),
                    "sha256": checksum,
                    "duplicate_inode": b.st_ino,
                    "duplicate_mtime_ns": b.st_mtime_ns,
                    "bytes": b.st_size,
                    "allocated_bytes": b.st_blocks * 512,
                    "duplicate_links": b.st_nlink,
                }
            )
    return {
        "schema": "qwen-exact-storage-dedup-v1",
        "roots": list(map(str, roots)),
        "receipts_before": receipts,
        "operations": operations,
        "scope": "PNG storage only; keep every filename and exact file contents; never remove manifests, labels or model files",
    }


def apply(pack, journal):
    roots = [Path(p).resolve() for p in pack["roots"]]
    if set(roots) != {p.resolve() for p in ROOTS}:
        raise ValueError("cleanup_roots_not_allowlisted")
    if active_writers(roots):
        raise ValueError("do_not_clean_an_active_output")
    completed, freed = [], 0
    for item in pack["operations"]:
        keep, duplicate = Path(item["keeper"]), Path(item["duplicate"])
        if any(p.is_symlink() or not any(r in p.resolve().parents for r in roots) for p in (keep, duplicate)):
            raise ValueError("cleanup_path_outside_allowlist")
        before = duplicate.stat()
        if (
            before.st_ino != item["duplicate_inode"]
            or before.st_mtime_ns != item["duplicate_mtime_ns"]
            or sha256(keep) != item["sha256"]
            or sha256(duplicate) != item["sha256"]
        ):
            raise ValueError("cleanup_plan_stale")
        temp = duplicate.with_name(".dedup-" + uuid4().hex)
        try:
            os.link(keep, temp)
            if sha256(temp) != item["sha256"]:
                raise ValueError("replacement_bytes_changed")
            os.replace(temp, duplicate)
        finally:
            if temp.exists():
                temp.unlink()
        assert sha256(duplicate) == item["sha256"]
        reclaimed = before.st_blocks * 512 if before.st_nlink == 1 else 0
        freed += reclaimed
        completed.append({**item, "reclaimed_allocated_bytes": reclaimed})
        atomic(journal, {"completed": completed, "reclaimed_allocated_bytes": freed})
    for root in roots:
        assert sha256(root / "receipt.json") == pack["receipts_before"][str(root)]
        bundle(root, "results.json")
    return {
        "status": "ok",
        "physical_duplicate_files_reclaimed": len(completed),
        "reclaimed_allocated_bytes": freed,
        "original_receipts_unchanged_and_verified": True,
        "paths_deleted": 0,
        "native_image_bytes_changed": 0,
        "GPU_inference": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    report = args.report.resolve()
    if (ROOT / "runs/level3/qwen200").resolve() not in report.parents:
        raise ValueError("report_outside_unified_runs")
    report.mkdir(parents=True, exist_ok=False)
    pack = plan(ROOTS)
    write(report / "plan.json", pack)
    result = apply(pack, report / "journal.json") if args.apply else {"planned_duplicates": len(pack["operations"])}
    write(report / "result.json", result)
    print(result)


if __name__ == "__main__":
    main()
