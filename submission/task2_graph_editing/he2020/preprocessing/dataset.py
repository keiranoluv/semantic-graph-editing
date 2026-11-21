from common.io import encode, digest, sha256, fixed, write, read
"""Pinned official archive, immutable cohorts and sandboxed structural reading."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from task2_graph_editing.he2020.preprocessing import VERSION

from common.paths import ROOT
ARCHIVE_SHA256 = "6e13b096f2756b638f59e38b13b80658f91f53cbfa6750a210929fa8a091d337"
OFFICIAL_COMMIT = "6e1b396007a0c938bccda2bbddade332745b3f42"
REPOSITORY = "https://github.com/xlhex/SceneGraphModification"
PAPER = "https://aclanthology.org/2020.findings-emnlp.87/"
DRIVE = "https://drive.google.com/file/d/1K2lo1Dt7GJskyUVR9x5LH-mZya28KcDY/view?usp=sharing"
DOWNLOAD = (
    "https://drive.usercontent.google.com/download?id=1K2lo1Dt7GJskyUVR9x5LH-mZya28KcDY&export=download&confirm=t"
)
ARCHIVE = ROOT / "data/raw/emnlp2020-graph-modification-v1/scene_graph_modification_data.tgz"
OUTPUT = ROOT / "data/processed/level2/emnlp2020-crowdsourced-pilot-v1-run1"
POLICY = "A_preserve_topology_reciprocal_undirected_v1"
ALIGNMENT = "exhaustive_unique_minimum_maximum_cardinality_v1"














def code_identity():
    files = (
        list(Path(__file__).parent.glob("*.py"))
        + list(Path(__file__).parent.glob("review_assets/*"))
        + [
            ROOT / "common/editing/operations.py",
            ROOT / "common/editing/evaluation.py",
            ROOT / "task2_graph_editing/magicbrush/sequence_model/codec.py",
            ROOT / "task2_graph_editing/magicbrush/sequence_model/data.py",
            ROOT / "task2_graph_editing/magicbrush/review_interface.py",
        ]
    )
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
    }


def verified_archive(path):
    actual = sha256(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError("official_archive_checksum_mismatch")
    return actual


def download(path=ARCHIVE):
    path = Path(path)
    if path.exists():
        verified_archive(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".download")
    if partial.exists():
        raise ValueError("previous_partial_download_preserved")
    with urllib.request.urlopen(DOWNLOAD, timeout=90) as response, partial.open("xb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            if stream.tell() > 80 * 1024**2:
                raise ValueError("download_size_budget")
    verified_archive(partial)
    os.link(partial, path)
    partial.unlink()


def member_bytes(archive, split, name):
    if split not in ("train", "dev") or name not in ("src_graph.bin", "tgt_graph.bin", "src_text.txt"):
        raise ValueError("only_crowdsourced_train_dev_allowed_never_test")
    path = f"data/crowdsourced_data/{split}_{name}"
    with tarfile.open(archive, "r:gz") as tar:
        matches = [m for m in tar.getmembers() if m.name == path]
        if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 32 * 1024**2:
            raise ValueError("missing_unsafe_or_oversized_member:" + path)
        return tar.extractfile(matches[0]).read()


def sandbox_scan(content, indices):
    if os.geteuid() != 0:
        raise ValueError("inspection_requires_linux_root_for_empty_chroot_no_unsafe_fallback")
    with tempfile.TemporaryDirectory(prefix="emnlp2020-empty-jail-") as jail:
        process = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("safe_pickle.py")), jail, json.dumps(indices)],
            input=content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C", "PYTHONHASHSEED": "0"},
            cwd=jail,
            timeout=90,
            close_fds=True,
            check=False,
        )
    if process.returncode:
        raise ValueError("safe_reader_failed:" + process.stderr.decode(errors="replace")[-2000:])
    return json.loads(process.stdout)


def inspect(archive):
    checksum = verified_archive(archive)
    with tarfile.open(archive) as tar:
        members = [{"name": m.name, "bytes": m.size, "file": m.isfile()} for m in tar]
    text = member_bytes(archive, "train", "src_text.txt")
    return {
        "adapter_version": VERSION,
        "repository": REPOSITORY,
        "official_code_commit": OFFICIAL_COMMIT,
        "paper": PAPER,
        "citation": "He et al. 2020. Scene Graph Modification Based on Natural Language Commands. Findings of EMNLP, 972–990. doi:10.18653/v1/2020.findings-emnlp.87",
        "dataset_url": DRIVE,
        "download_url": DOWNLOAD,
        "archive_sha256": checksum,
        "checksum_origin": "computed_from_official_README_download; no_author_signed_checksum_available",
        "archive_bytes": Path(archive).stat().st_size,
        "archive_members": members,
        "available_variants": ["crowdsourced", "mscoco", "GCC"],
        "selected_variant": "crowdsourced",
        "train_instruction_count": len(text.decode("utf-8").splitlines()),
        "train_instructions_sha256": hashlib.sha256(text).hexdigest(),
        "license": {
            "dataset": "not_specified_in_official_README_or_archive",
            "code": "no_LICENSE_file_at_verified_commit",
            "redistribution_authorized": False,
            "action": "retain local research copy; do not commit or redistribute raw/converted records; seek author clarification before redistribution",
        },
        "official_test_content_opened": False,
        "assignment_source_requirement_satisfied": False,
        "instructor_approval": "pending",
    }


def select_indices(count, size=100, seed=42):
    if count < size:
        raise ValueError("not_enough_actual_records")
    # Hash ordering avoids Python RNG/version differences and ignores conversion.
    return sorted(range(count), key=lambda i: digest([VERSION, seed, "train", i]))[:size]


def prepare(archive, output):
    started = time.monotonic()
    output = Path(output)
    inspection = inspect(archive)
    write(output / "inspection.json", inspection)
    count = inspection["train_instruction_count"]
    selection = {
        "adapter_version": VERSION,
        "archive_sha256": ARCHIVE_SHA256,
        "source_variant": "crowdsourced",
        "official_split": "train",
        "seed": 42,
        "population": count,
        "size": 100,
        "selection": "first_100_by_sha256_version_seed_split_original_index_before_quality_filters",
        "indices": select_indices(count),
    }
    write(output / "cohort.json", selection)
    if (output / "raw-records.json").exists() and (output / "raw-receipt.json").exists():
        receipt = read(output / "raw-receipt.json")
        if receipt["cohort_sha256"] != sha256(output / "cohort.json") or receipt["raw_sha256"] != sha256(
            output / "raw-records.json"
        ):
            raise ValueError("prepared_receipt_mismatch")
        return receipt
    contents = {
        name: member_bytes(archive, "train", name) for name in ("src_graph.bin", "tgt_graph.bin", "src_text.txt")
    }
    graphs = {name: sandbox_scan(contents[name], selection["indices"]) for name in ("src_graph.bin", "tgt_graph.bin")}
    if any(g["record_count"] != count for g in graphs.values()):
        raise ValueError("graph_instruction_record_count_mismatch")
    instructions = contents["src_text.txt"].decode("utf-8").splitlines()
    records = []
    for index in selection["indices"]:
        source, target = (graphs[name]["selected"][str(index)] for name in ("src_graph.bin", "tgt_graph.bin"))
        records.append(
            {
                "id": f"emnlp2020_crowd_train_{index:06d}",
                "source_dataset": "SceneGraphModification",
                "source_variant": "crowdsourced",
                "official_split": "train",
                "split": "train",
                "original_record_index": index,
                "instruction": instructions[index],
                "raw_source": source["graph"],
                "raw_target": target["graph"],
                "load_errors": [item["error"] for item in (source, target) if item["error"]],
            }
        )
    write(output / "raw-records.json", records)
    receipt = {
        "archive_sha256": ARCHIVE_SHA256,
        "cohort_sha256": sha256(output / "cohort.json"),
        "raw_sha256": sha256(output / "raw-records.json"),
        "members": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in contents.items()
        },
        "scans": {k: {field: v for field, v in scan.items() if field != "selected"} for k, scan in graphs.items()},
        "records_materialized": len(records),
        "test_content_opened": False,
        "code": code_identity(),
        "runtime_seconds": time.monotonic() - started,
    }
    write(output / "raw-receipt.json", receipt)
    return receipt


def prepare_expansion(archive, output, train_size, dev_size, pilot_evidence):
    """Separate bounded cohort, invoked only after the human pilot gate passes."""
    verified_archive(archive)
    output = Path(output)
    write(output / "inspection.json", inspect(archive))
    contents, selections, populations = {}, {}, {}
    for split, size in (("train", train_size), ("dev", dev_size)):
        for name in ("src_graph.bin", "tgt_graph.bin", "src_text.txt"):
            contents[f"{split}_{name}"] = member_bytes(archive, split, name)
        count = len(contents[f"{split}_src_text.txt"].decode().splitlines())
        populations[split] = count
        if size > count or size <= 0:
            raise ValueError("expansion_size_outside_official_population")
        selections[split] = sorted(range(count), key=lambda i: digest([VERSION, 42, split, i]))[:size]
    cohort = {
        "adapter_version": VERSION,
        "archive_sha256": ARCHIVE_SHA256,
        "source_variant": "crowdsourced",
        "seed": 42,
        "size": train_size + dev_size,
        "indices_by_split": selections,
        "populations": populations,
        "expansion_pilot_evidence": pilot_evidence,
        "selection": "bounded_hash_order_before_filters_official_train_dev_separate_never_test",
    }
    write(output / "cohort.json", cohort)
    if (output / "raw-receipt.json").exists():
        receipt = read(output / "raw-receipt.json")
        if receipt["cohort_sha256"] != sha256(output / "cohort.json") or receipt["raw_sha256"] != sha256(
            output / "raw-records.json"
        ):
            raise ValueError("expansion_resume_evidence_changed")
        return receipt
    records, scans = [], {}
    for split in ("train", "dev"):
        parsed = {}
        for name in ("src_graph.bin", "tgt_graph.bin"):
            result = sandbox_scan(contents[f"{split}_{name}"], selections[split])
            if result["record_count"] != populations[split]:
                raise ValueError("expansion_graph_text_count_mismatch")
            parsed[name] = result["selected"]
            scans[f"{split}_{name}"] = {k: v for k, v in result.items() if k != "selected"}
        instructions = contents[f"{split}_src_text.txt"].decode().splitlines()
        for index in selections[split]:
            source, target = (parsed[name][str(index)] for name in ("src_graph.bin", "tgt_graph.bin"))
            records.append(
                {
                    "id": f"emnlp2020_crowd_{split}_{index:06d}",
                    "source_dataset": "SceneGraphModification",
                    "source_variant": "crowdsourced",
                    "official_split": split,
                    "split": "train" if split == "train" else "val",
                    "original_record_index": index,
                    "instruction": instructions[index],
                    "raw_source": source["graph"],
                    "raw_target": target["graph"],
                    "load_errors": [r["error"] for r in (source, target) if r["error"]],
                }
            )
    write(output / "raw-records.json", records)
    receipt = {
        "archive_sha256": ARCHIVE_SHA256,
        "cohort_sha256": sha256(output / "cohort.json"),
        "raw_sha256": sha256(output / "raw-records.json"),
        "members": {k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in contents.items()},
        "scans": scans,
        "records_materialized": len(records),
        "test_content_opened": False,
        "code": code_identity(),
    }
    write(output / "raw-receipt.json", receipt)
    return receipt
