"""Full official split conversion using the frozen v1 safe reader and compiler."""

import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from itertools import permutations
from pathlib import Path

from common.editing.codec import decode_actions, encode_actions
from common.editing.data import Example
from common.editing.operations import KINDS, EditGraph, Operation, apply_operations
from common.io import digest, read, sha256, write
from task2_graph_editing.he2020.preprocessing import ARCHIVE_SHA256, OFFICIAL_COMMIT, verified_archive
from task2_graph_editing.he2020.preprocessing.alignment import validate_raw
from task2_graph_editing.he2020.preprocessing.compiler import compile_record

VERSION = "emnlp2020-task-b-v2"

MEMBERS = ("src_graph.bin", "tgt_graph.bin", "src_text.txt")


def scan_all(content, count):
    if os.geteuid() != 0:
        raise ValueError("safe_reader_requires_root_empty_chroot")
    with tempfile.TemporaryDirectory(prefix="emnlp-v2-jail-") as jail:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).parent / "preprocessing/safe_reader.py"), jail, str(count)],
            input=content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C", "PYTHONHASHSEED": "0"},
            cwd=jail,
            timeout=90,
            close_fds=True,
            check=False,
        )
    if result.returncode:
        raise ValueError("safe_reader_failed:" + result.stderr.decode(errors="replace")[-2000:])
    import json

    return json.loads(result.stdout)


def member(archive, split, name):
    if split not in ("train", "dev", "test") or name not in MEMBERS:
        raise ValueError("unknown_official_member")
    path = f"data/crowdsourced_data/{split}_{name}"
    with tarfile.open(archive, "r:gz") as stream:
        matches = [m for m in stream.getmembers() if m.name == path]
        if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 32 * 1024**2:
            raise ValueError("unsafe_member")
        return stream.extractfile(matches[0]).read()


def filter_rows(rows, reserved=None):
    """Input-only grouping. Earlier official splits win, even for rejected labels."""
    groups = dict(reserved or {})
    pairs = set()
    for row in rows:
        try:
            group = source_fingerprint(row["raw_source"])
        except (ValueError, TypeError):
            row["exclusion"] = "unfingerprintable_source"
            continue
        row["source_group_id"] = group
        pair = digest([group, " ".join(row["instruction"].split()).casefold()])
        split = row["split"]
        prior = groups.setdefault(group, split)
        if prior != split:
            row["exclusion"] = "source_overlap_earlier_official_split"
        elif pair in pairs:
            row["exclusion"] = "duplicate_source_instruction_within_split"
        elif row["status"] != "candidate":
            row["exclusion"] = "conversion_rejected"
        else:
            row["exclusion"] = None
        pairs.add(pair)
    return groups


def convert_split(archive, official):
    contents = {name: member(archive, official, name) for name in MEMBERS}
    instructions = contents["src_text.txt"].decode("utf-8").splitlines()
    parsed = {name: scan_all(contents[name], len(instructions)) for name in MEMBERS[:2]}
    if any(p["record_count"] != len(instructions) for p in parsed.values()):
        raise ValueError("graph_instruction_alignment_count_mismatch")
    rows = []
    for i, instruction in enumerate(instructions):
        source, target = (parsed[name]["selected"][str(i)] for name in MEMBERS[:2])
        rows.append(
            compile_record(
                {
                    "id": f"emnlp2020_crowd_{official}_{i:06d}",
                    "source_dataset": "SceneGraphModification",
                    "source_variant": "crowdsourced",
                    "official_split": official,
                    "split": {"dev": "val"}.get(official, official),
                    "original_record_index": i,
                    "instruction": instruction,
                    "raw_source": source["graph"],
                    "raw_target": target["graph"],
                    "load_errors": [r["error"] for r in (source, target) if r["error"]],
                }
            )
        )
    evidence = {name: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for name, v in contents.items()}
    return rows, evidence


def summarize(rows):
    result = {}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        accepted = [r for r in subset if not r["exclusion"]]
        result[split] = {
            "official_records": len(subset),
            "usable": len(accepted),
            "conversion_candidates_before_dedup": sum(r["status"] == "candidate" for r in subset),
            "exclusions": dict(Counter(r["exclusion"] for r in subset if r["exclusion"])),
            "conversion_problems": dict(Counter(p for r in subset for p in r["problems"])),
            "operation_support": {k: sum(r["operation_support"].get(k, 0) for r in accepted) for k in KINDS},
            "reindexed_correspondence_pairs": sum(
                any(
                    m["source_raw_id"] != m["target_raw_id"]
                    for m in r.get("alignment", {}).get("node_correspondence", [])
                )
                for r in accepted
            ),
            "raw_source_nodes": dict(
                sorted(Counter(len(r["raw_source"]["nodes"]) for r in subset if r["raw_source"]).items())
            ),
            "human_reviewed": 0,
            "semantic_accuracy": None,
        }
    return result


def prepare(archive, output, *, test=False):
    output = Path(output)
    verified_archive(archive)
    name = "test" if test else "trainval"
    path = output / f"{name}.json"
    if path.exists():
        load_bundle(output, name)
        return read(output / f"{name}-summary.json")
    reserved = read(output / "trainval-groups.json") if test else None
    rows, members = [], {}
    for split in ("test",) if test else ("train", "dev"):
        converted, evidence = convert_split(archive, split)
        rows.extend(converted)
        members[split] = evidence
    groups = filter_rows(rows, reserved)
    payload = {
        "schema": VERSION,
        "archive_sha256": ARCHIVE_SHA256,
        "official_commit": OFFICIAL_COMMIT,
        "members": members,
        "records": rows,
        "policy": "v1_A_reciprocal_undirected_unique_structural_correspondence",
        "human_review_status": "unreviewed_research_supervision_not_human_validated",
        "source_requirement_approval": "pending_instructor",
    }
    write(path, payload)
    if not test:
        write(output / "trainval-groups.json", groups)
    summary = summarize(rows)
    write(output / f"{name}-summary.json", summary)
    write(
        output / f"{name}-receipt.json",
        {
            "files": {
                p.name: sha256(p)
                for p in [path, output / f"{name}-summary.json"] + ([] if test else [output / "trainval-groups.json"])
            }
        },
    )
    return summary


def load_bundle(output, name):
    output = Path(output)
    for file, checksum in read(output / f"{name}-receipt.json")["files"].items():
        if sha256(output / file) != checksum:
            raise ValueError("dataset_artifact_changed")
    bundle = read(output / f"{name}.json")
    if bundle["schema"] != VERSION or bundle["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("dataset_identity_changed")
    examples = {s: [] for s in (("test",) if name == "test" else ("train", "val"))}
    for row in bundle["records"]:
        if row["exclusion"]:
            continue
        source, target = (EditGraph.from_dict(row[k]) for k in ("source_graph", "target_graph"))
        ops = tuple(Operation.from_dict(o) for o in row["operations"])
        if apply_operations(source, ops) != target or decode_actions(encode_actions(ops)[1:], source) != ops:
            raise ValueError("ground_truth_reconstruction_failed")
        examples[row["split"]].append(
            Example(row["id"], row["split"], row["source_group_id"], row["instruction"], source, target, ops)
        )
    return examples


def source_fingerprint(raw):
    """ID/order independent graph group; no target information, max seven nodes."""
    nodes, edges = validate_raw(raw)
    if len(nodes) > 7:
        raise ValueError("fingerprint_search_budget")
    # Only permute within equal labels; all permutations across labels would
    # produce a larger label tuple and cannot be the lexicographic minimum.
    groups = {}
    for k, v in nodes.items():
        groups.setdefault(v, []).append(k)
    orders = [()]
    for label in sorted(groups):
        orders = [prefix + order for prefix in orders for order in permutations(groups[label])]
    encodings = []
    for order in orders:
        ids = {k: i for i, k in enumerate(order)}
        triples = []
        for (u, v), p in edges.items():
            a, b = ids[u], ids[v]
            if not raw["directed"]:
                a, b = sorted((a, b))
            triples.append((a, p, b))
        encodings.append((tuple(nodes[k] for k in order), tuple(sorted(triples))))
    return digest([raw["directed"], min(encodings)])
