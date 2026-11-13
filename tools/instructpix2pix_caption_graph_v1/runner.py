"""CPU stages with frozen plans, per-record resume, grouped splits and export gates."""

import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
import unicodedata
from pathlib import Path

from src.editing.operations import EditGraph, Operation
from src.knowledge.transe.core import sha256
from tools.edit_model_v1.data import Example
from tools.instructpix2pix_caption_graph_v1 import VERSION, dataset
from tools.instructpix2pix_caption_graph_v1.compiler import compile_record
from tools.instructpix2pix_caption_graph_v1.report import summarize
from tools.instructpix2pix_caption_graph_v1.validator import validate
from tools.magicbrush_grounded_v1.language import TAGGER, TAGGER_FILES
from tools.magicbrush_joint_v1.storage import exclusive, read, write_once
from tools.prepare_magicbrush import ROOT, digest

DEFAULT = ROOT / "data/processed/level2/instructpix2pix-caption-graph-v1/pilot"


def code_identity():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / p
        for p in (
            "src/editing/operations.py",
            "src/editing/evaluation.py",
            "src/editing/splits.py",
            "tools/edit_model_v1/codec.py",
            "tools/edit_model_v1/data.py",
            "tools/review_magicbrush.py",
            "tools/prepare_magicbrush.py",
            "tools/magicbrush_grounded_v1/language.py",
            "tools/magicbrush_joint_v1/storage.py",
        )
    ]
    paths += sorted((ROOT / "tools/magicbrush_review_assets").glob("*"))
    return {
        "version": VERSION,
        "files": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "packages": {p: importlib.metadata.version(p) for p in ("nltk", "pyarrow", "Pillow", "requests")},
        "tagger_files": TAGGER_FILES,
        "inference_inputs": ["source_graph", "frozen_instruction_embedding"],
    }


def verify_run(output):
    run = read(Path(output) / "run.json")
    if run["code_identity"] != code_identity():
        raise ValueError("code_or_environment_changed; use a new output version")
    for name in ("source", "plan", "metadata"):
        if run[f"{name}_sha256"] != sha256(Path(output) / f"{name}.json"):
            raise ValueError(f"{name}_evidence_changed")
    selection = read(Path(output) / "plan.json")
    if run["diagnostic_ids"] != selection["diagnostic_ids"] or run["seed"] != selection["seed"]:
        raise ValueError("run_selection_changed")
    return run


def family(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()) if isinstance(value, str) else ""


def split_records(records, diagnostic_ids, seed, images=None):
    """Union caption families, upstream source handles and any shared image pixels.

    Diagnostic families are reserved for development (train) before parsing.
    All other groups use stable hash thresholds, so expansion does not reshuffle.
    """
    images = images or {}
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate_example_ids")
    parent = {r["id"]: r["id"] for r in records}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    keys_by_id = {}
    for row in records:
        keys = ["caption:" + digest(family(row.get("original_prompt")))]
        handle = row.get("upstream_image_paths", {}).get("original_image")
        if handle:
            keys.append("upstream:" + str(handle))
        for role in ("source_image", "target_image"):
            if role in images.get(row["id"], {}):
                keys.append("pixels:" + images[row["id"]][role]["pixel_sha256"])
        keys_by_id[row["id"]] = keys
        for key in keys:
            if key in owners:
                a, b = sorted((root(row["id"]), root(owners[key])))
                parent[b] = a
            owners[key] = row["id"]
    diagnostic_groups = {root(i) for i in diagnostic_ids}
    groups = {}
    for identifier, keys in keys_by_id.items():
        groups.setdefault(root(identifier), set()).update(keys)
    # Prefer caption/path family keys over image hashes to keep identities stable
    # when merely adding the optional image cache.
    group_ids = {
        g: "ip2p-family:" + digest(sorted(k for k in keys if not k.startswith("pixels:"))) for g, keys in groups.items()
    }
    splits = {k: [] for k in ("train", "val", "test")}
    result = []
    for row in records:
        group = root(row["id"])
        # Hash caption key of the component; all transitive matches stay together.
        score = int(hashlib.sha256(f"{seed}:{group_ids[group]}".encode()).hexdigest()[:16], 16) / 2**64
        split = (
            "train" if group in diagnostic_groups else "test" if score < 0.15 else "val" if score < 0.30 else "train"
        )
        result.append({**row, "source_group_id": group_ids[group], "split": split})
        splits[split].append(row["id"])
    return result, {k: sorted(v) for k, v in splits.items()}


def prepare(output, *, images=False, retry_images=False):
    output = Path(output)
    dataset.inspect(output)
    if not (output / "plan.json").exists():
        dataset.plan(output)
    if (output / "metadata.json").exists():
        rows = read(output / "metadata.json")["records"]
    else:
        rows = dataset.metadata(output)
    # Read-only model loading on CPU also verifies the existing pinned resources.
    from tools.magicbrush_grounded_v1.language import tagger

    tagger()
    selection = read(output / "plan.json")
    run = {
        "code_identity": code_identity(),
        **{f"{k}_sha256": sha256(output / f"{k}.json") for k in ("source", "plan", "metadata")},
        "seed": selection["seed"],
        "diagnostic_ids": selection["diagnostic_ids"],
        "scope": "caption_candidates_cpu_only; no_training; no_image_generation",
        "review_policy": "SRS 20 diagnostic IDs before filters plus independent SRS of up to 20 train candidates for conditional precision; failed diagnostic IDs are never replaced",
        "minimum_correct_fraction": 0.8,
    }
    write_once(output / "run.json", run)
    if images:
        # Image attempts are separate from caption construction and its identity.
        dataset.prepare_images(output, rows, retry=retry_images)
    return {"records": len(rows), "run_sha256": digest(run), "images_requested": images}


def image_receipts(output):
    # A successful explicit retry adds an immutable resolution beside the failed
    # original. Compilations hash this snapshot; older review is invalidated.
    directory = Path(output) / "image-receipts"
    result = {r["id"]: r for p in sorted(directory.glob("*.json")) if len(p.stem) == 64 for r in [read(p)]}
    for path in sorted(directory.glob("*-resolved.json")):
        row = read(path)
        if row["id"] not in result or row["status"] != "available" or path.stem != digest(row["id"]) + "-resolved":
            raise ValueError("invalid_image_resolution")
        result[row["id"]] = row
    return result


def compile_dataset(output):
    output = Path(output)
    run = verify_run(output)
    started = time.monotonic()
    receipts = image_receipts(output)
    rows = read(output / "metadata.json")["records"]
    rows = [
        {
            **r,
            **{k: v for k, v in receipts.get(r["id"], {}).items() if k in ("source_image", "target_image")},
            "image_availability": receipts.get(r["id"], {}).get("status", "not_requested"),
        }
        for r in rows
    ]
    records, splits = split_records(rows, run["diagnostic_ids"], run["seed"], receipts)
    diagnostic_review = sorted(run["diagnostic_ids"], key=lambda i: digest([run["seed"], "diagnostic-review", i]))[:20]
    manifest = {
        "schema_version": 1,
        "dataset": VERSION,
        "run_sha256": digest(run),
        "records": records,
        "splits": splits,
        "review_ids": diagnostic_review,
        "diagnostic_ids": run["diagnostic_ids"],
        "split_policy": "diagnostic families train; remaining families stable SHA thresholds 70/15/15; union identical caption, source handle and source/target pixel hashes",
        "image_receipts_sha256": digest(receipts),
    }
    # Each image availability snapshot gets a new compilation directory.
    destination = output / "compilations" / digest(manifest)
    write_once(destination / "manifest.json", manifest)
    labels = []
    reused = 0
    for row in records:
        if row["split"] == "test":
            continue
        path = output / "labels" / f"{digest([digest(run), row['id']])}.json"
        if path.exists():
            label = read(path)
            reused += 1
        else:
            label = compile_record(row)
            label["input_sha256"] = digest({k: row.get(k) for k in ("id", *dataset.CAPTIONS)})
            write_once(path, label)
        if label.get("input_sha256") != digest({k: row.get(k) for k in ("id", *dataset.CAPTIONS)}):
            raise ValueError("resumed_record_input_changed")
        labels.append({**label, **{k: row[k] for k in ("split", "source_group_id")}, "instruction": row["edit_prompt"]})
    edits = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(destination / "manifest.json"),
        "run_sha256": digest(run),
        "label_origin": "caption_derived_unverified_pseudo_labels",
        "records": labels,
    }
    write_once(destination / "edits.json", edits)
    by_id = {r["id"]: r for r in labels}
    summary_path = destination / "summary.json"
    if summary_path.exists():
        summary = read(summary_path)
    else:
        runtime = {
            "wall_seconds": time.monotonic() - started,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "scope": "compile only; metadata/network excluded",
            "resumed_labels": reused,
        }
        summary = summarize([by_id[i] for i in run["diagnostic_ids"]], runtime=runtime, image_receipts=receipts)
        summary["all_development_records"] = summarize(labels, image_receipts=receipts)
        summary["held_out_test_records_not_parsed"] = len(splits["test"])
        summary["split_counts"] = {k: len(v) for k, v in splits.items()}
        write_once(summary_path, summary)
    return {"compilation": str(destination), "summary": summary}


def compilation(output):
    verify_run(output)
    paths = sorted((Path(output) / "compilations").glob("*/manifest.json"))
    if len(paths) != 1:
        raise ValueError("specify_--compilation: zero_or_multiple_immutable_compilations")
    return paths[0].parent


def check_compilation(output, path=None):
    output = Path(output)
    run = verify_run(output)
    path = Path(path) if path else compilation(output)
    manifest, edits = read(path / "manifest.json"), read(path / "edits.json")
    if (
        path.name != digest(manifest)
        or edits["dataset_manifest_sha256"] != sha256(path / "manifest.json")
        or manifest["run_sha256"] != digest(run)
        or edits["run_sha256"] != digest(run)
    ):
        raise ValueError("compilation_evidence_changed")
    if manifest["image_receipts_sha256"] != digest(image_receipts(output)):
        raise ValueError("image_snapshot_changed; compile_again_before_review_or_export")
    return path, manifest, edits


def load_examples(path):
    """Native bridge to the unchanged predictor Example/codec, source-only inputs."""
    path = Path(path)
    manifest, edits = read(path / "manifest.json"), read(path / "edits.json")
    if edits["dataset_manifest_sha256"] != sha256(path / "manifest.json"):
        raise ValueError("loader_manifest_hash_mismatch")
    index = {r["id"]: r for r in manifest["records"]}
    labels = {r["id"]: r for r in edits["records"]}
    if len(index) != len(manifest["records"]) or len(labels) != len(edits["records"]) or set(labels) - index.keys():
        raise ValueError("loader_duplicate_or_unknown_id")
    membership = {}
    families, pixels, captions, handles = {}, {}, {}, {}
    for split, ids in manifest["splits"].items():
        if split not in ("train", "val", "test"):
            raise ValueError("unknown_split")
        for identifier in ids:
            if identifier in membership or identifier not in index:
                raise ValueError("overlapping_or_unknown_split_membership")
            membership[identifier] = split
    if set(membership) != set(index):
        raise ValueError("incomplete_split_membership")
    for row in index.values():
        split = membership[row["id"]]
        if split != row["split"]:
            raise ValueError("split_metadata_mismatch")
        for table, key in (
            (families, row["source_group_id"]),
            (captions, family(row["original_prompt"])),
            (handles, row.get("upstream_image_paths", {}).get("original_image")),
        ):
            if key and table.setdefault(key, split) != split:
                raise ValueError("source_family_leakage")
        for role in ("source_image", "target_image"):
            if role in row and pixels.setdefault(row[role]["pixel_sha256"], split) != split:
                raise ValueError("image_pixel_leakage")
    result = {"train": [], "val": []}
    for row in edits["records"]:
        original = index[row["id"]]
        if (
            any(row[k] != original[k] for k in ("split", "source_group_id"))
            or row["instruction"] != original["edit_prompt"]
        ):
            raise ValueError("label_metadata_mismatch")
        if (
            row["split"] not in result
            or row["status"] != "candidate"
            or row["problems"]
            or not row["eligible_for_training"]
        ):
            continue
        source, target = (EditGraph.from_dict(row[k]) for k in ("source_graph", "target_graph"))
        operations = tuple(Operation.from_dict(o) for o in row["operations"])
        source, target, operations = validate(source, target, operations)
        result[row["split"]].append(
            Example(row["id"], row["split"], row["source_group_id"], row["instruction"], source, target, operations)
        )
    return result


def export(output, path, csv_path, review_directory, destination):
    from tools.instructpix2pix_caption_graph_v1.review import audit

    path, manifest, edits = check_compilation(output, path)
    gate = audit(output, path, review_directory, csv_path)
    examples = load_examples(path)
    rejected = set(gate["rejected_ids"])
    examples = {k: [e for e in rows if e.id not in rejected] for k, rows in examples.items()}
    problems = []
    if not gate["ready"]:
        problems.append("human_review_gate_not_met")
    if sum(len(v) for v in examples.values()) < 300:
        problems.append("fewer_than_300_usable_train_val_examples")
    if not all(examples.values()):
        problems.append("empty_train_or_validation_partition")
    receipts = image_receipts(output)
    if any(receipts.get(e.id, {}).get("status") != "available" for rows in examples.values() for e in rows):
        problems.append("selected_images_unavailable_or_unverified")
    for rows in examples.values():
        for example in rows:
            for role in ("source_image", "target_image"):
                image = receipts.get(example.id, {}).get(role)
                if image and sha256(ROOT / image["path"]) != image["sha256"]:
                    problems.append("selected_image_bytes_changed")
    if problems:
        raise ValueError(
            json.dumps(
                {"export_blocked": problems, "gate": gate, "counts": {k: len(v) for k, v in examples.items()}},
                ensure_ascii=False,
            )
        )
    selected = {e.id for rows in examples.values() for e in rows}
    bundle = {
        "schema_version": 1,
        "dataset": VERSION,
        "manifest_sha256": sha256(path / "manifest.json"),
        "edits_sha256": sha256(path / "edits.json"),
        "review_gate": gate,
        "label_origin": "caption_pseudo_labels_with_sampled_human_visual_audit; not_exhaustive_GT",
        "model_input_fields": ["source_graph", "instruction"],
        "records": [r for r in edits["records"] if r["id"] in selected],
        "splits": {k: [e.id for e in v] for k, v in examples.items()},
        "training_started": False,
    }
    write_once(Path(destination) / "dataset.json", bundle)
    return {"export": str(Path(destination) / "dataset.json"), "examples": len(selected), "training_started": False}


def prepare_tagger():
    # The existing English POS model is ~6 MB and contains no vision weights.
    for suffix, expected in TAGGER_FILES.items():
        name = f"averaged_perceptron_tagger_eng.{suffix}.json"
        path = TAGGER / "taggers/averaged_perceptron_tagger_eng" / name
        if not path.exists():
            response = requests_get_tagger(name)
            if hashlib.sha256(response).hexdigest() != expected:
                raise ValueError("downloaded_tagger_hash_mismatch")
            from tools.prepare_magicbrush import fixed

            fixed(path, response)
        if sha256(path) != expected:
            raise ValueError("existing_pinned_tagger_changed")


def requests_get_tagger(name):
    import io
    import zipfile

    import requests

    response = requests.get(
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip",
        timeout=60,
    )
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return archive.read(f"averaged_perceptron_tagger_eng/{name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "inspect",
            "plan",
            "prepare",
            "pilot",
            "compile",
            "review-build",
            "review-serve",
            "review-audit",
            "expand",
            "export",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--retry-images", action="store_true")
    parser.add_argument("--compilation", type=Path)
    parser.add_argument("--review-directory", type=Path)
    parser.add_argument("--reviews-csv", type=Path)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--port", type=int, default=8774)
    args = parser.parse_args(argv)
    from tools.instructpix2pix_caption_graph_v1 import review

    if args.stage == "review-serve":
        return review.serve(args.output, args.compilation, args.review_directory, args.port)
    with exclusive(args.output):
        if args.stage == "inspect":
            source = dataset.inspect(args.output)
            result = {k: v for k, v in source.items() if k != "shards"}
            result["shard_count"] = len(source["shards"])
        elif args.stage == "plan":
            plan = dataset.plan(args.output, count=args.count, seed=args.seed, workers=args.workers)
            result = {k: v for k, v in plan.items() if k != "records"}
        elif args.stage == "prepare":
            prepare_tagger()
            result = prepare(args.output, images=args.images, retry_images=args.retry_images)
        elif args.stage in ("pilot", "compile"):
            if args.stage == "pilot":
                dataset.plan(args.output, count=100, seed=args.seed, workers=args.workers)
                prepare_tagger()
                prepare(args.output, images=args.images)
            result = compile_dataset(args.output)
        elif args.stage == "review-build":
            result = review.build(args.output, args.compilation, args.review_directory)
        elif args.stage == "review-audit":
            result = review.audit(args.output, args.compilation, args.review_directory, args.reviews_csv)
        elif args.stage == "expand":
            if not args.pilot_output or not args.reviews_csv or args.count <= 100:
                parser.error("expand requires --pilot-output, --reviews-csv, --count > 100")
            gate = review.audit(args.pilot_output, args.compilation, args.review_directory, args.reviews_csv)
            if not gate["ready"]:
                raise ValueError("pilot_human_quality_gate_not_met")
            pilot_run = verify_run(args.pilot_output)
            if args.seed != pilot_run["seed"] or args.output.resolve() == args.pilot_output.resolve():
                raise ValueError("expansion_requires_same_seed_and_new_output")
            dataset.plan(args.output, count=args.count, seed=args.seed, workers=args.workers)
            write_once(
                args.output / "expansion.json",
                {"pilot_run_sha256": digest(pilot_run), "review_gate": gate, "approved_code_identity": code_identity()},
            )
            prepare(args.output, images=args.images)
            result = compile_dataset(args.output)
        else:
            if not all((args.reviews_csv, args.review_directory, args.destination)):
                parser.error("export requires --reviews-csv --review-directory --destination")
            result = export(args.output, args.compilation, args.reviews_csv, args.review_directory, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
