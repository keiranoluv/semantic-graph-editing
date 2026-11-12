"""CPU pilot diagnostics and byte-exact selection of an existing HUMAN review.

No generated review decisions, no graph repairs, no GPU calls. Full inference
must pass this review gate before loading a model.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.knowledge.transe.core import sha256
from tools.edit_model_v1.data import review_gate
from tools.infer_magicbrush_graphs import select_images
from tools.prepare_magicbrush import DATA, ROOT, digest, encode, fixed

DEFAULT_CONFIG = ROOT / "configs/level2/magicbrush-graphs-qwen25vl3b-v2.json"


def pilot_status(config, manifest_path=DATA / "manifest.json"):
    manifest = json.loads(manifest_path.read_text())
    folder = ROOT / config["output"]
    cache_file = folder / "manifest.json"
    cache = json.loads(cache_file.read_text()) if cache_file.is_file() else None
    if cache and (cache["dataset_manifest_sha256"] != sha256(manifest_path) or cache["config"] != config):
        raise ValueError("Pilot cache belongs to another dataset/config")
    if cache:
        for name, expected in cache["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Inference implementation changed: {name}; use a new version")
    counts, errors = Counter(), Counter()
    images = select_images(manifest, review_only=True)
    for image in images:
        path = folder / f"{image['sha256']}.json"
        if not path.is_file():
            counts["missing"] += 1
            continue
        row = json.loads(path.read_text())
        if not cache or row["image_sha256"] != image["sha256"] or row["inference_identity_sha256"] != digest(cache):
            raise ValueError("Prediction provenance mismatch")
        if row["status"] not in ("PASS", "ERROR"):
            raise ValueError("Unknown prediction status")
        counts[row["status"]] += 1
        if row["status"] == "ERROR":
            errors[row.get("error", "unknown")] += 1
    return {
        "expected_images": len(images),
        "image_counts": dict(counts),
        "error_counts": dict(errors),
        "all_pilot_image_graphs_structurally_valid": bool(images) and counts["PASS"] == len(images),
        "interpretation": "PASS is structural validation only, never human approval",
        "GPU_started": False,
    }


def verify_gate(config, csv_path, packet_path, manifest_path=DATA / "manifest.json"):
    status = pilot_status(config, manifest_path)
    if not status["all_pilot_image_graphs_structurally_valid"]:
        raise ValueError(
            f"All {status['expected_images']} original pilot image graphs are required; do not replace failed cases"
        )
    edits_path = ROOT / config["review_edits"]
    labels = {r["id"]: r for r in json.loads(edits_path.read_text())["records"]}
    manifest = json.loads(manifest_path.read_text())
    index = {r["id"]: r for r in manifest["records"]}
    for identifier in manifest["review_ids"]:
        for role, field in (("source_image", "source_prediction_sha256"), ("target_image", "target_prediction_sha256")):
            path = ROOT / config["output"] / f"{index[identifier][role]['sha256']}.json"
            if labels[identifier].get(field) != sha256(path):
                raise ValueError("Reviewed graph edit does not use the current image prediction")
    gate = review_gate(manifest_path, edits_path, packet_path, csv_path, minimum=20, minimum_correct_fraction=0.8)
    if not gate["ready"]:
        raise ValueError(
            f"Human quality gate not met: {gate['decision_counts']}; need 20 determinate reviews and >=80% correct"
        )
    return gate


def select_review(config, *, review_dir=None, manifest_path=DATA / "manifest.json"):
    review_dir = Path(review_dir) if review_dir else (ROOT / config["review_csv"]).parent.parent
    files = list((review_dir / "submissions").glob("review-*.csv"))
    if not files:
        raise ValueError("No human CSV saved from the viewer; no decisions will be fabricated")
    # Use newest save only: never silently select an older favorable review.
    source_csv = max(files, key=lambda p: (p.stat().st_mtime_ns, p.name))
    with source_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    hashes = {r.get("review_packet_sha256") for r in rows}
    if len(hashes) != 1:
        raise ValueError("Human CSV mixes evidence packets")
    packet_hash = hashes.pop()
    # Only accept a plain SHA256 filename within the archived packet directory.
    if (
        not isinstance(packet_hash, str)
        or len(packet_hash) != 64
        or any(c not in "0123456789abcdef" for c in packet_hash)
    ):
        raise ValueError("Invalid review packet hash")
    source_packet = review_dir / "packets" / f"{packet_hash}.json"
    gate = verify_gate(config, source_csv, source_packet, manifest_path)
    selected_csv, selected_packet = ROOT / config["review_csv"], ROOT / config["review_packet"]
    if any(not p.resolve().is_relative_to(ROOT / "reviews") for p in (selected_csv, selected_packet)):
        raise ValueError("Selected human evidence must stay under reviews/")
    selection = {
        "source_csv": str(source_csv),
        "source_packet": str(source_packet),
        "csv_sha256": sha256(source_csv),
        "packet_file_sha256": sha256(source_packet),
        "gate": gate,
        "selection_policy": "newest human-saved CSV only; byte-exact copies; no decision changes",
        "human_decisions_created": 0,
    }
    fixed(selected_csv, source_csv.read_bytes())
    fixed(selected_packet, source_packet.read_bytes())
    fixed(selected_csv.parent / "selection.json", encode(selection))
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--select-review", action="store_true", help="Pin latest existing human CSV, only if the gate passes"
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    result = select_review(config) if args.select_review else pilot_status(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
