"""Build/serve a local, train-only GQA WordNet human review packet.

Read normalized records only from pinned train HDF5 shards, following the
Level 1 deterministic image/box review workflow. No annotation JSON, visual
encoder, training, mapping update or approval compilation is performed.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import mimetypes
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import h5py
from PIL import Image, ImageDraw

from scripts.prepare_knowledge import _save_outputs
from src.knowledge.transe.core import sha256

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("gqa_wordnet_review")
ORIGINAL_FIELDS = (
    "class_id",
    "label",
    "status",
    "synset_id",
    "definition",
    "candidates",
    "decision",
    "selected_synset",
    "reviewer",
    "notes",
)
EXPORT_FIELDS = ORIGINAL_FIELDS + ("train_image_ids", "evidence_status", "review_packet_sha256")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def under(path, parent):
    path = path.resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"Output must be inside {parent}: {path}")
    return path


def load_sample(sample_path, mapping_path, manifest_path, wordnet):
    """Bind every immutable ID, status and candidate to the original mapping."""
    manifest = json.loads(manifest_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    if mapping.get("source_version") != wordnet.get_version():
        raise ValueError("WordNet version differs from pinned mapping")
    labels = manifest["vocabulary"]["label_to_idx"]
    with sample_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ORIGINAL_FIELDS:
            raise ValueError("Review sample CSV schema changed")
        sample = list(reader)
    if not sample:
        raise ValueError("Empty review sample")
    seen = set()
    classes = []
    for source in sample:
        class_id = int(source["class_id"])
        if class_id in seen or not 0 < class_id < len(mapping["rows"]):
            raise ValueError("Repeated or unknown class ID")
        seen.add(class_id)
        row = mapping["rows"][class_id]
        if (
            row["class_id"] != class_id
            or labels.get(source["label"]) != class_id
            or row["label"] != source["label"]
            or row["status"] != source["status"]
            or (row["synset_id"] or "") != source["synset_id"]
        ):
            raise ValueError(f"Sample/mapping/vocabulary disagree at class {class_id}")
        # A frozen sample is input evidence, never an existing approval source.
        if any(source[k] for k in ("decision", "selected_synset", "reviewer", "notes")):
            raise ValueError("Expected untouched review sample with blank human decisions")
        candidates = []
        for candidate in row["candidates"]:
            sense = wordnet.synset(candidate["synset_id"])
            if (
                sense.pos() != "n"
                or sense.name() != candidate["synset_id"]
                or sense.definition() != candidate["definition"]
            ):
                raise ValueError("Candidate definition differs from pinned WordNet")
            candidates.append({"synset_id": sense.name(), "definition": sense.definition()})
        encoded = " | ".join(f"{c['synset_id']}: {c['definition']}" for c in candidates)
        if source["candidates"] != encoded:
            raise ValueError("Sample candidate list differs from mapping")
        current_definition = wordnet.synset(row["synset_id"]).definition() if row["synset_id"] else ""
        if source["definition"] != current_definition:
            raise ValueError("Sample current definition differs from WordNet")
        classes.append(
            {
                "class_id": class_id,
                "label": row["label"],
                "status": row["status"],
                "synset_id": row["synset_id"],
                "definition": current_definition,
                "candidates": candidates,
                "original": source,
            }
        )
    return manifest, classes


def valid_box(box, width, height):
    if (
        len(box) != 4
        or min(width, height) <= 0
        or not all(math.isfinite(v) for v in box)
        or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
    ):
        raise ValueError("Invalid normalized object box")


def select_examples(records, classes, *, per_class=5, seed=2026):
    """One largest matching instance per image, then lowest seeded SHA256 ranks."""
    if not 3 <= per_class <= 5:
        raise ValueError("per-class must be 3..5")
    selected = {row["class_id"]: [] for row in classes}
    counts, invalid = Counter(), Counter()
    for record in records:
        best = {}
        width, height = record["width"], record["height"]
        for obj in record["objects"]:
            class_id = obj["label_id"]
            if class_id not in selected:
                continue
            box = obj["bbox_xyxy"]
            try:
                valid_box(box, width, height)
            except (ValueError, TypeError):
                invalid[class_id] += 1
                continue
            key = (-(box[2] - box[0]) * (box[3] - box[1]), obj["object_id"])
            if class_id not in best or key < best[class_id][0]:
                best[class_id] = (key, obj)
        for class_id, (_, obj) in best.items():
            counts[class_id] += 1
            image_id = record["image_id"]
            rank = hashlib.sha256(f"gqa-wordnet-human-v1:{seed}:{class_id}:{image_id}".encode()).hexdigest()
            entry = {
                "image_id": image_id,
                "object_id": obj["object_id"],
                "annotation_size": [width, height],
                "annotation_box_xyxy": obj["bbox_xyxy"],
                "rank": rank,
            }
            selected[class_id].append(entry)
            selected[class_id].sort(key=lambda e: (e["rank"], e["image_id"]))
            del selected[class_id][per_class:]
    return selected, counts, invalid


def train_records(cache_dir, manifest, manifest_path, provenance):
    metadata = json.loads((cache_dir / "manifest.json").read_text())
    identity = metadata["identity"]
    groups = [manifest["splits"][k] for k in ("train", "val", "test")]
    if any(len(g) != len(set(g)) for g in groups) or any(
        set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)
    ):
        raise ValueError("Duplicate or overlapping manifest splits")
    if (
        not metadata["complete"]
        or identity["manifest_sha256"] != sha256(manifest_path)
        or identity["splits"]["train"] != groups[0]
        or identity["annotation_sha256"] != manifest["annotation_sha256"]["train_sceneGraphs.json"]
    ):
        raise ValueError("Train cache differs from GQA manifest")
    allowed, seen = set(groups[0]), set()
    identity_hash = canonical_hash(identity)
    provenance.update(
        cache_manifest_sha256=sha256(cache_dir / "manifest.json"),
        declared_image_archive_sha256=identity["image_archive_sha256"],
        train_shards_verified=[],
    )
    for entry in sorted(metadata["shards"], key=lambda e: e["file"]):
        ids = set(entry["ids"])
        if not ids & allowed:
            continue  # Do not open/hash/decode validation shards at all.
        if not ids <= allowed or len(ids) != len(entry["ids"]) or ids & seen:
            raise ValueError("Train shard has non-train or repeated images")
        name = Path(entry["file"])
        if name.name != entry["file"] or entry["identity_sha256"] != identity_hash:
            raise ValueError("Invalid train shard identity/path")
        path = cache_dir / name
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Train shard hash mismatch: {path}")
        provenance["train_shards_verified"].append({"file": str(name), "sha256": entry["sha256"]})
        with h5py.File(path, "r") as shard:
            if shard.attrs["identity_sha256"] != identity_hash:
                raise ValueError("Train HDF5 identity mismatch")
            for image_id in entry["ids"]:
                record = json.loads(shard[image_id]["record"][()].decode())
                if record["image_id"] != image_id:
                    raise ValueError("Train image ID differs from record")
                seen.add(image_id)
                # Predicates/attributes/features are never used for selection or UI.
                yield {
                    "image_id": image_id,
                    "width": record["width"],
                    "height": record["height"],
                    "objects": record["objects"],
                }
        if len(provenance["train_shards_verified"]) % 32 == 0:
            print(f"[TRAIN ONLY] scanned {len(seen)}/{len(allowed)} images", flush=True)
    if seen != allowed:
        raise ValueError("Train cache does not cover the complete train split")
    provenance["train_images_scanned"] = len(seen)


def render_evidence(jpeg_bytes, example):
    """Annotate real JPEG pixels; resize coordinates from annotation to JPEG."""
    with Image.open(io.BytesIO(jpeg_bytes)) as source:
        image = source.convert("RGB")
    aw, ah = example["annotation_size"]
    width, height = image.size
    sx, sy = width / aw, height / ah
    x1, y1, x2, y2 = example["annotation_box_xyxy"]
    box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    valid_box(box, width, height)
    bounds = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
    crop = image.crop(tuple(bounds))
    annotated = image.copy()
    ImageDraw.Draw(annotated).rectangle(tuple(box), outline=(255, 30, 60), width=max(2, round(max(image.size) / 250)))
    outputs = []
    for item in (annotated, crop):
        stream = io.BytesIO()
        item.save(stream, format="JPEG", quality=92)
        outputs.append(stream.getvalue())
    return outputs, {
        "jpeg_size": [width, height],
        "jpeg_box_xyxy": box,
        "crop_bounds_xyxy": bounds,
        "source_jpeg_sha256": hashlib.sha256(jpeg_bytes).hexdigest(),
    }


def build(args):
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()
    output = under(args.output, ROOT / "reviews")
    assets = under(args.assets, ROOT / "data")
    manifest, classes = load_sample(args.sample, args.mapping, args.manifest, wordnet)
    provenance = {
        "schema_version": 1,
        "split": "project_train_only",
        "seed": args.seed,
        "requested_examples_per_class": args.per_class,
        "sample_csv_sha256": sha256(args.sample),
        "mapping_sha256": sha256(args.mapping),
        "gqa_manifest_sha256": sha256(args.manifest),
        "wordnet_version": wordnet.get_version(),
        "selection": "lowest SHA256(gqa-wordnet-human-v1:seed:class_id:image_id); distinct images; largest matching object, lowest object ID tie",
        "annotation_json_files_opened": [],
        "validation_shards_opened": False,
        "test_annotations_opened": False,
        "human_reviews_created": 0,
        "predicate_annotations_used_for_decisions": False,
        "assets_directory": str(assets.relative_to(ROOT)),
        "source_code_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), *sorted(TEMPLATE.iterdir())] if p.is_file()
        },
        "reference_workflow": ["scripts/prepare_wordnet_review_examples.py", "scripts/prepare_level1_review_images.py"],
        "image_verification": "ZIP member CRC checked by zipfile; per-source JPEG SHA256; full archive digest declared by pinned cache, not recomputed",
    }
    selected, counts, invalid = select_examples(
        train_records(args.cache, manifest, args.manifest, provenance),
        classes,
        per_class=args.per_class,
        seed=args.seed,
    )
    with zipfile.ZipFile(args.archive) as archive:
        index = {}
        for info in archive.infolist():
            if info.filename.lower().endswith(".jpg") and not info.filename.startswith("__MACOSX/"):
                image_id = Path(info.filename).stem
                if image_id in index:
                    raise ValueError("Duplicate archive image ID")
                index[image_id] = info.filename
        for row in classes:
            row.update(
                examples=[],
                image_errors=[],
                eligible_train_images=counts[row["class_id"]],
                invalid_box_instances=invalid[row["class_id"]],
            )
            for example in selected[row["class_id"]]:
                image_id = example["image_id"]
                try:
                    member = index[image_id]
                    images, metadata = render_evidence(archive.read(member), example)
                except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
                    row["image_errors"].append({"image_id": image_id, "reason": str(error)})
                    continue
                directory = f"class-{row['class_id']}"
                name = f"{image_id}-object-{example['object_id']}"
                context_name, crop_name = f"{name}-box.jpg", f"{name}-crop.jpg"
                context_path, crop_path = f"{directory}/{context_name}", f"{directory}/{crop_name}"
                _save_outputs(assets / directory, {context_name: images[0], crop_name: images[1]}, dry_run=False)
                row["examples"].append(
                    {
                        **example,
                        **metadata,
                        "archive_member": member,
                        "context_path": str((assets / context_path).relative_to(ROOT)),
                        "crop_path": str((assets / crop_path).relative_to(ROOT)),
                        "context_sha256": hashlib.sha256(images[0]).hexdigest(),
                        "crop_sha256": hashlib.sha256(images[1]).hexdigest(),
                    }
                )
            row["evidence_status"] = (
                "supported"
                if len(row["examples"]) >= 3
                else "insufficient_examples"
                if row["examples"]
                else "no_image_evidence"
            )
            row["mapping_issue"] = (
                "no_wordnet_candidates"
                if not row["candidates"]
                else "ambiguous_sense"
                if row["status"] == "ambiguous"
                else "context_review_required"
            )
    packet = {"provenance": provenance, "classes": classes, "export_fields": EXPORT_FIELDS}
    packet["packet_sha256"] = canonical_hash(packet)
    dataset = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = (TEMPLATE / "index.html").read_text().replace("__REVIEW_DATA__", dataset)
    _save_outputs(
        output,
        {
            "index.html": page.encode(),
            "review-data.json": json_bytes(packet),
            "viewer.js": (TEMPLATE / "viewer.js").read_bytes(),
            "viewer.css": (TEMPLATE / "viewer.css").read_bytes(),
        },
        dry_run=False,
    )
    print(
        f"[READY] {len(classes)} labels, {sum(len(r['examples']) for r in classes)} examples: {output / 'index.html'}"
    )
    for row in classes:
        if row["mapping_issue"] != "context_review_required" or row["evidence_status"] != "supported":
            print(
                f"[REVIEW] {row['class_id']} {row['label']}: {row['mapping_issue']}; {len(row['examples'])} images; {row['evidence_status']}"
            )
    return packet


def export_csv(packet, reviews):
    """Validate explicit human choices and export original IDs/labels verbatim."""
    known = {str(row["class_id"]): row for row in packet["classes"]}
    if not isinstance(reviews, dict) or set(reviews) - known.keys():
        raise ValueError("Unknown review class IDs")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=EXPORT_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    for key, row in known.items():
        item = reviews.get(key, {})
        if not isinstance(item, dict) or any(
            not isinstance(item.get(k, ""), str) for k in ("decision", "selected_synset", "reviewer", "notes")
        ):
            raise ValueError("Review fields must be strings")
        decision, sense, reviewer = [item.get(k, "").strip() for k in ("decision", "selected_synset", "reviewer")]
        candidates = {c["synset_id"] for c in row["candidates"]}
        if decision not in ("", "accept", "defer") or sense and sense not in candidates:
            raise ValueError(f"Invalid decision/synset: {row['label']}")
        if decision and not reviewer:
            raise ValueError(f"Reviewer name required: {row['label']}")
        if decision == "accept" and (not sense or not row["examples"]):
            raise ValueError(f"Accept requires selected candidate and actual image evidence: {row['label']}")
        record = {
            **row["original"],
            "decision": decision,
            "selected_synset": sense,
            "reviewer": reviewer,
            "notes": item.get("notes", ""),
            "train_image_ids": ";".join(e["image_id"] for e in row["examples"]),
            "evidence_status": row["evidence_status"],
            "review_packet_sha256": packet["packet_sha256"],
        }
        writer.writerow(record)
    return stream.getvalue().encode("utf-8-sig")


def review_handler(output, assets, packet):
    """Serve only viewer/generated assets; explicit saves create new CSV files."""
    output, assets = output.resolve(), assets.resolve()
    prefix = "/" + assets.relative_to(ROOT).as_posix() + "/"

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, data, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path in ("/", "/index.html", "/viewer.js", "/viewer.css", "/review-data.json"):
                target = output / (path.lstrip("/") or "index.html")
            elif path.startswith(prefix):
                target = (assets / path[len(prefix) :]).resolve()
                if not target.is_relative_to(assets):
                    return self.send(404, b"Not found", "text/plain")
            else:
                return self.send(404, b"Not found", "text/plain")
            if not target.is_file():
                return self.send(404, b"Not found", "text/plain")
            self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

        def do_POST(self):
            if self.path != "/api/save":
                return self.send(404, b"Not found", "text/plain")
            origin = self.headers.get("Origin")
            if origin != f"http://{self.headers.get('Host')}":
                return self.send(403, b"Same-origin browser request required", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if payload.get("packet_sha256") != packet["packet_sha256"]:
                    raise ValueError("Review packet version differs")
                content = export_csv(packet, payload["reviews"])
                name = f"review-{hashlib.sha256(content).hexdigest()[:16]}.csv"
                _save_outputs(output / "submissions", {name: content}, dry_run=False)
                self.send(
                    200,
                    json_bytes({"path": str((output / "submissions" / name).relative_to(ROOT))}),
                    "application/json",
                )
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, json_bytes({"error": str(error)}), "application/json")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("build", help="Build immutable HTML and real train image evidence")
    prepare.add_argument("--sample", type=Path, default=ROOT / "reports/level2/gqa-wordnet-v1/review_sample.csv")
    prepare.add_argument("--mapping", type=Path, default=ROOT / "data/knowledge/gqa-wordnet-v1/wordnet_mapping.json")
    prepare.add_argument("--manifest", type=Path, default=ROOT / "data/processed/level2/gqa-v1/manifest.json")
    prepare.add_argument("--cache", type=Path, default=ROOT / "data/cache/gqa-full-clip-v1")
    prepare.add_argument("--archive", type=Path, default=ROOT / "data/raw/gqa/archives/images.zip")
    prepare.add_argument("--per-class", type=int, choices=(3, 4, 5), default=5)
    prepare.add_argument("--seed", type=int, default=2026)
    server = actions.add_parser("serve", help="Loopback viewer with explicit CSV saves under reviews/")
    server.add_argument("--port", type=int, default=8765)
    for command in (prepare, server):
        command.add_argument("--output", type=Path, default=ROOT / "reviews/gqa-wordnet-human-v1")
        command.add_argument("--assets", type=Path, default=ROOT / "data/knowledge/gqa-wordnet-human-v1/images")
    args = parser.parse_args(argv)
    if args.action == "build":
        build(args)
    else:
        output, assets = under(args.output, ROOT / "reviews"), under(args.assets, ROOT / "data")
        packet = json.loads((output / "review-data.json").read_text())
        if canonical_hash({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"]:
            raise ValueError("Review packet content hash mismatch")
        if packet["provenance"]["assets_directory"] != str(assets.relative_to(ROOT)):
            raise ValueError("Assets directory differs from review packet")
        with ThreadingHTTPServer(("127.0.0.1", args.port), review_handler(output, assets, packet)) as httpd:
            print(f"Review viewer: http://127.0.0.1:{httpd.server_port}/", flush=True)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
