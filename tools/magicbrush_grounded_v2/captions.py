"""Verify and rewrap existing BLIP captions, without changing parent artifacts."""

import json

from src.knowledge.transe.core import sha256
from tools.prepare_magicbrush import ROOT, digest, encode, fixed

CAPTION_KEYS = ("caption_model", "caption_revision", "caption_max_new_tokens", "seed", "dtype", "batch_size")


def verified_captions(config, output):
    source = (ROOT / config["caption_import_source"]).resolve()
    if not source.is_relative_to(ROOT / "data/processed/level2") or source == output.resolve():
        raise ValueError("Caption parent must be a separate versioned data directory")
    if not (source / "inference-identity.json").exists():
        return None
    identity_path = source / "inference-identity.json"
    identity = json.loads(identity_path.read_text())
    # The historical source files are frozen; refuse silently relabeling a
    # changed parent cache or using another caption model/recipe.
    for name, expected in identity["source_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError("Parent caption source fingerprint differs")
    if any(identity["config"][k] != config[k] for k in CAPTION_KEYS):
        raise ValueError("Parent caption model/recipe differs")
    parent_path = source / "manifest.json"
    if sha256(parent_path) != identity["dataset_manifest_sha256"]:
        raise ValueError("Parent dataset manifest changed")
    parent = json.loads(parent_path.read_text())
    current = json.loads((output / "manifest.json").read_text())
    if any(parent[k] != current[k] for k in ("records", "splits", "review_ids")):
        raise ValueError("Parent pilot records/splits/selection differ")
    image_ids = sorted({r["source_image"]["sha256"] for r in current["records"] if r["id"] in current["review_ids"]})
    entries = []
    for image_sha in image_ids:
        path = source / "captions" / f"{image_sha}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if (
            row["identity_sha256"] != digest(identity)
            or row["id"] != image_sha
            or not isinstance(row["caption"], str)
            or not row["caption"].strip()
        ):
            raise ValueError("Parent caption identity/content is invalid")
        entries.append(
            {
                "id": image_sha,
                "caption": row["caption"],
                "source_path": str(path.relative_to(ROOT)),
                "source_sha256": sha256(path),
            }
        )
    return {
        "parent_identity_path": str(identity_path.relative_to(ROOT)),
        "parent_identity_sha256": sha256(identity_path),
        "parent_identity_digest": digest(identity),
        "entries": entries,
    }


def import_captions(config, output, fingerprint):
    bundle = verified_captions(config, output)
    if bundle is None:
        return 0
    fixed(output / "caption-import.json", encode(bundle))
    for entry in bundle["entries"]:
        fixed(
            output / "captions" / f"{entry['id']}.json",
            encode(
                {
                    "id": entry["id"],
                    "identity_sha256": fingerprint,
                    "caption": entry["caption"],
                    "imported_from": {
                        "path": entry["source_path"],
                        "sha256": entry["source_sha256"],
                        "parent_identity_sha256": bundle["parent_identity_sha256"],
                    },
                }
            ),
        )
    print(f"[REUSED] {len(bundle['entries'])} verified BLIP captions; no caption GPU rerun", flush=True)
    return len(bundle["entries"])
