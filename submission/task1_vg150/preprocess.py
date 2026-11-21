"""Generate aligned VG150 metadata and benchmark-style splits."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

CORRUPTED_IMAGE_IDS = {1592, 1722, 4616, 4617}


def build_manifest(
    h5_path: Path,
    metadata_path: Path,
    output_dir: Path,
    validation_size: int = 5000,
    filter_empty_relations: bool = True,
) -> dict[str, list[int]]:
    """Select image-row splits without shuffling or dropping annotations."""
    if validation_size < 0:
        raise ValueError("validation_size must be nonnegative")

    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Raw image metadata must be a list")

    seen = set()
    aligned = []
    for item in raw:
        image_id = int(item["image_id"])
        if image_id in seen:
            raise ValueError(f"Duplicate image_id: {image_id}")
        seen.add(image_id)

        if image_id not in CORRUPTED_IMAGE_IDS:
            aligned.append(item)

    with h5py.File(h5_path, "r") as h5:
        split = np.asarray(h5["split"][:]).reshape(-1)
        first_box = np.asarray(h5["img_to_first_box"][:]).reshape(-1)
        first_rel = np.asarray(h5["img_to_first_rel"][:]).reshape(-1)

    count = len(split)

    if not (len(aligned) == len(first_box) == len(first_rel) == count):
        raise ValueError(
            "Metadata/HDF5 row count mismatch after corrupted-image filtering. " "Verify the VG150 source variant."
        )

    if any(int(item["width"]) <= 0 or int(item["height"]) <= 0 for item in aligned):
        raise ValueError("Invalid image dimensions")

    eligible = first_box >= 0

    if filter_empty_relations:
        eligible &= first_rel >= 0

    train_pool = np.flatnonzero((split == 0) & eligible).tolist()
    test = np.flatnonzero((split == 2) & eligible).tolist()

    if len(train_pool) < validation_size:
        raise ValueError("Insufficient images for validation split")

    manifest = {
        "train": train_pool[validation_size:],
        "val": train_pool[:validation_size],
        "test": test,
    }

    all_rows = manifest["train"] + manifest["val"] + manifest["test"]

    if len(all_rows) != len(set(all_rows)):
        raise ValueError("Overlapping split rows")

    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = output_dir / "image_data.json"
    split_path = output_dir / "splits.json"

    if aligned_path.exists() or split_path.exists():
        raise FileExistsError("Output already exists. Select a new output directory.")

    aligned_path.write_text(
        json.dumps(aligned, indent=2) + "\n",
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    """Generate metadata alignment and split manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--keep-empty-relations", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        h5_path=args.h5,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        filter_empty_relations=not args.keep_empty_relations,
    )

    for split, indices in manifest.items():
        print(f"{split}: {len(indices)} images")


if __name__ == "__main__":
    main()
