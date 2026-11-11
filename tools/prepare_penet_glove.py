"""Pin verified GloVe 6B 300D mappings for the train-derived GQA vocabulary."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.features.glove import map_vocabulary, needed_tokens, read_glove_subset
from src.knowledge.transe.core import sha256
from tools.prepare_level2_gqa import write_fixed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    p.add_argument("--glove", type=Path, default=Path("data/external/glove/glove.6B.300d.txt"))
    p.add_argument("--provenance", type=Path, default=Path("data/external/glove/source-verified.json"))
    p.add_argument("--output", type=Path, default=Path("data/processed/level2/gqa-glove-v1"))
    p.add_argument("--report", type=Path, default=Path("reports/level2/penet-glove-v1.json"))
    args = p.parse_args(argv)
    provenance = json.loads(args.provenance.read_text())
    if sha256(args.glove) != provenance["text_sha256"]:
        raise ValueError("GloVe text hash mismatch")
    manifest = json.loads(args.manifest.read_text())
    vocabulary = manifest["vocabulary"]
    labels, predicates = vocabulary["label_to_idx"], vocabulary["predicate_to_idx"]
    vectors, lines = read_glove_subset(args.glove, needed_tokens([labels, predicates]))
    objects, object_info = map_vocabulary(labels, vectors, "object")
    relations, relation_info = map_vocabulary(predicates, vectors, "predicate")
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = args.output / "vectors.npz"
    if arrays.exists():
        with np.load(arrays, allow_pickle=False) as saved:
            if not np.array_equal(saved["objects"], objects) or not np.array_equal(saved["predicates"], relations):
                raise FileExistsError("Different existing GloVe mapping; use new output path")
    else:
        with arrays.open("xb") as f:
            np.savez(f, objects=objects, predicates=relations)
    report = {
        "schema_version": 1,
        "source": provenance,
        "manifest_sha256": sha256(args.manifest),
        "vocabulary_policy": "only project-train-derived vocabulary from frozen GQA manifest",
        "mapping_policy": "exact; object longest-word fallback; predicate known-word mean; fully OOV and ID0 seeded N(0,1)",
        "fallback_seed": 42,
        "fallback_seed_derivation": "first 8 bytes SHA256(glove-fallback-v1:seed:kind:label), big endian; NumPy default_rng",
        "upstream": "VL-Group/PENET@9c9f50777c66647799cb7a17dfb855aa98aecd1e/utils_motifs.py",
        "deviation": "fallback is label-keyed and safe for all-OOV phrases; upstream can divide by zero on all-OOV predicates",
        "glove_rows_scanned": lines,
        "dimension": 300,
        "arrays": str(arrays),
        "arrays_sha256": sha256(arrays),
        "objects": object_info,
        "predicates": relation_info,
        "test_annotation_opened": False,
        "code_sha256": {s: sha256(Path(s)) for s in ("src/features/glove.py", "tools/prepare_penet_glove.py")},
    }
    write_fixed(args.output / "manifest.json", report)
    write_fixed(args.report, report)
    print(json.dumps({"objects": object_info["counts"], "predicates": relation_info["counts"]}, indent=2))


if __name__ == "__main__":
    main()
