# Level 3 VN v2 — Provenance-only selection (local evaluation pilot)

## Goal

Freeze the exact 30 selected images with reproducible upstream identity, not a
rights-clearance assertion. Original requirement of 100 images and 150-200
editing instructions remains for the full benchmark. This stage only prepares
a small pilot and 45 blank edit slots.

## Inputs

- `data/processed/level3/vietnam-benchmark-v2/pilot/candidate-manifest.json`
- `data/processed/level3/vietnam-benchmark-v2/pilot/review.csv`, 15 `accept` Sidewalk,
  15 `accept` VietTravelVQA-v2; other 30 rows remain `pending`.
- Image bytes stored at each candidate's immutable local path.
- `source_revision_sha`: exactly 40 lowercase hexadecimal characters; matching
  `https://huggingface.co/datasets/<repo>/blob/<sha>/<relative_path>`.
- Existing `configs/level3/vietnam-benchmark-v2.json` unchanged and manifest
  `config_sha256` still matching.

## Run

```bash
python -m pytest -q tests/level3/test_vietnam_benchmark_v2.py
bash scripts/run_level3_vn_provenance_pilot_v2.sh
```

Do not delete or overwrite an existing `pilot/locked/` directory. A new selection
or different review protocol requires a new versioned output directory.

## Output contract

`locked/manifest.json` contains ID, dataset, source relative path, exact pinned
upstream URL, SHA256, dimensions, local path and `selection_mode=provenance_only`.
`locked/source_graph_tasks.jsonl` contains 30 rows with null graphs. The model
must fill each row later and record model version/checkpoint and image hash;
no empty graph should be considered a real observed scene graph.
`locked/edit_slots.csv` contains 45 blank prompts for later authoring.

Metadata provided by the upstream dataset card is not enough to establish a
specific image's copyright/creator, especially for Commons-sourced images;
selecting an image does not mean it is cleared for downstream distribution.
Check privacy before uploading images to external APIs.

## Next milestone: source graph generation

1. Select frozen, inference-only object detection or a pretrained SGG model;
   record model hashes, object thresholds and vocabulary.
2. Generate a **candidate** directed scene graph for each of 30 images.
3. Manually check object existence, object identity, attribute and relation
   correctness; do not invent invisible objects/edges.
4. Store graph annotation and review evidence separately, with the image hash
   as the join key. Do not train/fit a detector on these images.
5. Only then prepare 45 image edit instructions. Training of the EMNLP Task B
   graph edit model on this new image-derived graph representation is **not
   assumed** to transfer without normalization and domain checks.
