# Milestone B.2.1 — ConceptNet 5.7 offline extraction (draft, not training-ready)

## What this step does

B.1 produced a WordNet-only draft KG (124 provisional VG150 class links,
122 unique roots, 967 entities, 1,156 edges in the last observed run).
B.2.1 adds **a second, independent ConceptNet subgraph**, rather than merging
WordNet synsets and ConceptNet concepts prematurely. This is a prerequisite for
B.2.2 (curated linking/merging), not final GAT+KG or TransE input.

Key protections:

- Uses only VG150 **label strings** from the existing mapping, never VG images,
  split labels, GT predicates, or train/validation/test relation frequencies.
- Requires an **on-disk**, exact SHA-256-pinned ConceptNet 5.7.0 assertions
  snapshot; no API calls and no silent implicit downloads.
- Streams five-field raw `.csv.gz`, `.tsv.gz`, `.csv`, or `.tsv` assertions
  (tab-separated despite the official `.csv.gz` name).
- Selects **one-hop** directed English assertions incident on the normalized
  VG150 class label. Exact lexical matches are *candidates*, not identity.
- Allows `IsA`, `PartOf`, `HasA`, `MadeOf`, `UsedFor`, `CapableOf`, `AtLocation`,
  `HasProperty`, `ReceivesAction`, `RelatedTo`, with `weight >= 1.0` by default.
  These relations are not substitutes for VG image predicates.
- Excludes `dataset=/d/wordnet...` assertions to avoid accidentally duplicating
  WordNet 3.1 facts or treating them as independent common-sense evidence.
- Preserves each assertion's original URI, dataset, license and weight.
  Duplicate triples with different source evidence remain distinct lines;
  `unique_triple_count` is reported separately from `edge_evidence_count`.
- Uses `cn:/c/en/...` node IDs distinct from `wn:<synset>`; **NO `SameAs` links**.
- Always sets `human_verified_class_count=0` and
  `ready_for_gat_kg_training=false`.
- Bounds retained evidence to `--max-edges` (default 50,000); refuses writes
  that conflict with existing artifacts.

## Quick validation (no ConceptNet download needed)

From the repository root after installing the patch:

```bash
python -m pytest tests/knowledge/test_conceptnet_builder.py -q
python -m scripts.build_conceptnet_graph --help
```

Tests use **invented synthetic assertions** only. They check the 5-column
format, gzip, SHA mismatch, class alignment, English terms, filtering, and
idempotent output. Synthetic assertions must never be reported as ConceptNet
facts, scientific data or real experimental KG statistics.

## Real ConceptNet 5.7.0 data

Official index: https://github.com/commonsense/conceptnet5/wiki/Downloads

Official 5.7.0 assertions link published in that index:
https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

This is a large snapshot: download only once, on a machine with sufficient
storage/network resources. If S3 access fails, retrieve from the official
index; never replace with an unversioned API result while retaining the 5.7.0
provenance label. The pipeline does not download automatically.

```bash
mkdir -p data/external/conceptnet
curl -fL --retry 3 \
  'https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz' \
  -o data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz
sha256sum data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz
```

Read the hash printed by `sha256sum`, then run (replace `YOUR_SHA256_HEX_HERE`):

```bash
python -m scripts.build_conceptnet_graph \
  --assertions data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
  --expected-sha256 YOUR_SHA256_HEX_HERE \
  --dry-run
python -m scripts.build_conceptnet_graph \
  --assertions data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
  --expected-sha256 YOUR_SHA256_HEX_HERE
```

The SHA-256 obtained from the same download detects later changes, but alone
is **not independent authenticity verification**. For reproducible group runs,
verify the dataset origin and commit the exact download URL, download date,
version and digest in your research record. Do not commit the large dataset.
Run with a new `--output-dir` if you intentionally change policy or snapshot.

Default class input:
`data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json`.
Although this file contains 124 AI-provisional WordNet links, the ConceptNet
extractor uses only the **150 VG150 labels**, without trusting those senses.

Outputs under `data/knowledge/conceptnet-b2-v1/`:

- `entities.json` — deterministic `cn:` node namespace.
- `relations.json` — permitted directed relation IDs.
- `triples.tsv` — subject/relation/object IDs, assertion URI, dataset,
  per-edge license and weight.
- `class_concept_candidates.json` — 151 class entries including background,
  each with zero or more **unverified lexical** candidates.
- `review_queue.csv` — 150 review rows (blank decisions/reviewers).
- `metadata.json` — input digest, canonical label-map digest, filters,
  counts, curation tier, licensing and always-false training readiness.

**License and attribution.** ConceptNet's complete dataset is shared under
Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0). Any redistribution
of ConceptNet-derived graph data must respect attribution and applicable
ShareAlike terms. Keep dataset and per-assertion license metadata intact.
Research reference: Robyn Speer, Joshua Chin, Catherine Havasi (2017),
*ConceptNet 5.5: An Open Multilingual Graph of General Knowledge*, AAAI.

Sources:
- https://github.com/commonsense/conceptnet5/wiki/Downloads
- https://github.com/commonsense/conceptnet5/wiki/Edges
- https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet

## Gate before B.2.2 / C / D

B.2.1 **does not** add class-to-ConceptNet embedding vectors for training.
Do not simply concatenate WordNet and ConceptNet node IDs or add automatic
`SameAs(wn:dog.n.01, cn:/c/en/dog)` edges. These are not semantically
interchangeable. Review lexical candidates, decide whether a sense-level
bridge is justified, document agreement and unresolved cases, then create
a separately versioned, auditable B.2.2 bridge file. Do not use B.2.1 draft
as final TransE/GAT+KG ground truth.

For all human-review obligations, see `docs/knowledge/HUMAN_REVIEW_REQUIRED.md`.
