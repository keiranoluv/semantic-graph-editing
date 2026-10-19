# C.1 — TransE training skeleton

This is an **additive** TransE implementation; it does not modify MLP/GAT,
existing knowledge artifacts or any previous training runs.

## Input provenance

Read only an existing B.1 WordNet or B.2.1 ConceptNet KG directory:
`entities.json`, `relations.json`, `triples.tsv`, `metadata.json`.
ConceptNet evidence duplicates are collapsed to unique `(head, relation, tail)`
triples. `wn:` and `cn:` entity namespaces remain separate; no cross-KG merging,
lexical alignment, `SameAs` or candidate-pair consumption occurs.

## Modes and validation

- **C-Draft (`--mode draft`)**: isolated KG technical smoke test, even when
  `ready_for_gat_kg_training=false`. Never report as official accuracy or
  use its embeddings in official GAT+KG ablations.
- **C-Reviewed (`--mode reviewed`)**: fail-closed. Requires pinned identical KG
  input file hashes, an independently prepared human-reviewed manifest with
  150 individual class decisions/reviewer/evidence/rationale records, and
  `ready_for_gat_kg_training=true` in KG metadata. No automatic promotion of
  B.2.2 candidate alignments; this milestone has **no review compiler**.
  A manifest is an audit input, not independently verifiable evidence that a
  review occurred; release policy additionally requires a human audit.

Reviewed manifest schema (must be authored and verified **by people**, not
produced by this tool): `schema_version: 1`, `review_status: human_approved`,
`kg_source: wordnet|conceptnet`, `kg_path: <absolute resolved KG directory>`,
`kg_sha256: {entities.json,relations.json,triples.tsv,metadata.json: SHA256}`,
`reviews`: exactly 150 entries containing `class_id` 1..150, `decision`
(`accept|reject|defer`), `reviewer`, `evidence_reference`, `rationale`.
Provisional B.1/B.2.1 artifacts cannot pass the Reviewed gate.

## Commands

```bash
python -m pytest tests/knowledge/test_transe_c1.py -q
python -m scripts.train_transe --help
python -m scripts.train_transe \
  --kg-dir data/knowledge/wordnet-ai-draft-v1/kg \
  --mode draft --output-dir runs/transe/c1-wordnet-draft-seed42 \
  --seed 42 --epochs 10 --dry-run
python -m scripts.train_transe \
  --kg-dir data/knowledge/wordnet-ai-draft-v1/kg \
  --mode draft --output-dir runs/transe/c1-wordnet-draft-seed42 \
  --seed 42 --epochs 10
```

To train ConceptNet use `--kg-dir data/knowledge/conceptnet-b2-v1` and a fresh
`--output-dir`; avoid mixing provenance. Install `torch`, `numpy`, `pytest`.
CUDA may be enabled using `--device cuda`; deterministic algorithms are set,
but bitwise GPU reproducibility across hardware/drivers is not guaranteed.

## Run outputs

Each new exclusive run directory contains `manifest.json` with input hashes,
`split_indices.npz` storing the exact train/val/test triples, one `.pt`
checkpoint per epoch (model/optimizer/RNG), and `metrics.json` recording
filtered head-and-tail MRR/Hits@1/3/10. Validation is computed every epoch;
test only after all epochs. Split is unique-triple-level, seeded random
holdout with all positive triples used for filtering corruptions/rankings.
This skeleton does **not** support resuming from checkpoints or selecting best
checkpoint; checkpoints are snapshots. It does not prevent inverse/near-duplicate
information leakage (random triple holdout is not inductive evaluation).
For publication-quality conclusions, add relation-stratified splits,
entity-coverage checks, baseline comparisons and at least three seeds.

**Not PredCls metrics.** These link prediction metrics are NOT equivalent to
VG150 scene-graph Recall@50/100 or mean Recall; those must be measured later
with the same MLP/GAT/GAT+KG image split protocol.
