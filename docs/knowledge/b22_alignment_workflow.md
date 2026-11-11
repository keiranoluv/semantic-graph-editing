# B.2.2 — WordNet ↔ ConceptNet candidate alignment (draft only)

**Purpose:** build an inspectable, class-ID-aligned *candidate index* from the
WordNet A.2 **AI-provisional** KG and ConceptNet B.2.1 lexical-candidate KG.
This is **NOT a merged knowledge graph**, and does **not** add `SameAs` edges,
train TransE, change GAT, or establish any human-reviewed sense identity.

## Prerequisites (from Step 5 and Step 6)

Run from the repository root. These outputs are machine-local and are **not in Git**:

- `data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json`
- `data/knowledge/wordnet-ai-draft-v1/kg/{entities.json,class_entity_ids.json,metadata.json}`
- `data/knowledge/conceptnet-b2-v1/{entities.json,class_concept_candidates.json,metadata.json}`

Use the team setup and download instructions in
`docs/TEAM_REPRODUCIBILITY_AND_PROTOCOLS.md`. ConceptNet extraction for the
observed 5.7.0 snapshot must explicitly set `--max-edges 300000`; its default
50,000 is insufficient for the observed 56,703 retained assertions. The
ConceptNet snapshot SHA-256 on the original project machine was
`accd65fe94038584295574ddc26e1500c1919c8c4532bf771811cafd0948af7e`.
The prior AI-provisional CSV is a **separately supplied input artifact**;
a fresh clone cannot independently recreate the exact 124 mappings without it.

## Run

```bash
python -m pytest tests/knowledge/test_alignment_candidates.py -q
python -m scripts.build_alignment_candidates --dry-run
python -m scripts.build_alignment_candidates
cat data/knowledge/wordnet-conceptnet-b22-draft-v1/metadata.json
```

To use other **versioned, matching** inputs pass
`--wordnet-mapping`, `--wordnet-kg`, `--conceptnet-kg`, and a new `--output-dir`.
The script never downloads anything or overwrites differing outputs.

## Outputs

`data/knowledge/wordnet-conceptnet-b22-draft-v1/` contains:

- `class_alignment_candidates.json` — all 151 VG class IDs (background ID 0);
  WordNet linked sense or null, ConceptNet candidates, **no approval**.
- `candidate_pairs.tsv` — one row per WordNet–ConceptNet candidate pair when
  *both* exist, grounded only by shared VG150 label; **NOT an assertion edge**.
- `review_queue.csv` — 150 editable rows; class-level WordNet/ConceptNet
  decisions, reviewer, evidence and notes left blank by the generator.
- `bridge_review_queue.csv` — one row per possible WordNet–ConceptNet pair,
  with empty accept/reject/defer, human reviewer and evidence fields. This is
  a **review worksheet**, not approved edges.
- `metadata.json` — candidate counts, missing classes, input SHA-256 fingerprints,
  explicit zero `sameas_edge_count` and `ready_for_gat_kg_training=false`.

The script crosschecks labels/IDs in both sources, WordNet mapping SHA-256
against B.1 metadata, canonical mapping hash against B.2.1 metadata, and
referenced `wn:`/`cn:` entity IDs against actual entity lists. A mismatch
**fails closed**; never bypass checks simply to boost coverage.

## Human review gate (future work — NOT IMPLEMENTED HERE)

A human reviewer must separately inspect WordNet glosses, relevant approved
training-only object examples, ConceptNet assertion evidence and sense-specific
ConceptNet URI meaning. On 150 classes, record whether to accept, reject or
defer individual WordNet senses and ConceptNet candidates. For a bridge,
explicitly select precise source and target URIs, semantic relation, evidence,
reviewer and versioned source hashes. For polysemous labels use an unresolved
or per-instance policy; **do not invent a global `SameAs`**.

There is **no** compiler in this step that converts review CSV to graph edges:
review decisions must be validated by a separate future audited implementation.
All outputs remain drafts, not data for official TransE / GAT+KG comparisons.
Human-review requirements are tracked in
`docs/knowledge/HUMAN_REVIEW_REQUIRED.md`.

**Evaluation modes:** This step is knowledge preparation, not model evaluation.
Existing graph-constrained PredCls R@50/100 and mR@50/100 is implemented;
no-graph-constraint metrics are TODO. W&B `online`/`offline`/`disabled` modes
for *training logs* are distinct from offline ConceptNet KG extraction.
