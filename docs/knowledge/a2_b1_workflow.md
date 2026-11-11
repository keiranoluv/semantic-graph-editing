# VG150 WordNet — A.2 Sense Review and B.1 Knowledge Graph

## Research protocol

The Milestone A mapping is an immutable input. Exactly 23/150 labels were
resolved in its first observed run; all 150 yielded WordNet candidates. This
is **candidate coverage**, not WordNet sense correctness. No VG relationship
GT, image-derived split statistics, or validation/test data are read here.

### A.2 — Rank, then explicitly review

```bash
python -m scripts.prepare_wordnet_review rank --dry-run
python -m scripts.prepare_wordnet_review rank
```

Outputs in `data/knowledge/wordnet-review-v1/`:

- `wordnet_rankings.json`: every sense ranked with its **lexical** score,
  WordNet gloss, usage count and source WordNet version.
- `wordnet_review.csv`: an editable 150-row template. Fill `selected_synset`,
  `decision=accept`, `reviewer` and (for ambiguous classes) `notes`.
  You may leave uncertain labels blank/deferred or mark them rejected.

Suggestions are **not automatically accepted**. The ranking uses an exact
surface-lemma bonus, log-scaled WordNet lemma usage counts and an extremely
small tie breaker on sense number. This is **not** a probabilistic confidence
or a semantics-based proof. Inspect definitions and, where necessary, examples
of VG object names or approved training samples; do not use validation/test
predicate labels. A `bike` may genuinely be underspecified, so do not force
an arbitrary sense merely to reach a 100% link rate.

Compile accepted rows:

```bash
python -m scripts.prepare_wordnet_review compile \
  --review-csv data/knowledge/wordnet-review-v1/wordnet_review.csv \
  --dry-run
python -m scripts.prepare_wordnet_review compile \
  --review-csv data/knowledge/wordnet-review-v1/wordnet_review.csv
```

This creates `configs/knowledge/wordnet_overrides.reviewed.json`. It checks
class IDs/labels against original mapping and forbids choices not listed as
candidates. Existing overrides cannot be replaced silently.

Then regenerate Milestone A in a **fresh** output directory:

```bash
python -m scripts.prepare_knowledge \
  --overrides configs/knowledge/wordnet_overrides.reviewed.json \
  --output-dir data/knowledge/wordnet-vg150-reviewed-v1
```

Caution: commit the curated override JSON once decisions have been checked;
keep generated review CSVs/artifacts local unless you intentionally publish a
versioned, documented artifact. The review identity/rationale is provenance,
not a substitute for measuring semantic correctness.

### B.1 — Controlled WordNet KG extraction

Draft smoke test (allowed before review is complete):

```bash
python -m scripts.build_wordnet_graph --max-hops 2 --dry-run
python -m scripts.build_wordnet_graph --max-hops 2
```

A final reviewed mapping must use a **new output dir** and the strict gate:

```bash
python -m scripts.build_wordnet_graph \
  --mapping data/knowledge/wordnet-vg150-reviewed-v1/wordnet_mapping.json \
  --output-dir data/knowledge/wordnet-kg-reviewed-v1 \
  --reviewed-overrides configs/knowledge/wordnet_overrides.reviewed.json \
  --require-fully-reviewed
```

Outputs: `entities.json` (prefixed WordNet synset IDs), `relations.json`,
`triples.tsv` (ID triples and WordNet provenance), `class_entity_ids.json`
(151 VG rows including background null), and `metadata.json` (versions,
input SHA256, source, limits and review readiness). All outputs are
**deterministic** for pinned inputs and WordNet corpus; differing existing
outputs are refused, not silently overwritten.

Five directional relation types are used: `IsA` (hypernym), `InstanceOf`
(instance hypernym), `HasPart` (part meronym), `HasMember` (member meronym),
and `HasSubstance` (substance meronym). Only WordNet nouns are admitted. The
bounded BFS expands at most 1–3 hops, aborts on size limits, and never uses
relation annotations from VG150. The background and unresolved classes get
`null` entity IDs, **not** dummy knowledge vectors.

The draft KG is intentionally **not** ready for final GAT+KG training until
all classes have been explicitly audited (with a matching override SHA256
and per-class reviewer provenance) and remaining ambiguities resolved;
`ready_for_gat_kg_training` in metadata enforces that distinction.

**ConceptNet is deferred to B.2**, because a reproducible ConceptNet subset
requires a pinned snapshot (including source, license and extraction hashes)
and a verified URI-linking policy. Do not claim ConceptNet assertions from a
WordNet-only graph. No embeddings are trained in B.1: TransE is Milestone C.
