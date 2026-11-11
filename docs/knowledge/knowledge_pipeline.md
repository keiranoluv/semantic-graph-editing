# VG150 external knowledge preparation — Milestone A

## Scope

This stage links **object vocabulary IDs** to WordNet noun synsets. It does not
read Visual Genome predicates, images, split manifests, or relation statistics.
It does not modify existing MLP/GAT models or training scripts. ConceptNet
retrieval, graph construction and TransE are later milestones.

## Input and ID contract

Input: `data/raw/vg150/VG-SGG-dicts-with-attri.json`, key `label_to_idx`.
Every object label must have exactly one ID from 1 through 150. Index 0 is
reserved for background and is always unmapped. `idx_to_label`, if supplied,
is checked for consistency. Never sort rows lexicographically.

## Linking policy

1. Case-fold and replace whitespace/hyphens with underscores.
2. Lookup WordNet **noun** synsets for the normalized lemma. Try WordNet noun
   morphology as a fallback only when exact lookup is empty.
3. With one candidate sense, produce `linked_auto`. With multiple senses,
   produce `ambiguous` (no chosen synset). With no senses, `unmatched`.
4. Review `wordnet_review.csv` manually and add explicit `label: synset_id`
   decisions to `configs/knowledge/wordnet_overrides.json`. Decisions create
   `linked_manual` rows. Curated overrides must point to a valid noun synset.
5. Auto links are **not ground truth**. Randomly audit a sample of auto links
   too, documenting discovered polysemy and inappropriate senses.

Do not use VG150 validation/test **relation annotations** or split-derived priors
when reviewing senses. WordNet is independent external knowledge, but careless
manual mapping could still inject task-specific knowledge; document reviewers
and decisions. ConceptNet concept URIs must not be treated as verified graph
entities until they have been checked against a pinned ConceptNet snapshot.

## Reproduction

Install: `pip install nltk` (pin a compatible NLTK release for the final report).
Then explicitly download corpus: `python -m nltk.downloader wordnet`.
Record WordNet version and NLTK version in the generated mapping.

```bash
python -m scripts.prepare_knowledge --dry-run
python -m scripts.prepare_knowledge
python -m pytest tests/knowledge -q
```

The output folder defaults to `data/knowledge/wordnet-vg150/`:

- `wordnet_mapping.json`: one row per class ID, including unmapped background
- `wordnet_coverage.json`: resolved and candidate coverage separately
- `wordnet_review.csv`: ambiguous or unmatched labels requiring review

Each output is deterministic for pinned vocabulary, overrides and WordNet corpus.
Existing divergent files cannot be silently overwritten; choose a new output
folder for a new version. Exclude machine-local generated artifacts from Git
unless there is an explicit review/versioning decision. Source vocab and
review-override SHA256 hashes are included in the mapping.

## Interpretation

`resolved_coverage` is `(linked_auto + linked_manual) / 150`.
`candidate_coverage` is the fraction of labels with at least one lexical
candidate. They are **not** semantic alignment accuracy. Report separately the
number of auto links audited and their verified correctness. Unmatched labels
remain explicitly null, not quietly mapped to an arbitrary concept.

Later integration will create an embedding matrix of shape `[151, D_kg]` aligned
to the same class IDs. ID 0 must use the documented background policy. Do not
start GAT+KG training until linkage and the KG extraction/training protocol
have been reviewed.
