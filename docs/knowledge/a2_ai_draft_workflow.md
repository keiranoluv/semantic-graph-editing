# A.2 — AI-provisional VG150 mapping and WordNet KG smoke test

This optional step uses suggestions from `wordnet_review_ai_provisional.csv`
without passing them through the human-only `prepare_wordnet_review compile` command.
It **does not validate any sense with image examples**. It should never be
called a manually curated or human-reviewed dataset.

The initial VG150 A.1 output linked 23 classes; the provided A.2 CSV accepts
124 lexical suggestions (including the existing 23) and defers 26. The script
checks these numbers from actual inputs rather than hardcoding them.

Run from the repository root after installing NLTK WordNet 3.0:

```bash
python -m scripts.build_ai_wordnet_draft --dry-run
python -m scripts.build_ai_wordnet_draft
```

Inputs (never modified):
- `data/knowledge/wordnet-vg150/wordnet_mapping.json`
- `data/knowledge/wordnet-review-v1/wordnet_rankings.json`
- `data/knowledge/wordnet-review-v1/wordnet_review_ai_provisional.csv`

Generated `data/knowledge/wordnet-ai-draft-v1/`:
- `mapping/wordnet_mapping.json` — B.1-compatible provisional class linking.
- `mapping/ai_draft_summary.json` — counts, source hashes, and zero human-verification claim.
- `mapping/deferred_review.csv` — the 26 classes requiring image-context or candidate review.
- `kg/{entities.json,relations.json,triples.tsv,class_entity_ids.json,metadata.json}` — WordNet KG.

The current B.1 schema accepts only `linked_manual` or `linked_auto` for linked
classes. Therefore selected AI suggestions use `linked_manual` *only for
compatibility*, with `curation_tier=AI_PROVISIONAL_NOT_HUMAN_VERIFIED` and
`verified_by_human=false`. **Never infer human review from this enum.**

`kg/metadata.json` always explicitly marks `ready_for_gat_kg_training=false`;
all 124 linked classes are unreviewed by a person. Do not supply an AI reviewer
as `--reviewed-overrides`, and do not use these outputs for final GAT+KG
experiments. Future human audit should start from the ORIGINAL A.1 mapping and
compile only genuinely reviewed choices with `prepare_wordnet_review compile`.

Class linking depends only on vocabulary labels and WordNet concepts; no
VG150 train/val/test predicate annotations are read. Generated data are local
artifacts and need not be committed. Never run `git add .` on this directory.
