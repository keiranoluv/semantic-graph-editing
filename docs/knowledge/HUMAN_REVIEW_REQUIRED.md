# ACTION REQUIRED — Human review gates for A.2, B.2, C and D

**This page is a blocker checklist, not a certification that review is done.**
Record decisions and provenance before any results are described as official.
The current counts below reflect the user's B.1 AI draft, **not** human
verification and not an independent measurement by this document.

## Current ground truth of the workflow (before human review)

| Artifact | Last observed result | Human-reviewed? | Use |
| --- | --- | --- | --- |
| A.1 WordNet label candidates | 150 labels, 23 unique-sense auto links, 127 ambiguous | **No** | Lexical baseline |
| A.2 WordNet AI draft | 124 proposed linked classes: 23 A.1 + 101 AI suggestions | **No** | Development only |
| A.2 deferred labels | 26 classes, IDs below | **No** | Manual audit priority |
| B.1 WordNet KG draft | 122 unique roots, 967 entities, 1,156 triples | **No** | Topology smoke test |
| B.2.1 ConceptNet labels | Counts determined only after pinned snapshot run | **No** | Candidate retrieval only |
| C TransE / D GAT+KG | Not started | N/A | Blocked for official runs |

**Deferred VG150 IDs** (from last observed AI draft):
`11, 13, 25, 44, 48, 53, 54, 62, 71, 74, 75, 76, 79, 85, 98,
101, 107, 108, 113, 115, 123, 134, 138, 145, 147, 148`.

All 124 provisional suggestions (including the 23 original auto links) still
require a human check. `linked_manual` in the AI-draft mapping is a compatibility
status **and does not imply that a human checked it**.

## Human review protocol (required before official GAT+KG)

1. **WordNet senses** — start from ORIGINAL A.1
   `data/knowledge/wordnet-vg150/wordnet_mapping.json`, not the AI-draft
   mapping. Inspect all 150 labels' candidate glosses/lexical suggestions,
   prioritizing 26 `defer` and homonyms such as `bike`, `glass`, `plate`,
   `plant`, `pant`, `kite`, `table`. If the intended meaning genuinely varies
   by object instance, flag as unresolved rather than forcing a sense.
2. **Review evidence** — whenever a label is unclear, use WordNet definitions
   and only approved **training** images/box examples if needed. No VG150
   validation/test predicates, ground-truth relations or split statistics.
   Note the evidence inspected and distinguish objective synset meaning from
   assumptions about a particular image.
3. **Record reviewer decisions** — fill the full 150-row
   `data/knowledge/wordnet-review-v1/wordnet_review.csv` (separate from the
   AI-generated CSV): `class_id`, `selected_synset`, `decision=accept/defer`,
   non-empty human `reviewer`, and a rationale in `notes`. Verify that selected
   synsets are in the original candidates; expand candidates with a separate,
   audited policy if a correct meaning is missing (e.g. `short`/`shorts`).
   Ideally use two independent reviewers for difficult labels and record
   disagreement/resolution; never copy AI as a fictitious person.
4. **Compile human decisions** — after humans actually review the CSV:

   ```bash
   python -m scripts.prepare_wordnet_review compile \
     --review-csv data/knowledge/wordnet-review-v1/wordnet_review.csv \
     --output configs/knowledge/wordnet_overrides.reviewed-v2.json \
     --dry-run
   python -m scripts.prepare_wordnet_review compile \
     --review-csv data/knowledge/wordnet-review-v1/wordnet_review.csv \
     --output configs/knowledge/wordnet_overrides.reviewed-v2.json
   ```

   Then regenerate the WordNet mapping into a **new** directory via
   `python -m scripts.prepare_knowledge --overrides ... --output-dir ...`.
   Do not enable `--require-fully-reviewed` while ambiguous/unreviewed classes
   remain, and do not claim 100% correctness from coverage alone.
5. **ConceptNet lexical candidate review** — inspect
   `data/knowledge/conceptnet-b2-v1/review_queue.csv` for all 150 labels.
   A ConceptNet `/c/en/<term>` is a *term*, not the corresponding WordNet
   synset sense. Inspect incident assertion evidence, URI sense specificity,
   multiple candidates, source dataset/license, and whether the relation
   describes the intended object. Record accept/reject/defer with reviewer,
   rationale, and evidence reference. A WordNet noun synset from corpus 3.0
   is NOT automatically equal to a ConceptNet URI sourced from WordNet 3.1.
6. **Bridge audit (future B.2.2)** — make a *separately versioned* explicit
   WordNet-to-ConceptNet alignment file. Include class ID, both full URI IDs,
   bridge semantics, approval status, evidence, reviewer identity, timestamp
   and source hashes. Do not add `SameAs` merely because labels have identical
   spelling. Keep ambiguous/unresolved bridges absent from training graph.
7. **Pre-training gate** — after expert review, report total accepted/rejected/
   deferred class mappings, inter-reviewer disagreements, artifact hashes,
   knowledge sources and their licenses. Only then decide which classes/edges
   enter the official C/D experiments. Keep AI-provisional and human-approved
   experiments in **different** directories and describe them separately.

## Prohibited shortcuts

- Treating `reviewer=ChatGPT-AI-provisional...` as verified human provenance.
- Calling `linked_manual` in A.2 AI draft proof of human review.
- Adding `SameAs` on lexical similarity alone.
- Choosing senses using VG val/test ground-truth predicates.
- Treating ConceptNet edge weights as calibrated correctness probabilities.
- Omitting attribution/license when publishing a ConceptNet-derived dataset.
- Claiming a full graph integration or official TransE/GAT+KG validation from
  the B.1 or B.2.1 draft topology alone.

**Owners:** assign at least one named human reviewer and one independent check
for difficult senses before marking this document's obligations complete.
