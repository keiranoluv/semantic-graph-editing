# Step 6 repository audit (2026-10-19)

Audited GitHub branch: https://github.com/keiranoluv/semantic-graph-editing/tree/feat/step6-conceptnet-b2

HEAD: `db760747babea21f4e40cfbba78f2e57fd293ecf`
Base Step 5: `bd13b0286f924f5f97c105f8c45e0509796c6886`
Comparison: exactly **one** commit ahead, **zero** behind.

## Commits and scope

- `e6fa40c` - WordNet A.1 entity linking.
- `82fb642` - A.2/B.1 docs/tests/override file (commit title repeats A.1; misleading title, but content present).
- `7957319` - A.2/B.1 implementation (4 Python source files).
- `bd13b02` - A.2 provisional WordNet draft.
- `db76074` - B.2.1 extraction, exactly 5 additions: `scripts/build_conceptnet_graph.py`, `src/knowledge/conceptnet_builder.py`, `tests/knowledge/test_conceptnet_builder.py`, `docs/knowledge/b2_conceptnet_workflow.md`, `docs/knowledge/HUMAN_REVIEW_REQUIRED.md`.

No Step 6 modifications to the Step 5 implementation, GAT, MLP or baseline evaluators in the branch compare. Filenames and layout match `scripts/`, `src/knowledge/`, `tests/knowledge/`, `docs/knowledge/`.

## Code review and caveats

| Item | Finding | Required follow-up |
| --- | --- | --- |
| CLI and five-field parsing | Implemented. Reads compressed or uncompressed snapshot; checks exact SHA-256 before extraction. | Keep snapshot pin and input schema tests. |
| No image/relation leakage | B.2 extractor validates class-ID order and reads only input mapping labels; no VG predicate annotation access in extraction. | Avoid GT-derived mapping decisions in downstream review. |
| Network usage | B.2 extractor is offline only. | An online API mode would need separate implementation and versioned caching. |
| Concepts vs synsets | `cn:/c/en/...` is separated from `wn:...`; no `SameAs` bridges. | Implement audited B.2.2 bridge later. |
| Source/edge licensing | Original assertion URI, dataset, license and weight retained. | Check attribution and CC BY-SA conditions for publication. |
| Retained edge cap | CLI default is 50,000, but the observed reference keeps **56,703** assertions. Default run fails. | Use explicit `--max-edges 300000` in team runbook; optionally change default in a dedicated tested PR. |
| ConceptNet lexical coverage | 150/150 labels have lexical candidates. | This is **not** sense accuracy or human validation. |
| Status flag | `ready_for_gat_kg_training=false` and `human_verified_class_count=0`. | Keep false until human gates pass. |
| VG150 raw source pinning | HDF5 SHA-256 is fixed by script, but vocabulary JSON and metadata URLs are mutable. | Record SHA-256 of both and pin immutable download revisions for group reproducibility. |
| Dependency management | Root `requirements.txt` lists only `wandb>=0.18`; no complete lock for Torch/PyG/NLTK/h5py. | Add pinned environment file when machine environments are agreed. |
| Repo discoverability | Root README before this docs patch had only a title. | Link onboarding runbook and current protocol/status. |
| Generated data tracking | `.gitignore` before docs patch omits `/data/knowledge/` and `/data/external/`. | Add these ignores; never commit snapshot or generated KG. |
| AI draft reproducibility | `wordnet_review_ai_provisional.csv` is NOT produced from a fresh clone by the audited scripts and is not tracked in the repo. | Distribute reviewed-source artifact + checksum, or implement deterministic generation and document the input proposals. |
| Two evaluation protocols | Graph-constrained PredCls exists. No-graph-constraint is **TODO**. | Add test-backed second evaluator and explicit mode in later step. |
| W&B modes | GAT + current MLP runner support online, offline, disabled. | Keep separate from ConceptNet offline extraction. |
| GAT+KG config | `configs/gat_kg.yaml` is empty. | Do not claim TransE/GAT+KG completed. |
| Test gate | GAT evaluator requires `--allow-test`; MLP evaluator lacks equivalent guard. | Add test lock to MLP evaluator separately; until then enforce policy manually. |
| CI | No `.github/workflows/` visible in branch root. | Add CI for tests/pre-commit if desired. |

## Observed B.2.1 real-data reference (user-generated, not independently reproduced here)

- Snapshot: ConceptNet 5.7.0
- Snapshot SHA-256: `accd65fe94038584295574ddc26e1500c1919c8c4532bf771811cafd0948af7e`
- `max_edges=300000`, `min_weight=1.0`
- 34,074,917 raw input assertions
- 46,414 retained entities
- 56,703 retained evidence rows / 56,703 unique triples
- 10 relations; 150 of 150 classes with exact lexical candidates
- 0 human-verified and 0 asserted WordNet-ConceptNet identity links.

## Existing evaluation details

Implemented `src/evaluation/scene_graph.py`: use full background-inclusive softmax, select one maximum-scoring foreground predicate for each ordered object pair, rank predicted triplets within each image. `R@50/100` image macro; `mR@50/100` mean of per-predicate recall across GT-supported predicates. Both share best checkpoint selection by validation mR@50. `ngR@K` and `ngmR@K` have **no implementation yet**. Published benchmark equivalence has not been established.

## Bottom line

**Step 6 source layout and commit are correct for a draft KG extractor.** The reproducibility/documentation and security-of-workflow problems are missing/pinned dependency setup, AI-draft CSV distribution, stale max-edges command, uncovered data directories, and unclear mode/status distinctions. **No code audit alone can certify semantic alignment or runtime on every teammate machine.**
