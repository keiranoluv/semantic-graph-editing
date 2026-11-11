# Level 1 audit — 2026-11-11

## Verdict

The core MLP/GAT/GAT+KG pipeline implements the Level 1 architecture requested
in Project01_Graph_Editing.pdf. Existing results remain usable as **internal,
project-protocol comparisons**. They are not numerically interchangeable with
published VG150 benchmark results. Resolve the evaluator differences before
claiming benchmark parity. KG benefit remains unproven within the tested designs.

This audit does not change the evaluator, training, checkpoints, or existing
experiment reports. It does not run full-split inference or retrain any model.

## Confirmed findings, ordered by impact

### High: mean Recall is not the reference implementation's mean Recall

`src/evaluation/scene_graph.py:95` pools hits/support across images for each
predicate, then averages over supported predicates. The referenced
`SGMeanRecall` averages per-image predicate recalls, then averages over all
50 foreground categories (absent categories contribute zero).

An executable probe using the actual reference class and identical matches:
one class has 9/9 hits in image A and 0/1 in image B. Project mR is 0.90;
reference mR is 0.50. Adding an absent class leaves project mR at 0.90 and
makes reference mR 0.25. These are synthetic examples, **not estimates of the
size or direction of the difference on VG150**.

Impact: all reported mR@50/100 and checkpoint selection use the project's
metric. Validation has 49 supported predicates; test has 50. Existing aggregate
reports cannot recover per-image averages. Re-evaluate fixed checkpoints under
a versioned evaluator. Do not select a different checkpoint using test results.
A change to validation selection semantics may require a new training protocol;
rescoring saved best checkpoints does not retroactively change their selection.

### High for external comparisons: instance matching and duplicate handling differ

The project matches exact subject/object node IDs and deduplicates identical GT
triplets. The reference matcher uses object classes and subject/object box IoUs;
its evaluation data retains relation tuples. Same-class overlapping boxes can
therefore match differently even in PredCls. Changing mR aggregation alone is
insufficient for full compatibility. This audit inspected the matcher source;
it did not quantify the effect over the full dataset.

### Medium: relation-prior scores switch TransE distance

`TransE` defaults to L1, and `scripts/train_transe.py` constructs it without
changing that norm. `src/knowledge/pair_relation.py` exports negative L2 scores.
Recomputing the saved tensor gives maximum absolute standardized-score difference
0.0 for L2 and 2.1347661 for L1.

Impact: the six `relation_real`/`relation_shuffled` fusion runs use a derived L2
feature, not the trained TransE scoring norm. This does not invalidate the
node-concatenation, pair, or visual-gate results. Either explicitly describe L2
as a design variation, or export a new L1 artifact and rerun the affected six
conditions. Do not overwrite existing artifacts. No gain from fixing this is
established by this audit.

### Medium: multi-predicate pairs lose supervision

`src/graph/builder.py` assigns -1 when a pair has multiple distinct predicates;
`src/training/sampling.py` excludes those pairs. Recounting the full train split:
16,894 / 297,318 positive pairs (5.68%) are excluded, containing 48,516 / 405,860
relation annotations (11.95%). The reference training loader instead samples
one predicate for each pair under duplicate filtering.

This is a documented modeling choice, not evidence of leakage. It affects all
compared models and cannot alone explain why KG underperforms. Disclose it;
a future sampled-label or multi-label training design requires new experiments.

## Checks that passed

- Full existing CPU suite: **204 passed in 3.82s**.
- **87 individual reports** checked: 27 initial validation, 27 initial test,
  12 pair/loss validation and 21 fusion validation. No mismatches in checkpoint
  SHA-256, epoch, saved config, data hashes, CLIP manifest identity, or full-split
  image counts. All 60 validation reports agree with the four saved checkpoint
  validation metrics. Test metrics were not independently recomputed.
- All 27 original test-plan checkpoint hashes still match current files.
- Actual annotation, vocabulary, aligned metadata and split hashes verified.
- Split rows reproduce the reference selection rule on the local HDF5 before
  optional non-overlap filtering: 57,723 train / 5,000 val / 26,446 test.
  Selected image IDs are unique across splits. This is not proof that every
  paper uses identical preprocessing or that original upstream metadata order
  has been independently reconstructed in this audit.
- Frozen CLIP cache content hash and source manifests verified.
- All **21 unique GAT+KG checkpoints** checked have KG buffers exactly matching
  their C.2 bundles: WordNet covers 150 classes; ConceptNet covers 138.
  Export validates vocabulary alignment and provenance. This does not replace
  human semantic review of each mapping or each external assertion.
- Recomputed class prior over all 57,723 train images equals the saved prior.
- Common report configs agree on image budgets, negative ratio, batch size,
  learning rate, weight decay, CLIP input width, box policy and selection metric.

## Code-level observations and limits

Graph topology and candidate pairs are constructed independently of predicate
labels. Predictors receive node features, object labels and candidate topology;
GT predicates are retained separately for supervision/evaluation. CLIP runs
frozen inference; KG builders consume external lexical/conceptual sources.
No direct VG test-predicate-to-feature path was found in the inspected code.
This is not a claim that every possible leakage source has been exhaustively
excluded (including upstream pretraining data).

Node KG vectors are concatenated before the GAT projection, matching the Level 1
proposal. KG-off zeros vectors and validity bits. Edges-off removes inter-object
message edges but retains self loops and supervised candidate pairs. The original
KG-off control uses the WordNet architecture/bundle; it is not a fully matched
control for every ConceptNet experiment. MLP and GAT also differ in depth and
capacity, so their gain is an architectural comparison, not isolated proof that
message passing alone causes it.

Test-plan hashes establish artifact identity, not independent evidence of the
chronology of model-selection decisions. Fusion was designed after earlier test
results were seen; its validation results remain development results.

## Required work before submission

1. Add a separately versioned reference-compatible evaluator, with differential
   tests covering averaging, absent classes, duplicate GT and overlapping boxes.
   Preserve historical project-protocol reports and label both protocols clearly.
2. Re-evaluate already fixed checkpoints, beginning with validation; publish old
   and reference-compatible metrics separately. Do not promise unchanged rankings.
3. Resolve or exclude the six relation-prior conditions from the main conclusions.
   No need to discard or retrain all models immediately.
4. Complete human review of five hit/five miss cases, final report/slides,
   reproducibility on a teammate machine and timed demo/video. Clarify with the
   instructor whether the general editing-demo requirement applies to Level 1.
5. Refresh stale README/progress sections after the final protocol is chosen.

## Reproduction and evidence

From repository root (CPU; downloads two pinned reference source files):

```bash
myenv/bin/python -m tools.audit_level1_artifacts
myenv/bin/python -m pytest -q
```

[Machine-readable evidence](../../reports/level1/audit-2026-11-11.json).
The audit script writes only that audit report and temporary reference copies.

Reference commit: `ceb71fa88461c2a97a6258a80f47669d89207296`.
[Evaluator](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/ceb71fa88461c2a97a6258a80f47669d89207296/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py),
[data loader](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/ceb71fa88461c2a97a6258a80f47669d89207296/maskrcnn_benchmark/data/datasets/visual_genome.py).
Source SHA-256 values are recorded in the JSON evidence.
