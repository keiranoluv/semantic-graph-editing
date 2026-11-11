# VG150 PredCls evaluation — explicit protocols

## Zero-shot recall: train-unseen class triples

`myenv/bin/python -m tools.level1_zero_shot --device cuda --batch-size 128`
evaluates nine fixed checkpoints: fusion baseline, pair KG real and pair KG
shuffled, each with seeds 42/123/3407. No training or checkpoint reselection.
`--prepare-only` writes the vocabulary/cohort/checkpoint definitions first.

The seen vocabulary contains **29,283 distinct (subject class, predicate, object
class) triples** from all 405,860 raw annotations of the 57,723 training images.
It includes duplicate rows and multi-predicate pairs ignored by the training
sampler. Validation/test annotations do not determine membership in this set.
Within 5,000 validation images, 722 images contain 1,130 unseen GT annotation
rows (851 distinct triples; 3.40% of the 33,203 GT rows). All component classes
and predicates have appeared individually in train in their corresponding roles.
This evaluates unseen combinations, not unseen object classes or absence from KG.

For each image, rank **all** model predictions as in reference GC/NG. Filter only
the GT to triples absent from the train vocabulary, keeping duplicate annotation
rows. Match predicates, object classes and both boxes at IoU >= 0.5 with the same
reference matcher. Let Z_i be these GT rows and H_i(K) the matched subset:

`zR@K = mean over images with |Z_i| > 0 of |H_i(K)| / |Z_i|`.

Images without unseen GT are excluded from this mean, rather than contributing
zero. The denominator is **722 images**, not 5,000 images, 851 unique triples or
1,130 pooled annotations. This is image-macro recall, not predicate-macro mR.
GC retains one predicate per ordered pair; NG allows multiple predicates.
Predictions for seen relations still consume top-K slots. For example, if an
image has two unseen GT rows, one matched at rank 45 and one at rank 70, its
zR@50 is 1/2 and zR@100 is 2/2, regardless of its seen GT relations.

The checker in `tools/check_zero_shot_reference.py` compares both modes against
`SGZeroShotRecall` / `SGNGZeroShotRecall` at the same pinned upstream commit used
below. Local tuples are ordered (s, p, o); the checker explicitly converts to
upstream's (s, o, p). It passed 100 synthetic cases and 32 real validation images.
Each full evaluation also reproduces the historical full-set GC/NG R/mR and
per-predicate metrics before accepting its report. Per-image unseen supports and
hit counts are stored to allow independent recomputation.

[Results and limitations](../../reports/level1/zero-shot-reference-v1/analysis.md) ·
[Frozen train vocabulary](../../reports/level1/zero-shot-reference-v1/train-seen-triplets.json) ·
[Validation cohort](../../reports/level1/zero-shot-reference-v1/validation-cohort.json) ·
[Differential check](../../reports/level1/zero-shot-reference-v1/differential-check.json).

## Long-tail analysis of existing reference reports

`myenv/bin/python -m tools.analyze_level1_long_tail` computes group-macro mR
for head/body/tail without running inference or training. Optional `--groups-only`
freezes the grouping before reading model scores. All 50 predicates are sorted by
raw **train annotation** count descending, breaking ties by ascending predicate
ID; head/body/tail contain 10/20/20 classes. Counts include duplicates and
multi-predicate pairs, even if those pairs were ignored by the training sampler.
This is a declared project convention, not a universal benchmark partition.

Group mR is the arithmetic mean of the already stored image-macro class recalls
in that group. The original top-K ranking across the entire image is retained.
The fixed group denominator includes classes absent from validation, with zero
recall. In particular `flying in` stays in the 20-class tail denominator.
Group R aggregated over images cannot be recovered from these aggregate reports;
the analysis reports **group mR**, not that different metric.

Every report must satisfy `mR = 0.2*head_mR + 0.4*body_mR + 0.4*tail_mR` for
K=50/100. The tool checks this to absolute tolerance 1e-12, source/report/checkpoint
hashes, fixed inputs and support, then calculates three-seed mean/sample std and
paired differences. GC and NG remain separate. Outputs are immutable; choose a
new `--output-dir` for a different analysis version.

[Full tables](../../reports/level1/long-tail-reference-v1/analysis.md) ·
[Frozen class groups](../../reports/level1/long-tail-reference-v1/predicate-groups.json) ·
[Per-class recall and support](../../reports/level1/long-tail-reference-v1/per-predicate.csv).

## Supplementary evaluation: `reference-ng-v1`

`--protocol reference-ng-v1` uses the same GT, box/class matching, duplicate-row
policy, and image/per-predicate averaging as `reference-v1`, but allows multiple
predicates per ordered object pair. Every foreground predicate is a candidate;
top-K is taken **globally per image**, not per pair. Background participates in
softmax normalization but is never a ranked relationship. Each triplet consumes
one slot. Two correct predicates on the same pair can both count, if both fit
top-K. Extra predicates can displace another correct pair, so recall is not
mathematically guaranteed to increase.

To match the pinned upstream postprocessor and `SGNoGraphConstraintRecall`,
first sort pairs by their maximum foreground probability with CPU `torch.sort`,
then flatten all foreground scores and use NumPy `argsort(-scores.ravel())`.
PredCls object scores are one. Equal-score ordering is implementation dependent;
NumPy/PyTorch versions are recorded. The differential checker includes ties.

Report identifier: `PredCls_reference_ng_v1_iou50_image_macro_per_predicate_50`.
Metric keys are explicitly **ng-R@50/100 and ng-mR@50/100**, including per-class
ng-R. Never mix these with graph-constrained R/mR in a model comparison.
Historical trainers and checkpoint selection are unchanged.

```bash
myenv/bin/python -m tools.level1_reference_matrix --protocol reference-ng-v1 --matrix main --device cuda --batch-size 128
myenv/bin/python -m tools.level1_reference_matrix --protocol reference-ng-v1 --matrix pair --device cuda --batch-size 128
myenv/bin/python -m tools.level1_reference_matrix --protocol reference-ng-v1 --matrix fusion --device cuda --batch-size 128
myenv/bin/python -m tools.compare_reference_protocols
```

Outputs go to `reports/level1/reference-ng-v1{,-pair,-fusion}`; all 54 eligible
frozen checkpoints use the same 5,000 validation images. The comparison checks
report/checkpoint hashes, identical inputs/configs/support, and recomputes all
three-seed means/sample standard deviations. Original GC reports retain their
historical evaluator hashes; NG reports must match current evaluator code.
To rescore old GC reports after source changes, use a **new `--output-dir`**;
the runner deliberately refuses to silently reuse reports from changed code.

[Side-by-side tables](../../reports/level1/reference-comparison/comparison.md) ·
[Differential check for both modes](../../reports/level1/reference-ng-v1-differential-check.json).

## New evaluation: `reference-v1`

Use `scripts.evaluate_level1 --protocol reference-v1` for new scoring. The
implementation is `src/evaluation/reference_predcls.py`. It matches the metric
semantics of KaihuaTang/Scene-Graph-Benchmark.pytorch at commit
`ceb71fa88461c2a97a6258a80f47669d89207296` on the supplied predictions/GT:

- GT object boxes and labels are given. All ordered non-self object pairs are
  candidates, independently of GT relation labels.
- Full softmax includes background; each pair selects its best foreground
  predicate. Rank by that confidence with unit GT object scores.
- Ranking uses CPU `torch.sort(descending=True)`, like the reference sorting
  operation. Exact ties may depend on PyTorch/device; no claim of tie invariance
  across arbitrary platforms is made.
- Match object classes and both subject/object boxes at IoU >= 0.5, using inclusive
  XYXY pixel areas (`+1`), rather than requiring identical object indices.
- Keep duplicate GT relation rows and different predicates on the same pair.
  A prediction can match several GT rows; each GT row is counted once.
- R@50/100 averages per-image recall over images with annotated relations.
- For mR, average each predicate's recall over images containing that predicate,
  then average over **all 50 foreground predicates**. An absent predicate
  contributes zero. Empty evaluated datasets return null metrics.
- Per-predicate reports contain annotation support, images_with_gt and the same
  image-macro class recall used in mR (not pooled hits/support).

Report protocol identifier:
`PredCls_reference_v1_iou50_image_macro_per_predicate_50`.
Reports include reference commit, evaluator source hashes and historical checkpoint
selection protocol. Output creation refuses to overwrite an existing report.

This is an evaluator change, not a claim of identical training/preprocessing to
every paper. Local GT box conversion/clipping and the existing training treatment
of multi-predicate pairs remain unchanged. Differential checks use the same
supplied boxes, labels and predictions on both sides.

## Fixed-checkpoint re-evaluation

From repository root, evaluate a frozen checkpoint on full validation:

```bash
myenv/bin/python -m scripts.evaluate_level1 \
  --checkpoint runs/level1-clip-v1/clip-gat-seed42-v1/run_0001/best.pt \
  --protocol reference-v1 --device cuda --batch-size 128 \
  --output reports/level1/reference-v1/gat-seed42-val5000.json
```

For the 27 main checkpoints in the original frozen plan:

```bash
myenv/bin/python -m tools.level1_reference_matrix --dry-run
myenv/bin/python -m tools.level1_reference_matrix --device cuda --batch-size 128
```

For the later matrices, fixed from their existing legacy report hashes:

```bash
myenv/bin/python -m tools.level1_reference_matrix --matrix pair --device cuda --batch-size 128
myenv/bin/python -m tools.level1_reference_matrix --matrix fusion --device cuda --batch-size 128
```

These commands rescore 12 pair/loss and 15 eligible fusion runs respectively,
writing separate `reference-v1-pair` / `reference-v1-fusion` directories. Fusion
explicitly excludes the six relation-prior runs with audited L1/L2 inconsistency.

The launcher runs sequentially, verifies checkpoint hashes, and resumes only from
compatible existing reference-v1 validation reports. It writes a separate 3-seed
mean/sample-std summary. It never reselects checkpoints and does not evaluate test.
The original test-plan file is used only to identify the frozen checkpoints.

Historical trainers and validation checkpoint selection still use the original
metric. These rescored checkpoints were **not selected by reference-v1 mR**.
Introducing reference-v1 into future training selection requires a separately
versioned experiment; do not silently resume old runs with new selection semantics.
Test evaluation still requires explicit `--allow-test`; previously viewed test is
not an independent holdout for later model designs.

## Historical evaluation: `legacy`

`scripts.evaluate_level1` defaults to `legacy` to preserve existing launchers and
historical reproduction commands. Always pass `--protocol reference-v1` for the
new metric. The legacy functions in `src/evaluation/scene_graph.py` remain intact:

- Strict subject/predicate/object **node-index** matching.
- Deduplicate identical GT triplets; keep different predicates for the same pair.
- Image-macro R@K.
- Pool hits/support per predicate over the dataset, then average only over
  predicates with support (49 on current val, 50 on test).
- Stable descending score order.

Legacy identifier: `PredCls_graph_constrained_top1_per_pair`.
Do not compare legacy mR numerically with reference-v1 or paper tables.
`scripts.summarize_level1` rejects mixing different protocol identifiers.
Use the separate `reference-ng-v1` protocol above for no-graph-constraint recall.

## Verification

```bash
myenv/bin/python -m pytest -q
myenv/bin/python -m tools.check_reference_predcls \
  --checkpoint runs/level1-clip-v1/clip-gat-seed42-v1/run_0001/best.pt
```

The differential checker downloads pinned source into a temporary directory and
executes the actual reference metric/IoU definitions without installing its
detector stack. It compares image hits, R@50/100, mR@50/100 and all per-class
recalls in both graph-constrained and no-graph-constraint modes on 100 synthetic cases plus, optionally, the first 16 real validation
images on CPU. Cases cover duplicates, overlapping boxes, multiple predicates,
background and tied scores. Offline tests additionally cover absent classes,
empty inputs, inclusive IoU boundary and independent batching for MLP/GAT.

[Recorded differential check](../../reports/level1/reference-v1-differential-check.json).
[Prior audit](../experiments/level1_audit_2026-11-11.md).

Reference sources:
[evaluator](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/ceb71fa88461c2a97a6258a80f47669d89207296/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py),
[postprocessor](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/ceb71fa88461c2a97a6258a80f47669d89207296/maskrcnn_benchmark/modeling/roi_heads/relation_head/inference.py),
[box IoU](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/ceb71fa88461c2a97a6258a80f47669d89207296/maskrcnn_benchmark/structures/boxlist_ops.py).
