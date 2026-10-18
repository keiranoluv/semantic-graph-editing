# Step 1 — VG150 MLP Baselines

**Date:** 2026-10-19
**Status:** 6 experiments running (2 models x 3 seeds)

## 1. Objective

Build reproducible Predicate Classification (PredCls) baselines
for VG150 before introducing GNNs and external knowledge.

Models:
- Geometry MLP: subject/object bounding-box geometry.
- Semantic MLP: geometry + trainable object-class embeddings.

Neither model uses visual features or GNN message passing.

## 2. Completed

- VG150 annotation loading and scene graph construction.
- Geometry feature extraction.
- Negative sampling (3 negatives per positive).
- CUDA mini-batch training and batched validation.
- W&B logging and checkpoint saving.
- Early stopping using validation mR@50.
- Experiment run isolation and resume support.

## 3. Training Configuration

| Parameter | Value |
|---|---|
| Dataset | VG150 |
| Train images | 57,723 |
| Validation images | 5,000 |
| Train batch size | 512 graphs |
| Validation batch size | 128 graphs |
| Maximum epochs | 100 |
| Early stopping patience | 10 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Negative sampling ratio | 3:1 |
| Seeds | 42, 123, 3407 |
| Checkpoint selection | Validation mR@50 |
| Box policy | clip |

Six independent experiments:
- Geometry MLP: seeds 42, 123, 3407.
- Semantic MLP: seeds 42, 123, 3407.

Report final results as mean +/- standard deviation.

## 4. Preliminary Results

Previous 15-epoch experiments, seed 42.
NOT a controlled ablation because batch sizes differ.

| Metric | Geometry MLP (BS=2048) | Semantic MLP (BS=512) |
|---|---:|---:|
| R@50 | 42.40% | 60.97% |
| R@100 | 47.06% | 64.54% |
| mR@50 | 3.08% | 13.25% |
| mR@100 | 3.81% | 16.57% |

## 5. Evaluation Protocols

### Mode A — Graph-Constrained PredCls [IMPLEMENTED]

For each directed subject-object pair:
1. Compute foreground predicate probabilities.
2. Keep only the highest-scoring foreground predicate.
3. Rank predicted triplets within each image.
4. Select top-K predictions.

Metrics:
- R@50, R@100
- mR@50, mR@100

Current checkpoint selection: graph-constrained mR@50.

### Mode B — No-Graph-Constraint PredCls [TODO]

For each directed subject-object pair:
1. Compute all foreground predicate probabilities.
2. Allow multiple predicate predictions for the same pair.
3. Rank all candidate triplets within each image.
4. Select top-K predictions.

Metrics:
- ngR@50, ngR@100
- ngmR@50, ngmR@100

Both evaluation modes must:
- Use identical ground-truth annotations.
- Use identical dataset splits.
- Use K = 50 and 100.
- Evaluate rankings separately per image.
- Preserve multiple ground-truth predicates per object pair.
- Include regression tests for metric correctness.

Do not assume equivalence to published VG150 benchmark
protocols without independently verifying their conventions.

## 6. Data and Training Notes

- Training targets:
  - -1: ambiguous predicate pair (ignored)
  -  0: background
  - >0: foreground predicate
- Training skips graphs with no positive training pairs.
- Previous runs skipped 517 graphs per epoch.
- Validation uses all candidate pairs.
- Current R@K is image-macro.
- Current mR@K is predicate-macro over classes with GT support.
- The current model outputs one probability distribution
  per ordered object pair.

## 7. Next Steps

- Complete all six experiments.
- Verify best checkpoints and early stopping.
- Report mean +/- std across seeds.
- Implement no-graph-constraint evaluation.
- Test equivalence of batched and single-image evaluation.
- Evaluate both modes using saved best checkpoints.
- Build a GNN baseline.
- Investigate external semantic knowledge integration.

## 8. Research Principle

Hold the dataset, split, features protocol, sampling policy,
evaluation protocol, and experiment budget consistent
when comparing Geometry MLP, Semantic MLP, and future GNNs.
