# VG150 PredCls Evaluation Protocol

## Current implementation

Evaluation code: `src/evaluation/scene_graph.py`

- Ground-truth object instances and boxes are provided.
- All directed subject-object pairs, except self-pairs, are candidates.
- Ground-truth relation pairs are NOT supplied to the classifier
  as the candidate selection.
- Each candidate pair predicts its highest-scoring foreground predicate.
- Background class 0 is excluded from predicted triplets.
- Predicted triplets are ranked by foreground softmax confidence.
- Top-K matching uses subject index, predicate ID, and object index.
- Different GT predicates for one pair are preserved.
- Duplicate identical GT triplets are deduplicated.

## Metrics

- R@K: mean of per-image recall over images containing GT relations.
- mR@K: aggregate hits/support independently for each observed
  predicate, then average across predicates with GT support.
- K values: 50 and 100.

## Limitations

This is our project-specific graph-constrained PredCls protocol.

Published benchmark compatibility is NOT yet fully established:

- Community VG150 preprocessing and split equivalence require verification.
- Some benchmark implementations use different duplicate or
  relationship filtering policies.
- Mean Recall averaging and treatment of absent classes must be
  aligned before comparing numerical results with published papers.
- No-graph-constraint Recall is not implemented here.
- Detector scores are not used because PredCls provides GT objects.
- Geometry-only features are an initial baseline, not the final
  semantic model input.

All models within this project must use the same preprocessing,
candidate generation, ranking, and evaluation protocol.

## Reference

https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/METRICS.md
