# Full GQA checkpoint quality diagnosis — 2026-11-12

GPU inference on all 5,000 validation images reproduces the stored best GC
mR@50 exactly for all six available checkpoints. The completed models have
poor recall across rare predicates, despite high overall R@50. At the one
completed paired seed (42), GAT has not improved on MLP.

All scores below are percentages. These are validation-selected results on
the project's custom 310-predicate GQA protocol, not held-out test results.

| Model | Seed | Best epoch | Complete at snapshot | R@50 | mR@50 | R@100 | mR@100 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| MLP | 42 | 141 | Yes | 77.26 | 3.63 | 87.65 | 6.17 |
| MLP | 123 | 79 | Yes | 76.09 | 2.96 | 86.85 | 4.82 |
| MLP | 3407 | 108 | Yes | 76.63 | 3.41 | 87.23 | 6.05 |
| GAT | 42 | 59 | Yes | 77.20 | 3.01 | 87.71 | 4.68 |
| GAT | 123 | 90 | No | 77.39 | 3.15 | 87.85 | 5.48 |
| GAT | 3407 | 3 | No | 56.90 | 0.65 | 71.54 | 0.92 |

Completed MLP runs: R@50 **76.66 ± 0.59**, mR@50 **3.33 ± 0.34**,
R@100 **87.24 ± 0.40**, mR@100 **5.68 ± 0.75** (mean ± sample SD,
three seeds). GAT minus MLP at seed 42 is **−0.63 percentage points**
mR@50; one paired seed cannot establish a general architecture advantage.
Do not average the two unfinished GAT runs with the completed one as a
converged three-seed result. No full-run PE checkpoint existed at snapshot time.

## Where the low mean recall comes from

The validation set has 251,006 GT relation rows. Left/right account for
223,894 rows (**89.20%**). The ten predicates most frequent in training
account for **96.78%** of validation rows. These proportions describe annotation
counts; the reported overall R@K averages image recalls, so it is not a direct
weighted average using those proportions.

Groups below are fixed using training frequencies, not selected by validation
performance. mR includes all classes in each group, with absent classes set to zero.

| Predicate group | Classes present / total | MLP 42 mR@50 | GAT 42 mR@50 |
| --- | ---: | ---: | ---: |
| Ten most frequent in train | 10 / 10 | 39.94 | 39.77 |
| Next forty in train | 40 / 40 | 7.06 | 4.81 |
| Remaining predicates | 217 / 260 | 1.71 | 1.31 |

MLP 42 recalls at least one GT relation at K=50 for only **61/267** supported
classes; **206 supported classes** have zero recall. GAT 42 recalls **51/267**.
In the least frequent group, the corresponding coverage is **20/217** and
**17/217**. This is a substantive coverage problem, not merely a small change
in how a global metric is displayed.

There are also **43/310 classes absent from validation**. Excluding them from
the mean raises MLP 42 mR@50 from **3.63 to 4.22**, and GAT 42 from
**3.01 to 3.49**. That denominator convention explains only part of the low
score. The official score for this experiment remains the fixed 310-class mean.

## Diagnostic intervention: suppress left/right

The `non_spatial` JSON key means **all predicates except “to the left of” and
“to the right of”**. It still includes spatial predicates such as on and above.
It must not be interpreted as a semantic non-spatial subset.

First, evaluate the 308 remaining classes within the unchanged original top-K
predictions. Second, for a separate diagnostic, set the two left/right logits
to −1e9, renormalize softmax, select the new foreground label for each pair,
rerank all pairs and evaluate only the remaining GT relations.

| Measure | MLP 42 | GAT 42 |
| --- | ---: | ---: |
| Left/right share of original top-50 predictions | 73.63 | 72.94 |
| Original R@50 on GT outside left/right | 31.14 | 30.89 |
| Original mR@50 over 308 classes | 3.10 | 2.47 |
| Suppressed-left/right R@50 on retained GT | 59.42 | 59.43 |
| Suppressed-left/right mR@50 over 308 classes | 9.40 | 7.25 |
| Original foreground accuracy on GT pairs having a label outside left/right | 59.26 | 59.12 |

The intervention demonstrates sensitivity to left/right competition, but it
changes both pair classification and ranking, including softmax normalization.
It does not isolate a pure ranking-budget effect, does not measure performance
after retraining, and does not produce a replacement benchmark score. Its
remaining low mR also shows that removing those two labels alone does not solve
the coverage problem.

GT-pair accuracy above accepts any annotated non-left/right label on each exact
GT pair. Its denominator is 25,139 unique pairs, not GT rows; it is micro averaged
and omits candidate retrieval and IoU matching. It is not directly comparable to
image-averaged R@50 or class-averaged mR@50. Frequent retained labels can dominate
this accuracy even when most rare classes have no successful retrievals.

## Published GQA context

The DRM paper reports the following PredCls results on **GQA200**:

| Published method | R@50 | mR@50 |
| --- | ---: | ---: |
| MOTIFS | 65.3 | 16.4 |
| VETO (2023) | 64.5 | 21.2 |
| DRM (2024) | 43.2 | 41.9 |

GQA200 uses 200 object and 100 predicate categories. Our vocabulary has 1,699
foreground objects and 310 predicates, with a custom split and candidate/data
pipeline. These published numbers provide context, not a matched comparison;
neither our higher R@50 nor lower mR@50 establishes a performance ratio against
those models. A direct benchmark requires their complete data and evaluation
protocol, not just filtering our output to 100 labels.
Source: [DRM, experimental settings and Table 3](https://arxiv.org/html/2406.02038v1#S4).

## Interpretation and next experiment

The evidence supports investigating class imbalance and prediction/ranking
competition. It does not establish whether optimization, feature quality,
label ambiguity or another modeling limitation is the dominant remaining cause.
Exact replay agreement rules out a discrepancy between these saved scores and
this reference evaluation; it does not independently prove that every shared
pipeline assumption is correct.

Finish the frozen sequential matrix, then compare PE-CE, PE-PL and PE-PLPR
against the completed MLP/GAT controls using this same per-class breakdown and
three seeds. Any later class-balanced loss or sampling intervention should be a
separate experiment with its own output directory and provenance. If comparison
to published GQA200 scores is required, prepare that protocol as a separate
matched experiment. This diagnosis does not modify the current training recipe.

## Evidence and execution

- [Manifest](manifest.json): checkpoint SHA256s, selected epochs, completion
  flags, experiment/source/audit hashes and validation image-order hash.
- [Summary CSV](summary.csv): six checkpoints, raw derived metrics in percent.
- [Per-predicate CSV](per-predicate.csv): 1,860 rows with names, training counts,
  validation support, R@50/100 and top-50 prediction counts. JSON scores use
  fractions in [0,1]; CSV scores explicitly use percent columns.
- Individual `mlp-*.json` / `gat-*.json`: complete reference results and
  subgroup diagnostics, including exact stored-score agreement.
- Script: `tools/diagnose_gqa_quality.py`; log: `logs/gqa-quality-v1.log`.
  Immutable checkpoint copies: `runs/diagnostics/gqa-quality-v1/` (ignored by Git).
  The one-shot script refuses an existing evidence directory; a new execution
  must use a new versioned destination and record its new source hash.
- Inference: frozen cached CLIP inputs, float32, eval/inference mode, batch 64,
  CUDA allocator cap 3 GiB. Maximum measured allocated tensors: **781.80 MiB**
  (GAT); driver/context and reserved memory are additional. No training, test
  evaluation, checkpoint update or W&B logging was performed by this tool.
  The existing single training process continued throughout the diagnosis.
- Metrics: graph-constrained PredCls with the existing reference matcher,
  original GT duplicate rows retained, per-image R@K, per-class image-averaged
  recalls then a fixed-class mean for mR@K. Group R averages only images having
  GT in that group; supported-class mR is separately labeled.
- Validation: all six mR@50 differences from checkpoint scores are exactly zero;
  all 310 per-class validation supports match the prior scale audit. Two targeted
  tests check original versus intervened ranking, duplicate GT handling,
  absent-class means and non-mutation of logits.
