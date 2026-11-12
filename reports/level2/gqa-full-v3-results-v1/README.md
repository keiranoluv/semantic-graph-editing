# Full GQA v3: completed 15-run validation results

All 5 configurations × seeds 42, 123, 3407 completed successfully. Each run stopped by early stopping (patience 15). Best checkpoints were selected using validation GC mR@50. These are validation-selected results on custom GQA with 310 predicates, not test or GQA200 benchmark results.

Scores are percent, mean ± sample standard deviation across three seeds.

| Model | GC R@50 | GC mR@50 | GC R@100 | GC mR@100 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 76.66 ± 0.59 | 3.33 ± 0.34 | 87.24 ± 0.40 | 5.68 ± 0.75 |
| gat | 77.34 ± 0.13 | 3.10 ± 0.10 | 87.83 ± 0.12 | 4.84 ± 0.22 |
| pe-ce | 78.07 ± 0.27 | 3.00 ± 0.28 | 88.39 ± 0.25 | 4.88 ± 0.58 |
| pe-pl | 78.14 ± 0.07 | 3.42 ± 0.14 | 88.51 ± 0.05 | 5.61 ± 0.48 |
| pe-plpr | 77.97 ± 0.07 | 4.16 ± 0.17 | 88.28 ± 0.08 | 7.13 ± 0.31 |

## Paired changes in GC mR@50

Differences are percentage points. Positive values favor the first configuration. Three observed seeds do not establish statistical significance.

| Comparison | Mean change | Seed 42 | Seed 123 | Seed 3407 |
| --- | ---: | ---: | ---: | ---: |
| gat minus mlp | -0.233 | -0.628 | +0.247 | -0.318 |
| pe-pl minus pe-ce | +0.423 | +0.163 | +0.794 | +0.313 |
| pe-plpr minus pe-pl | +0.735 | +0.677 | +0.455 | +1.072 |
| pe-plpr minus mlp | +0.823 | +0.512 | +1.035 | +0.922 |

PE-PLPR has the highest mean GC mR@50 and mR@100. PL improves GC mR@50 over PE-CE on all three seeds; PR improves it further on all three seeds. PE-PLPR also beats MLP in GC mR@50 on each seed. Its mean mR@50 improves by 0.823 percentage points (24.7% relative) over MLP; the absolute score remains low.

GAT improves mean R@50 over MLP but has lower mean mR@50. PE-PL has the highest mean R@50/100; PE-PLPR gives up approximately 0.17 percentage points of R@50 versus PE-PL for improved mean recall. These findings apply to this implementation and recipe. They are not claims about the original papers or universal model rankings.

## Rare-class coverage

Groups are fixed by training frequency. Tail mR uses all 260 predicates outside the 50 most frequent, including zero for absent classes. There are 267 supported classes overall, 217 in this tail group. Coverage counts classes with at least one successful top-50 retrieval, not reliable recognition across all examples.

| Model | Classes recalled, seed 42 / 123 / 3407 | Mean tail-260 mR@50 (%) |
| --- | --- | ---: |
| mlp | 61 / 51 / 60 | 1.472 |
| gat | 51 / 51 / 49 | 1.434 |
| pe-ce | 53 / 46 / 48 | 0.994 |
| pe-pl | 49 / 57 / 50 | 1.336 |
| pe-plpr | 64 / 64 / 66 | 1.895 |

## Evidence and remaining scope

- [Per-run CSV](runs.csv): selected epoch, stopping epoch, metrics and coverage.
- [Summary JSON](summary.json): original full aggregation, including separately labeled NG metrics.
- [Verification receipt](verification.json): source paths, result/checkpoint hashes and completed read-only checks.
- Checkpoint hashes were verified; no additional inference or test evaluation was run for this report. The training code itself reloads the best state and checks final GC mR@50 against its selection score before writing each result.
- The earlier `gqa-quality-v1` diagnosis retains historical snapshots: GAT 123 and 3407 were unfinished then. Use this report for the completed 15-run comparison.
- This completes the planned five-model GQA training matrix. GAT+KG, matched VG150 comparisons for the adapted 2023+ method, and the graph-editing dataset/training/evaluation remain separate requirements; these 15 runs do not by themselves complete Level 2.
- Keep the test split held out during further modeling decisions. Plan the final test evaluation only after the protocol and model choices are frozen.
