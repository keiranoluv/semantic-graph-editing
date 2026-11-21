# semantic-graph-editing

VG150 scene graph learning, external knowledge exploration and planned controllable image editing.

**Current scope:** VG150 PredCls MLP/GAT/GAT+KG with frozen CLIP region features; reviewed WordNet/ConceptNet mappings and TransE/RotatE/ComplEx exports. The 54 eligible frozen checkpoints have both graph-constrained and no-graph-constraint validation evaluations: [side-by-side results](reports/level1/reference-comparison/comparison.md). The main matrix also has historical legacy validation/test results; six relation-prior runs remain excluded pending L1/L2 resolution. See the [short Level 1 report](docs/LEVEL1_REPORT.md), [simple demo with both scoring modes](notebooks/level1_reference_demo.ipynb), [evaluation protocol](docs/evaluation/PredCls.md) and [audit](docs/experiments/level1_audit_2026-11-11.md). Human error analysis and the final submission remain unfinished.

**Level 3:** [One pipeline, two independent commands](docs/LEVEL3_PIPELINE.md): Qwen-Image-2.1 creates 200 edited images from 100 source images; REACT++ produces 300 scene graphs from the same run. Both steps resume their own completed artifacts. The current run is complete; use the same new `RUN_ID` for both commands to reproduce it. See the [cleanup record](docs/LEVEL3_CLEANUP.md) for retained artifacts and retired pilot code.

Start with the [submission demo for all three tasks](notebooks/project_submission_demo.ipynb). Run it on CPU in the project environment with `myenv/bin/jupyter-lab notebooks/project_submission_demo.ipynb`. It reproduces the VG150 and GQA validation Hit/Miss examples, runs a MagicBrush checkpoint, replays saved He et al. predictions and scores a VN100 image pair.

Open the [live graph editing notebook](notebooks/graph_editing_live_demo.ipynb) in the project environment with `myenv/bin/jupyter-lab notebooks/graph_editing_live_demo.ipynb`. It runs the selected MagicBrush GNN on CPU and recomputes graph consistency for the independently edited Qwen image using the local experiment artifacts.

- [Level 1 progress and collaboration rules](docs/LEVEL1_PROGRESS.md)
- [Model guide: inputs, outputs, scoring examples and purpose of every experiment](docs/LEVEL1_MODEL_GUIDE.md)
- [Long-tail analysis: train-defined head/body/tail for all 54 checkpoints](reports/level1/long-tail-reference-v1/analysis.md)
- [Zero-shot recall: nine fixed fusion checkpoints, GC/NG and per-seed comparisons](reports/level1/zero-shot-reference-v1/analysis.md)
- [Team environment and VG150 download commands](docs/TEAM_SETUP_LEVEL1.md)
- [Reviewed WordNet export, GAT+WordNet and GPU commands](docs/knowledge/c2_gat_wordnet.md)
- [Frozen CLIP region features and matched Level 1 GPU matrix](docs/experiments/level1_clip_regions.md)
- [WordNet mapping review and representative-sense notes](docs/knowledge/LEVEL1_MAPPING_REVIEW.md)
- [Level 1 handoff: Markdown report and demo](docs/LEVEL1_SUBMISSION.md)
- [Level 2 startup: GQA adapter, graph-edit operations, scoring and remaining requirements](docs/LEVEL2_PROGRESS.md)

- [Team reproducibility guide and offline/online + two evaluation modes](docs/TEAM_REPRODUCIBILITY_AND_PROTOCOLS.md)
- [Step 6 repository audit and known issues](docs/STEP6_REPOSITORY_AUDIT.md)
- [VG150 dataset setup](docs/datasets/VG150.md)
- [PredCls evaluation protocol](docs/evaluation/PredCls.md)
- [Human review required](docs/knowledge/HUMAN_REVIEW_REQUIRED.md)
- [ConceptNet B.2.1 pipeline](docs/knowledge/b2_conceptnet_workflow.md)

Never commit downloaded data, generated graphs, checkpoints or private credentials. Respect ConceptNet CC BY-SA attribution rules.
