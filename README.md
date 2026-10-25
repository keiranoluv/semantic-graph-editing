# semantic-graph-editing

VG150 scene graph learning, external knowledge exploration and planned controllable image editing.

**Current scope:** VG150 PredCls MLP/GAT/GAT+WordNet, Toan's official representative WordNet mapping for 150 classes, C.1 TransE, C.2 feature export and draft ConceptNet preparation. The annotation-only three-seed matrix is complete. A new matched matrix with frozen CLIP ViT-B/32 region features is being prepared; its GPU runs are pending. Cross-KG identity alignment, image editing and no-graph-constraint evaluation are not delivered.

- [Level 1 progress and collaboration rules](docs/LEVEL1_PROGRESS.md)
- [Team environment and VG150 download commands](docs/TEAM_SETUP_LEVEL1.md)
- [Reviewed WordNet export, GAT+WordNet and GPU commands](docs/knowledge/c2_gat_wordnet.md)
- [Frozen CLIP region features and matched Level 1 GPU matrix](docs/experiments/level1_clip_regions.md)
- [WordNet mapping review and representative-sense notes](docs/knowledge/LEVEL1_MAPPING_REVIEW.md)
- [Level 1 submission draft: report, slides and demo](docs/LEVEL1_SUBMISSION.md)

- [Team reproducibility guide and offline/online + two evaluation modes](docs/TEAM_REPRODUCIBILITY_AND_PROTOCOLS.md)
- [Step 6 repository audit and known issues](docs/STEP6_REPOSITORY_AUDIT.md)
- [VG150 dataset setup](docs/datasets/VG150.md)
- [PredCls evaluation protocol](docs/evaluation/PredCls.md)
- [Human review required](docs/knowledge/HUMAN_REVIEW_REQUIRED.md)
- [ConceptNet B.2.1 pipeline](docs/knowledge/b2_conceptnet_workflow.md)

Never commit downloaded data, generated graphs, checkpoints or private credentials. Respect ConceptNet CC BY-SA attribution rules.
