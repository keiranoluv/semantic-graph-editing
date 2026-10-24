# semantic-graph-editing

VG150 scene graph learning, external knowledge exploration and planned controllable image editing.

**Current scope:** MLP/GAT PredCls baselines, draft WordNet/ConceptNet preparation, C.1 TransE, C.2 draft WordNet feature export and a separate GAT+WordNet runner with CPU smoke validation. Full GAT+KG experiments and human-reviewed export remain pending. Cross-KG identity alignment, image editing and no-graph-constraint evaluation are not delivered.

- [Level 1 progress and collaboration rules](docs/LEVEL1_PROGRESS.md)
- [Team environment and VG150 download commands](docs/TEAM_SETUP_LEVEL1.md)
- [Draft feature export, GAT+WordNet and GPU commands](docs/knowledge/c2_gat_wordnet.md)
- [Level 1 submission draft: report, slides and demo](docs/LEVEL1_SUBMISSION.md)

- [Team reproducibility guide and offline/online + two evaluation modes](docs/TEAM_REPRODUCIBILITY_AND_PROTOCOLS.md)
- [Step 6 repository audit and known issues](docs/STEP6_REPOSITORY_AUDIT.md)
- [VG150 dataset setup](docs/datasets/VG150.md)
- [PredCls evaluation protocol](docs/evaluation/PredCls.md)
- [Human review required](docs/knowledge/HUMAN_REVIEW_REQUIRED.md)
- [ConceptNet B.2.1 pipeline](docs/knowledge/b2_conceptnet_workflow.md)

Never commit downloaded data, generated graphs, checkpoints or private credentials. Respect ConceptNet CC BY-SA attribution rules.
