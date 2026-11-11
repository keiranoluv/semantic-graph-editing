# Team reproduction guide: VG150, WordNet, ConceptNet, training and evaluation

Scope: `keiranoluv/semantic-graph-editing`, branch `feat/step6-conceptnet-b2`, audited at commit `db760747babea21f4e40cfbba78f2e57fd293ecf` on 2026-10-19. This is a *working draft pipeline*, **not** a finished scene-graph editing system or a validated GAT+KG model.

## 0. Rules for all team members

- Work from the repository root; use a separate branch for each next milestone. Record the **full** `git rev-parse HEAD` and `python --version` for each experiment.
- Do not commit downloaded datasets, large generated graphs, credentials, W&B run data or model checkpoints. Do not use `git add .` indiscriminately.
- Keep VG150 class IDs **1-150**, background **0**, predicates **1-50**, predicate background **0**. Never reorder by name.
- Separate ***candidate coverage*** from semantic correctness. `linked_manual` in the AI draft is a legacy schema compatibility status, **not** human approval.
- WordNet/ConceptNet preprocessing must not use validation/test ground-truth (GT) relations or GT-derived statistics. Test is for final evaluation only.
- Output writers refuse conflicting existing files. For a changed snapshot, mapping or policy, use a new versioned output directory instead of deleting previous results.

## 1. Clone and Python environment

```bash
# Do this once on a new machine.
git clone https://github.com/keiranoluv/semantic-graph-editing.git
cd semantic-graph-editing
git switch feat/step6-conceptnet-b2
python --version
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# Install a Torch/PyG build compatible with the host CPU/GPU first.
# Then install the other project dependencies in your environment:
python -m pip install h5py numpy nltk tqdm pytest pyyaml wandb pre-commit
python -m pip freeze > /tmp/semantic_graph_editing_packages.txt
```

`requirements.txt` currently contains only `wandb>=0.18`; **it is not a complete, pinned environment specification**. Torch + torch-geometric installation varies by CUDA/platform. Python 3.10 is the version targeted by `pyproject.toml`; a Python 3.12 local run has also been reported, but there is no committed full compatibility matrix. Pin package versions / environment lock before comparing machines.

For knowledge-only tests (not GAT training), install `nltk`, `pytest` and their dependencies. Full GAT training additionally requires `torch`, `torch_geometric`, and other training packages.

## 2. Dataset VG150 (download once per teammate)

Source script: `scripts/download_vg150.sh`. It retrieves the VG-SGG HDF5 from `kb-kim/LLM4SGG` (Hugging Face), class vocabulary JSON from `Prk10/VG150_VisualGenome` (GitHub), and image metadata ZIP from the Visual Genome website.

```bash
bash scripts/download_vg150.sh
python scripts/build_vg150_manifest.py \
  --h5 data/raw/vg150/VG-SGG-with-attri.h5 \
  --metadata data/raw/vg150/image_data_raw.json \
  --output-dir data/raw/vg150
```

Pinned HDF5 SHA-256 in the downloader:

```
ad52a01f8e6142bb7bc62d91f97dad1567f44772a23eac68d60a5b7766ffe4b1
```

Verify manually:

```bash
sha256sum data/raw/vg150/VG-SGG-with-attri.h5 \
  data/raw/vg150/VG-SGG-dicts-with-attri.json \
  data/raw/vg150/image_data.json.zip
python - <<'PY'
import json
from pathlib import Path
p=Path('data/raw/vg150/splits.json')
s=json.loads(p.read_text())
print({k:len(v) for k,v in s.items()})
PY
```

Only the HDF5 SHA is currently pinned in the download script. The vocabulary JSON and image metadata are fetched from mutable URLs, so save their SHA-256 values and check that the whole team uses the SAME files. Expected split sizes for this repository's exact preprocessing policy: **train 57,723; validation 5,000; test 26,446**. If existing `image_data.json` or `splits.json` exist, the manifest builder refuses to overwrite them. **Do not rerun it blindly**; verify existing files or generate into a fresh directory and update `--root` appropriately.

Geometry-only MLP/GAT PredCls uses annotation boxes/classes: JPEG images are **not required for that baseline**. Real image editing/visual features are separate future steps and may require actual images. The current split/preprocessing is project-specific and not yet proven benchmark-identical.

## 3. WordNet A.1, A.2 and B.1

Install WordNet explicitly; this is a separate corpus download, not a ConceptNet API call:

```bash
python -m nltk.downloader wordnet
python -m scripts.prepare_knowledge --dry-run
python -m scripts.prepare_knowledge
python -m scripts.prepare_wordnet_review rank --dry-run
python -m scripts.prepare_wordnet_review rank
python -m scripts.build_wordnet_graph --max-hops 2 --dry-run
python -m scripts.build_wordnet_graph --max-hops 2
```

Files produced:

- A.1: `data/knowledge/wordnet-vg150/wordnet_mapping.json` and `wordnet_coverage.json`.
- A.2: `data/knowledge/wordnet-review-v1/wordnet_rankings.json` and editable `wordnet_review.csv`.
- B.1 basic draft: `data/knowledge/wordnet-kg-draft-v1/{entities.json,relations.json,triples.tsv,class_entity_ids.json,metadata.json}`.

**Optional A.2 AI draft:** `python -m scripts.build_ai_wordnet_draft` requires an additional `data/knowledge/wordnet-review-v1/wordnet_review_ai_provisional.csv`, which **is not created by the commands above and is not tracked on the audited branch**. Obtain the exact 150-row AI-provisional CSV + its checksum from the group artifact owner; do not substitute the empty `wordnet_review.csv` or invent a `reviewer`.

Once the exact CSV is provisioned, run:

```bash
python -m scripts.build_ai_wordnet_draft --dry-run
python -m scripts.build_ai_wordnet_draft
```

Observed reference draft: **124/150** AI-provisional classes, **26** deferred; **122** unique WordNet roots, **967** entities and **1,156** triples. These are not human-verified. Keep `ready_for_gat_kg_training=false`.

Human decisions must start from the *original* A.1 map, not AI-draft compatibility `linked_manual` statuses. See `docs/knowledge/HUMAN_REVIEW_REQUIRED.md` and `docs/knowledge/a2_b1_workflow.md`.

## 4. ConceptNet B.2.1 - offline pinned assertions, exact reproduce

Official download index: https://github.com/commonsense/conceptnet5/wiki/Downloads

Source of the 2019 ConceptNet **5.7.0** assertions snapshot:

```
https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
```

The file is `*.csv.gz` **by name but TSV by format**, five columns per line. Stream directly from gzip; DO NOT unzip, query live API, or use another release while claiming 5.7.0.

```bash
mkdir -p data/external/conceptnet
curl -fL --retry 3 \
  'https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz' \
  -o data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz
sha256sum data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz
```

**Observed group snapshot SHA-256** (from the completed local B.2.1 extraction):

```
accd65fe94038584295574ddc26e1500c1919c8c4532bf771811cafd0948af7e
```

Do **not** assume this is independently authenticated by the ConceptNet maintainers; it is the team's reference snapshot fingerprint. Verify the upstream source and require exact equality across teammate machines.

```bash
CN_FILE=data/external/conceptnet/conceptnet-assertions-5.7.0.csv.gz
CN_SHA=accd65fe94038584295574ddc26e1500c1919c8c4532bf771811cafd0948af7e
printf '%s  %s\n' "$CN_SHA" "$CN_FILE" | sha256sum -c -

# Exact Step 6 group policy: do NOT omit --max-edges 300000.
# Default 50000 FAILS on the reference snapshot (56703 retained assertions).
python -m scripts.build_conceptnet_graph \
  --assertions "$CN_FILE" \
  --expected-sha256 "$CN_SHA" \
  --mapping data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json \
  --output-dir data/knowledge/conceptnet-b2-v1 \
  --min-weight 1.0 \
  --max-edges 300000 \
  --dry-run

python -m scripts.build_conceptnet_graph \
  --assertions "$CN_FILE" \
  --expected-sha256 "$CN_SHA" \
  --mapping data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json \
  --output-dir data/knowledge/conceptnet-b2-v1 \
  --min-weight 1.0 \
  --max-edges 300000
```

**Alternative if the AI CSV is not provisioned:** B.2 uses only class **labels** from its `--mapping` input, so use the original A.1 map instead, with a different output directory. Expect differences in metadata and the WordNet-review columns (do not claim byte-for-byte identical artifacts):

```bash
python -m scripts.build_conceptnet_graph \
  --assertions "$CN_FILE" --expected-sha256 "$CN_SHA" \
  --mapping data/knowledge/wordnet-vg150/wordnet_mapping.json \
  --output-dir data/knowledge/conceptnet-b2-a1-labels-v1 \
  --min-weight 1.0 --max-edges 300000
```

Reference B.2.1 local output: **34,074,917 input assertions**, **46,414 entities**, **56,703 evidence rows**, **56,703 unique triples**, **10 relations**, **150/150 class lexical-candidate coverage**. These are local reference results, not an independent online benchmark. Outputs: `entities.json`, `relations.json`, `triples.tsv`, `class_concept_candidates.json`, `review_queue.csv`, and `metadata.json` under `data/knowledge/conceptnet-b2-v1/`.

`150/150` means a term candidate was found, **NOT** that all 150 semantic senses are correct. No `SameAs` links are created. `human_verified_class_count=0` and `ready_for_gat_kg_training=false`. Respect ConceptNet CC BY-SA 4.0 and per-assertion attribution/license fields when redistributing derived graph data.

## 5. What "offline" and "online" mean (two DIFFERENT topics)

### A. Knowledge retrieval/extraction

- **Offline (IMPLEMENTED):** WordNet corpus installed locally; ConceptNet 5.7.0 assertions downloaded explicitly once, hashed and processed locally. The `build_conceptnet_graph` command has no network calls. Preferred for scientific reproducibility.
- **Online ConceptNet API (NOT IMPLEMENTED):** live lookup per concept via network. Potentially useful for demonstration, but endpoints/content may change and would require caching, version/response recording, throttling and extra tests. **Do not describe this as a supported execution mode in Step 6.**

### B. Training experiment tracking (Weights & Biases)

`python -m scripts.train_gat` and `python -m scripts.pilot_mlp` expose `--wandb-mode {online,offline,disabled}`:

| Mode | Effect | Use |
| --- | --- | --- |
| `online` | Send run metrics to W&B while training; requires network/auth | Shared experiment tracking |
| `offline` | Store W&B run logs locally for possible later sync | Machines without network (default for both runners) |
| `disabled` | Do not initialize W&B; local checkpoints still work | Smoke testing and debug |

To upload a saved offline run later, use `wandb sync <path-to-offline-run>` once connected, after checking the installed W&B CLI's help. This has **no relation** to ConceptNet offline extraction.

## 6. TWO PredCls evaluation modes (do not conflate with offline/online)

| Mode | Behavior per ordered (subject, object) pair | Metrics | Current status |
| --- | --- | --- | --- |
| **A. Graph-constrained** | Predict ONE highest-scoring foreground predicate per pair, rank per image | `R@50/100`, `mR@50/100` | **IMPLEMENTED** in `src/evaluation/scene_graph.py` |
| **B. No-graph-constraint** | Allow MULTIPLE foreground predicates per pair, rank all candidate triplets per image | `ngR@50/100`, `ngmR@50/100` | **TODO**, no runnable `--eval-mode` flag yet |

Both modes are *PredCls*: GT object labels/boxes, predicted relation labels, all ordered non-self object pairs as candidates. Neither mode means W&B online/offline, and neither is "train vs test". Preserve multiple ground-truth predicates for a pair. For controlled comparisons use the **same** annotations, split, box policy, K values and per-image ranking. Do not compare directly with published VG150 SGG scores until preprocessing/metric compatibility is verified.

Current implemented evaluator uses full softmax (including background) then ranks the best foreground predicate per pair; background is **not** emitted as a predicted triplet. The current `R@K` is image-macro over images with GT relations; `mR@K` aggregates per-predicate hits/support over predicates with GT support. Best model selection uses **validation mR@50**.

```bash
# Replace checkpoint path with an ACTUAL existing trained checkpoint.
python -m scripts.evaluate_gat \
  --checkpoint checkpoints/experiments/gat_seed42/run_0001/best.pt \
  --root data/raw/vg150 --split val --batch-size 32 \
  --output reports/gat_seed42_val.json

python -m scripts.evaluate_mlp \
  --checkpoint checkpoints/experiments/mlp_semantic_bs512_seed42/run_0001/best.pt \
  --root data/raw/vg150 --split val --batch-size 128 \
  --max-images 5000 --output reports/mlp_semantic_seed42_val.json
```

Checkpoint paths above are **examples**; inspect local experiment directories. The GAT evaluator requires `--allow-test` for a `test` split. The MLP evaluator currently does **not** implement an equivalent explicit test gate, so team policy must still restrict test evaluation until final reporting. The MLP evaluation default is `--max-images 100` (a pilot, NOT full validation); the GAT evaluator default is the full requested split.

Official GAT baseline protocol: `docs/experiments/step2_gat_baseline.md`. For future no-graph evaluation, implement separate regression tests, explicit CLI choice and mode-labeled reports rather than silently changing existing metrics.

## 7. Training, validation and test rules

- Validation can select best checkpoints and tune hyperparameters. Test only for a single final locked evaluation.
- Model comparison: Geometry MLP, Semantic MLP, GAT-only, future GAT+KG; preserve common dataset/metrics, record seed `42,123,3407`, epoch, optimizer steps, training budget, mR, R and GPU hours.
- The GAT-only baseline is implemented; `configs/gat_kg.yaml` is currently empty and TransE/GAT+KG training is **not** implemented in the audited branch. Do not run final KG training from either provisional graph.
- `scripts/evaluate.py`, `src/evaluation/ablation.py` and `src/evaluation/metrics.py` currently do not define independent implementations; use documented working entry points.
- Both B.2.2 (cross-KG sense bridge) and Mode B no-graph evaluation are later tasks.

## 8. Human review required before final KG research

1. Review **all 124 AI-provisional WordNet mappings** independently, including the 23 auto links; **26 deferred** VG class IDs require special attention.
2. Check **all 150 ConceptNet lexical candidates** and edge evidence (word matching is not meaning equality).
3. Build a separately reviewed WordNet-ConceptNet bridge with source hashes, human reviewer, decision, rationale, evidence and timestamps; do not auto-add `SameAs`.
4. Keep `ready_for_gat_kg_training=false` until approved. The KG draft is suitable for exploration/topology QA only.

Full gate and priorities: `docs/knowledge/HUMAN_REVIEW_REQUIRED.md`.

## 9. Teammate acceptance checks (no model training)

```bash
python -m pytest tests/knowledge -q
python -m pytest tests/evaluation -q
python -m scripts.prepare_wordnet_review --help
python -m scripts.build_ai_wordnet_draft --help
python -m scripts.build_conceptnet_graph --help
pre-commit run --all-files
```

Track the outputs and repository revision in the team log. For every real dataset, record the source URL, SHA-256, extraction command and metadata. No CI workflow was visible under `.github/workflows` in the audited Step 6 tree; local test PASS is **not** proof of CI execution.
