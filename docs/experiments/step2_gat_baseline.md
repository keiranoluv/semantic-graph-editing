# Level 1 / Step 2: GAT PredCls Baseline

## Goal

Compare a genuinely graph-based relation predictor with the preserved
Geometry MLP and Semantic MLP baselines on the **same VG150 validation split**.
This is the **GAT-only stage**, not knowledge enhancement or graph editing.

## Inputs and leakage controls

- GT object classes and boxes, as required by PredCls.
- 5D geometry + trainable 32D object label embeddings; no visual region features yet.
- Fully connected **directed** message-passing edges, excluding self-pairs.
- `GATConv` adds self-loops internally. No GT predicate IDs enter topology or features.
- All ordered object pairs are scoring candidates at validation time.
- At training time: 3 background pairs for each positive pair, ignoring `-1`
  ambiguous pairs; positive-free images are skipped (same as MLP pipeline).
- Validation: **graph-constrained** R@50/100 and mR@50/100 via the **existing**
  `src/evaluation/scene_graph.py` evaluator. Best checkpoint = validation mR@50.
- Test is deliberately locked behind `--allow-test` in `scripts/evaluate_gat.py`.

## Architecture

```
Object bbox geometry (5D) + trainable class embedding (32D)
  -> input projection to 256D
  -> GAT (4 heads, 256D output) + residual + LayerNorm + ELU
  -> GAT (4 heads, 256D output) + residual + LayerNorm + ELU
  -> concat ordered subject/object hidden states (512D)
  -> MLP (256D hidden) -> 51 predicate logits incl background
```

This design uses PyG `GATConv` with dropout=0.1 and AdamW. Training
uses a 32-graph micro-batch with 16 gradient accumulation steps (effective
512-graph batch), reducing optimizer-step differences versus the MLP baseline. Graphs are
independent within each mini-batch; GAT receives **full** message-passing
connectivity even though classification targets are sampled. Note that
fully-connected attention is a global context aggregator: shuffling all
edges in a complete graph does **not** provide a valid topology ablation.
A separate sparse-graph design must be specified for later edge ablations.

## Setup

From repository root, use the same Python environment as Step 1, with
`torch`, `torch_geometric`, `h5py`, `tqdm`, `wandb` and `pytest` available.

```bash
pytest -q tests/models/test_gat.py tests/graph/test_gnn_batching.py tests/evaluation/test_gat_evaluation.py tests/training/test_gat_accumulation.py
```

## Smoke training (run FIRST)

```bash
python -m scripts.train_gat \
  --root data/raw/vg150 \
  --device cuda \
  --seed 42 \
  --epochs 2 \
  --max-train-images 256 \
  --max-val-images 64 \
  --batch-size 16 \
  --accumulation-steps 1 \
  --val-batch-size 16 \
  --wandb-mode disabled \
  --run-name gat_smoke_seed42
```

## Full experiments (run only after the smoke test)

```bash
for seed in 42 123 3407; do
  python -m scripts.train_gat \
    --root data/raw/vg150 \
    --device cuda \
    --epochs 200 \
    --patience 15 \
    --batch-size 32 \
    --accumulation-steps 16 \
    --val-batch-size 32 \
    --max-train-images 57723 \
    --max-val-images 5000 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --negative-ratio 3 \
    --seed "$seed" \
    --wandb-mode online \
    --run-name "gat_seed${seed}"
done
```

Runs are created in `checkpoints/experiments/gat_seedNNNN/run_0001/` by
default, with `best.pt`, `last.pt`, `history.json` and `config.json`.
The model uses AdamW with `hidden_dim=256`, two GAT layers, four heads,
and object class embedding dim 32. For this configuration, train effective batch size is 512 graphs; the
final partial accumulation group can be smaller. The optimizer normalizes
summed cross-entropy gradients by the number of selected pairs in each
accumulation group. MLP also used batch 512, but actual compute is **not**
equal (GAT has message passing and up to 200 epochs). Report optimizer
steps, selected pairs, best epoch and elapsed GPU-hours rather than claiming
an equal-compute comparison. If CUDA OOM, lower `--batch-size` and increase
`--accumulation-steps` proportionally.

## Resume and evaluation

```bash
python -m scripts.train_gat \
  --resume checkpoints/experiments/gat_seed42/run_0001/last.pt \
  --epochs 200 --seed 42 --batch-size 32 --accumulation-steps 16 --val-batch-size 32 \
  --wandb-mode online

python -m scripts.evaluate_gat \
  --checkpoint checkpoints/experiments/gat_seed42/run_0001/best.pt \
  --root data/raw/vg150 \
  --split val \
  --batch-size 32 \
  --output reports/gat_seed42_val.json
```

When resuming, all frozen hyperparameters and the data limits must match.
The `--max-images` evaluation option defaults to **all images**.
Never use the test split for checkpoint selection or hyperparameter tuning.

## Acceptance criteria

- Synthetic tests pass (pair offsets, skip behavior, gradient accumulation, batch equivalence).
- Validation R/mR logic agrees with the current MLP evaluator.
- At least one complete seed has a valid `best.pt` and reproducible evaluation.
- Three seeds 42, 123, 3407 for the main comparison, report mean ± sample std.
- Checkpoint includes seed, config, training state and commit SHA.
- Existing MLP files/checkpoints/results remain untouched.

## Research limitations

- No visual features (frozen CLIP/ResNet region features can be added later).
- Fully connected graph topology may limit distinguishable graph structure;
  GAT improvements alone do not prove relationship-specific reasoning.
- Until benchmark preprocessing and metric conventions are fully aligned,
  do not compare these numbers directly with published VG150 SGG results.
