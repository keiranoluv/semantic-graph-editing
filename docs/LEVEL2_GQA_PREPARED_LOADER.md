# GQA prepared supervision in RAM

```bash
scripts/run_gqa_full_prepared.sh
```

This optional launcher extends the graph RAM cache with fixed supervision.
Eight spawned CPU workers call the original `soft_targets` once per graph,
then retain its exact float32 positive rows and ascending positive/background
candidate indices. Background rows are implicit, avoiding a dense 311-class
target allocation for every candidate in the whole dataset.

During training, the collator performs the same random permutations in the same
order, including empty candidate sets, and constructs the same selected pairs and
float32 targets. Validation reconstructs all original rows in candidate order.
Full graph topology is retained for GAT. Cached supervision is reused across
epochs and all 15 runs. Torch's training thread count stays at the frozen setting.

No learning recipe, source under `src/`, checkpoint identity, epoch limit,
early-stopping setting or W&B run ID changes. This is an explicit runtime code
overlay, recorded with both loader source hashes, worker count, tensor bytes and
loading time in `runtime-prepared-loader-<PID>.json`, plus W&B summary metadata.
Keep this provenance alongside `experiment.json` when reproducing results.

The tests compare all tensors, batch slices and sampler RNG for duplicate and
multi-label relations, shuffled candidate order, empty graphs, all-positive
graphs and graphs without GT. Interruption/resume tests for all five models on
CPU and CUDA compare exact weights, optimizer state, RNG and metrics.
The real-data benchmark compares 4,096 graphs at 1/2/4/8 CPU threads for both
training and validation; its scope is CPU collation, not full epoch throughput.

Stop a running launcher with SIGTERM and wait for checkpoint saving and exit
before starting this one. The output lock remains enforced. For fallback,
`scripts/run_gqa_full_ram.sh` and `scripts/run_gqa_full.sh --stage train` can resume
the same saved state. Preloading repeats once when a new process starts.
