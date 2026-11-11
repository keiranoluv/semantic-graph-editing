# GQA RAM loader

The optional launcher preloads the existing train/validation graphs into CPU RAM
once, using eight spawned CPU workers. The same resident graphs serve all 15 runs.
Workers call the original `GQAFullDataset`; every serialized tensor is checked
after transfer to the parent. Image order, graph construction, collation,
sampling, batch sizes, optimizer, validation and checkpoint selection are unchanged.
Training retains the frozen single-thread Torch setting. Workers exit after
preloading; this optimization removes repeated HDF5 reads and graph construction.

```bash
scripts/run_gqa_full_ram.sh
# Explicit equivalent:
scripts/run_gqa_full_ram.sh --stage train --loader-workers 8 \
  --config configs/level2/gqa-full-v3-bs512.json
```

Stop the existing launcher with SIGTERM and wait for it to save and exit before
starting this launcher. It uses the same output lock, `last.pt`, experiment
identity and W&B run ID. Do not delete the lock or alter the frozen recipe.
The original launcher remains available as a fallback and can resume the same
checkpoint after the RAM launcher exits.

This is an explicit runtime overlay: the frozen trainer/source fingerprint stays
unchanged, while the additional loader's source SHA256, worker count, image counts,
tensor bytes and loading times are recorded in
`runtime-ram-loader-<PID>.json` in the experiment directory. Loader identity and
worker count are also recorded in W&B summary. Reproduction requires this file
and the recorded overlay source in addition to `experiment.json`.

Validation includes exact tensor equality on a 4,096-image real-data sample,
identical sampled batches across three repetitions, and a synthetic PE-PLPR
interruption/resume test with exact final weights and validation history. The CPU
benchmark in `reports/level2/gqa-ram-loader-v1/benchmark.json` measures data loading
plus collation only: 4.70 seconds versus 1.10 seconds for eight batches, about
4.29 times faster. This ratio is not an end-to-end training speedup estimate.
GPU epoch timings must be measured separately. No test split is accessed.
