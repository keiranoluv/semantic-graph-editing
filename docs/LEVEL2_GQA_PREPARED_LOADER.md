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

The CPU benchmark measured eight 512-image batches at one thread in 0.384 seconds
with prepared supervision versus 1.138 seconds with the reference collator.
Four threads reduced prepared collation to 0.302 seconds, but the deployed
configuration retains one Torch thread to keep the complete frozen computation
unchanged. The large improvement comes from reusing fixed supervision.
All 332 tests passed before deployment; end-to-end timings are recorded separately.

The live run resumed at epoch 27, image offset 69,120, in tmux `0:gqa-ram` with
the same W&B run. Complete epochs 28 and 29 took 23.01 and 23.42 seconds. The mean
23.22 seconds is about 1.55 times faster than the earlier RAM-only measurement
(35.94 seconds), or 4.46 times faster than the original HDF5 loader (103.51 seconds).
These are MLP-only wall times including validation/checkpoints, not estimates for
the other methods. The resident graph/supervision tensor payload is 7.88 GiB.
See `reports/level2/gqa-prepared-loader-v1/live-resume.json` for provenance.

A read-only validation diagnosis of the MLP seed 42 best checkpoint at epoch 25
reproduced its mR@50 exactly. Ten predicates account for 96.78% of validation GT
annotation rows. Of 267 predicates with validation GT, only 41 had nonzero R@50.
The fixed mean includes all 310 foreground predicates, with absent predicates
assigned zero. R@50 was 73.87% and mR@50 was 1.87%. This indicates limited class
coverage despite improving aggregate recall. It is not a reason to change the
recipe halfway through the matched matrix. The complete per-class evidence is
in `reports/level2/gqa-prepared-loader-v1/validation-before-resume.json`.
