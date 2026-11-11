# Two concurrent GQA runs

```bash
scripts/run_gqa_full_parallel.sh
```

This coordinator schedules the existing 15 model/seed runs onto two persistent,
isolated Python processes sharing the GPU. Each trainer owns eight logical CPUs
through OS affinity, for a total budget of 16 out of 32. Four CPU loader processes
per trainer inherit that affinity. Tensor assembly uses eight threads per trainer;
the previous thread count is restored before model, loss, and optimizer operations.
Numerical Torch operations retain the frozen one-thread setting. Allocating 16 CPUs does not mean 50% continuous utilization;
actual utilization depends on the training and validation phases.

Each trainer retains prepared graphs/supervision and up to 2.5 GiB of immutable
validation batches in RAM. Two dataset copies plus validation caches target
approximately half of this host's 60 GiB RAM. These caches perform useful work;
no memory or CPU is consumed solely to raise resource utilization figures.

The coordinator holds the original experiment-wide lock. Each child also retains
a duplicate of that lock's open file descriptor, so an orphaned trainer continues
to exclude legacy launchers after an unexpected coordinator death. Jobs are
dequeued once and additionally hold a per-run lock. Existing checkpoints, frozen
source/config identity, random sampling, early stopping and W&B IDs are retained.
Each process has independent CPU/CUDA RNG state. Completed runs are verified and
skipped by the original trainer; the original matrix summary runs after success.

Send SIGTERM to the **coordinator PID** printed by the script to request safe
checkpoint saving in both workers. Rerun the same script to resume. Do not start
another launcher until the coordinator and workers have exited.

The coordinator log prints active runs and completed-epoch counts. Detailed
training logs are `logs/gqa-parallel-<coordinator PID>-worker0.log` and `worker1.log`.
Runtime CPU allocation, worker PIDs, cache sizes and source hashes are recorded in
the experiment directory and W&B summaries.

The disposable real-data benchmark compares 24 training batches per task, with
the previous production job still active in the background. Sequential versus
concurrent model hashes and sampling hashes match exactly. Measured total task
throughput increased by 2.06x for MLP/MLP, 1.62x for GAT/PE-PLPR, and 1.80x for
PE-PLPR/PE-PLPR. This benchmark excludes validation and is not a claim that every
individual epoch becomes faster. A single run can slow down while the full matrix
finishes sooner. Validation caching is tested against all five reference models.

The frozen recipe is intentionally unchanged; this runtime scheduling overlay and
its source hashes must accompany `experiment.json` when reproducing the run.
