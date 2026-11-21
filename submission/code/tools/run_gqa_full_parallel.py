"""Two isolated trainers sharing one GPU, with a 16-logical-CPU affinity budget."""

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import queue
import signal
import time
from contextlib import contextmanager
from multiprocessing.reduction import DupFd
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write, cache_identity, cohort_for, pin_json, verify_cache
from src.knowledge.transe.core import sha256
from tools import run_gqa_full_ram as ram
from tools.run_gqa_full_prepared import load_prepared_shard, prepared_collate


def threaded_collate(graphs, num_classes, generator=None):
    """Eight threads for tensor assembly only; restore numerical runtime afterwards."""
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        return prepared_collate(graphs, num_classes, generator)
    finally:
        torch.set_num_threads(previous)


class ValidationCache:
    """Bounded cache of immutable batches; training never enters this cache."""

    def __init__(self, max_bytes=int(2.5 * 1024**3)):
        self.max_bytes, self.bytes, self.hits = max_bytes, 0, 0
        self.batches = {}

    def __call__(self, graphs, num_classes, generator=None):
        if generator is not None:
            return threaded_collate(graphs, num_classes, generator)
        key = (num_classes, tuple(id(g) for g in graphs))
        if key in self.batches:
            self.hits += 1
            return self.batches[key]
        batch, slices = threaded_collate(graphs, num_classes)
        size = sum(t.numel() * t.element_size() for t in batch.values())
        if self.bytes + size <= self.max_bytes:
            self.batches[key] = batch, slices
            self.bytes += size
            return self.batches[key]
        return batch, slices


@contextmanager
def coordinator_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing {directory}") from error
        # Children retain duplicated descriptors for this same open file description.
        # If the coordinator dies, the root lock survives until all trainers exit.
        yield lock.fileno()


def worker(slot, cpus, config, metadata, experiment, jobs, messages, stopped, lock_fd, source_hashes, log_path):
    inherited_lock = lock_fd.detach()
    os.sched_setaffinity(0, cpus)
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    log = open(log_path, "a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    from src.training import gqa_full
    from src.training.gqa_tracking import RunTracker, preserve_rng
    from tools import train_gqa_pilot as evaluator
    from tools.run_gqa_full import exclusive

    gqa_full.setup_runtime(config)
    runtime = {
        "slot": slot,
        "pid": os.getpid(),
        "cpus": cpus,
        "loader_workers": 4,
        "collate_threads": 8,
        "source_sha256": source_hashes,
    }
    runtime_path = Path(config["output"]) / f"runtime-parallel-worker-{os.getpid()}.json"

    class ParallelTracker(RunTracker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.run is not None:
                with preserve_rng():
                    self.run.summary.update(
                        {
                            "runtime/parallel_workers": 2,
                            "runtime/cpu_affinity": cpus,
                            "runtime/collate_threads": 8,
                            "runtime/parallel_source_sha256": source_hashes,
                        }
                    )

    train = val = None
    try:
        ram.load_shard = load_prepared_shard
        train = ram.RAMGQADataset(config["cache"], metadata, "train", workers=4)
        if stopped.is_set():
            return
        val = ram.RAMGQADataset(config["cache"], metadata, "val", workers=4)
        validation = ValidationCache()
        for begin in range(0, len(val), config["val_batch_size"]):
            if stopped.is_set():
                return
            validation(val[begin : begin + config["val_batch_size"]], experiment["dimensions"]["num_classes"])
        runtime.update(train=train.statistics, val=val.statistics, validation_cache_bytes=validation.bytes)
        atomic_write(runtime_path, runtime)
        gqa_full.collate, evaluator.collate, gqa_full.RunTracker = threaded_collate, validation, ParallelTracker
        vectors = gqa_full.load_vectors(config, experiment["dimensions"])
        messages.put({"event": "ready", **runtime})
        while not stopped.is_set():
            task = jobs.get()
            if task is None or stopped.is_set():
                break
            kind, seed = task
            directory = Path(config["output"]) / kind / str(seed)
            messages.put({"event": "start", "slot": slot, "model": kind, "seed": seed})
            with exclusive(directory):
                result = gqa_full.train_run(
                    kind, seed, train, val, vectors, experiment, directory, stop_requested=stopped.is_set
                )
            if result is None:
                break
            messages.put(
                {"event": "complete", "slot": slot, "model": kind, "seed": seed, "epochs": result["epochs_completed"]}
            )
            torch.cuda.empty_cache()
    except BaseException as error:
        messages.put({"event": "error", "slot": slot, "error": repr(error)})
        stopped.set()
        raise
    finally:
        if train is not None:
            train.close()
        if val is not None:
            val.close()
        os.close(inherited_lock)
        log.close()


def main():
    from src.training.gqa_full import summarize_matrix
    from tools.run_gqa_full import experiment_identity, load_config, preflight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/gqa-full-v3-bs512.json"))
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 16:
        parser.error("This deployment requires at least 16 allowed logical CPUs")
    cpus = allowed[:16]
    os.sched_setaffinity(0, cpus)
    config = load_config(args.config)
    manifest, plan = preflight(config)
    output = Path(config["output"])
    ctx = mp.get_context("spawn")
    stopped, jobs, messages = ctx.Event(), ctx.Queue(), ctx.Queue()

    def stop(signum, _frame):
        print(f"[SIGNAL {signum}] both trainers will save at their next safe boundary", flush=True)
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    sources = [Path(__file__), Path(ram.__file__), Path("tools/run_gqa_full_prepared.py")]
    hashes = {str(p.resolve().relative_to(Path.cwd())): sha256(p) for p in sources}
    state = {
        "pid": os.getpid(),
        "status": "starting",
        "cpu_affinity": cpus,
        "workers": 2,
        "ram_target": "approximately half of host RAM; two prepared graph copies plus bounded validation caches",
        "source_sha256": hashes,
        "active": {},
        "completed": [],
        "logs": [],
    }
    receipt = output / f"runtime-parallel-coordinator-{os.getpid()}.json"
    processes = []
    with coordinator_lock(output) as lock:
        pin_json(output / "requested-config.json", config)
        atomic_write(output / "preflight.json", plan)
        metadata = verify_cache(config["cache"], cache_identity(config, manifest, cohort_for(config, manifest)))
        experiment = experiment_identity(config, manifest)
        pin_json(output / "experiment.json", experiment)
        for kind in config["models"]:
            for seed in config["seeds"]:
                jobs.put((kind, seed))
        for _ in range(2):
            jobs.put(None)
        try:
            Path("logs").mkdir(exist_ok=True)
            for slot in range(2):
                log = f"logs/gqa-parallel-{os.getpid()}-worker{slot}.log"
                state["logs"].append(log)
                process = ctx.Process(
                    target=worker,
                    args=(
                        slot,
                        cpus[slot * 8 : (slot + 1) * 8],
                        config,
                        metadata,
                        experiment,
                        jobs,
                        messages,
                        stopped,
                        DupFd(lock),
                        hashes,
                        log,
                    ),
                )
                process.start()
                processes.append(process)
            state["worker_pids"] = [p.pid for p in processes]
            state["status"] = "running"
            atomic_write(receipt, state)
            print(f"[PARALLEL] {json.dumps(state)}", flush=True)
            last_status = 0.0
            while any(p.is_alive() for p in processes):
                try:
                    event = messages.get(timeout=1)
                    print(f"[WORKER] {json.dumps(event)}", flush=True)
                    if event["event"] == "start":
                        state["active"][str(event["slot"])] = event
                    elif event["event"] == "complete":
                        state["completed"].append(event)
                        state["active"].pop(str(event["slot"]), None)
                    atomic_write(receipt, state)
                except queue.Empty:
                    pass
                if any(p.exitcode not in (None, 0) for p in processes):
                    stopped.set()
                if time.monotonic() - last_status >= 15:
                    for slot, task in state["active"].items():
                        history = output / task["model"] / str(task["seed"]) / "history.json"
                        rows = json.loads(history.read_text()) if history.exists() else []
                        print(
                            f"[STATUS] slot={slot} {task['model']}/{task['seed']} completed_epochs={len(rows)}",
                            flush=True,
                        )
                    last_status = time.monotonic()
            for process in processes:
                process.join()
            if any(p.exitcode != 0 for p in processes):
                state["status"] = "failed"
                return 1
            if stopped.is_set():
                state["status"] = "interrupted"
                return 130
            summarize_matrix(experiment, output)
            state["status"] = "complete"
            return 0
        finally:
            stopped.set()
            for process in processes:
                if process.is_alive():
                    process.join()
            atomic_write(receipt, state)


if __name__ == "__main__":
    raise SystemExit(main())
