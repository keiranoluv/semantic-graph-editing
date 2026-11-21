"""Evaluate the three validation-selected GQA GAT checkpoints against the existing CE controls."""

import argparse
from contextlib import closing
from pathlib import Path
from statistics import mean

import torch

from tools.gqa_test_v1 import runner as ref

OUT = Path("reports/level2/gqa-gat-loss-test-v1")
PLAN = OUT / "validation-plan.json"
CACHE = Path(ref.CACHE_CONFIG["cache"])
BATCH_SIZE = 128


class ProgressDataset(ref.TestDataset):
    def __getitem__(self, index):
        if isinstance(index, slice) and index.start % (8 * BATCH_SIZE) == 0:
            print(f"  test images scored: {index.start}/{len(self)}", flush=True)
        return super().__getitem__(index)


def check_files(files):
    for path, expected in files.items():
        if ref.sha256(Path(path)) != expected:
            raise ValueError(f"File differs from the evaluated protocol: {path}")


def load_inputs(plan):
    check_files(plan["files"])
    original = ref.read(ref.OUTPUT / "plan.json")
    if ref.digest(original) != plan["ce_plan_sha256"]:
        raise ValueError("Unexpected CE test protocol")
    check_files(original["code_sha256"])
    controls = []
    for job in plan["jobs"]:
        check_files({job["checkpoint"]: job["checkpoint_sha256"], job["ce_test"]: job["ce_test_sha256"]})
        row = ref.read(job["ce_test"])
        identity = {
            "plan_sha256": plan["ce_plan_sha256"],
            "cache_manifest_sha256": plan["files"][str(CACHE / "manifest.json")],
            "checkpoint_sha256": job["ce_checkpoint_sha256"],
        }
        if (
            row["identity"] != identity
            or row["model"] != "gat"
            or row["seed"] != job["seed"]
            or row["status"] != "PASS"
            or row["metrics_sha256"] != ref.digest(row["metrics"])
            or row["test_images"] != plan["test_images"]
            or row["batch_size"] != BATCH_SIZE
            or row["torch_version"] != str(torch.__version__)
            or row["device"] != "cuda"
        ):
            raise ValueError(f"CE test protocol mismatch for seed {job['seed']}")
        controls.append({**row, "model": "ce"})
    return controls


def compare(controls, adjusted):
    summary = ref.summarize([*controls, *adjusted])
    paired = {}
    for mode in ("gc", "ng"):
        paired[mode] = {}
        for metric, entry in summary["ce"][mode].items():
            values = summary["adjusted"][mode][metric]["values"]
            differences = [100 * (a - c) for a, c in zip(values, entry["values"], strict=True)]
            paired[mode][metric] = {"per_seed": differences, "mean": mean(differences)}
    return {
        "completed": len(adjusted),
        "complete": len(adjusted) == len(ref.SEEDS),
        "seeds": [r["seed"] for r in adjusted],
        "metrics_fraction": summary,
        "paired_delta_pp": paired,
    }


def run_test(plan, controls):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full test evaluation")
    print("[CHECK] Verifying the existing test cache", flush=True)
    metadata = ref.verify_cache(CACHE)
    ids = ref.read(ref.MANIFEST)["splits"]["test"]
    if metadata["identity"]["splits"] != {"test": ids} or len(ids) != plan["test_images"]:
        raise ValueError("Held-out test split mismatch")
    rows = []
    with closing(ProgressDataset(CACHE, metadata)) as dataset:
        for ix, job in enumerate(plan["jobs"], 1):
            seed = job["seed"]
            path = OUT / "adjusted" / str(seed) / "result.json"
            expected = {
                "model": "adjusted",
                "seed": seed,
                "status": "PASS",
                "plan_sha256": ref.digest(plan),
                "checkpoint_sha256": job["checkpoint_sha256"],
                "test_images": len(dataset),
                "batch_size": BATCH_SIZE,
                "device": "cuda",
                "torch_version": str(torch.__version__),
            }
            if path.exists():
                row = ref.read(path)
                if any(row[k] != v for k, v in expected.items()) or ref.digest(row["metrics"]) != row["metrics_sha256"]:
                    raise ValueError(f"Invalid saved result: {path}")
                print(f"[SKIP {ix}/3] seed={seed}", flush=True)
            else:
                print(f"[RUN {ix}/3] GAT adjusted seed={seed} epoch={job['selected_epoch']}", flush=True)
                metrics = ref.evaluate(job, dataset, "cuda", BATCH_SIZE)
                if any(metrics[m]["num_images"] != len(dataset) for m in ("gc", "ng")):
                    raise ValueError("Incomplete evaluation")
                row = {**expected, "metrics": metrics, "metrics_sha256": ref.digest(metrics)}
                ref.atomic_write(path, row)
            rows.append(row)
            summary = compare(controls[:ix], rows)
            ref.atomic_write(OUT / "summary.json", summary)
            score = row["metrics"]["gc"]["mR@50"]
            print(f"[DONE {ix}/3] seed={seed} GC mR@50={100 * score:.2f}%", flush=True)
    print(f"[SUMMARY] {OUT / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--val-smoke", action="store_true", help="Score two validation images on CPU")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    plan = ref.read(PLAN)
    controls = load_inputs(plan)
    if args.val_smoke:
        cache = Path(ref.CACHE_CONFIG["reference_cache"])
        with closing(ref.GQAFullDataset(cache, ref.read(cache / "manifest.json"), "val")) as dataset:
            for job in plan["jobs"]:
                metrics = ref.evaluate(job, dataset[:2], "cpu", 2)
                ref.digest(metrics)
                if any(metrics[m]["num_images"] != 2 for m in ("gc", "ng")):
                    raise ValueError("Incomplete validation smoke")
                print(f"[CPU PASS] validation GAT adjusted seed={job['seed']}", flush=True)
        return
    with ref.exclusive(OUT):
        run_test(plan, controls)


if __name__ == "__main__":
    main()
