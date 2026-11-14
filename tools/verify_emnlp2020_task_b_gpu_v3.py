"""Read-only verification of frozen source, data and all completed GPU artifacts."""

import argparse
from pathlib import Path

from tools.emnlp2020_graph_modification_v1.dataset import ROOT, read, sha256, write
from tools.emnlp2020_task_b_gpu_v3.runner import identity


def verify(config_path, receipt_path):
    config = read(config_path)
    directory = Path(config["output"])
    experiment = read(directory / "experiment.json")
    if identity(config) != experiment["source"]:
        raise ValueError("trained_code_or_config_changed")
    freeze = read(directory / "frozen-checkpoints.json")
    if freeze["experiment_identity"] != experiment["identity"]:
        raise ValueError("experiment_identity_changed")
    expected = {(k, s) for k in config["models"] for s in config["seeds"]}
    if {(r["model"], r["seed"]) for r in freeze["runs"]} != expected:
        raise ValueError("incomplete_frozen_runs")
    files = [directory / name for name in ("experiment.json", "frozen-checkpoints.json", "test-summary.json")]
    for run in freeze["runs"]:
        path = directory / f'{run["model"]}-seed{run["seed"]}'
        if sha256(path / "best.pt") != run["checkpoint_sha256"] or sha256(path / "result.json") != run["result_sha256"]:
            raise ValueError("frozen_run_changed")
        result = read(path / "test-result.json")
        if result["freeze_sha256"] != sha256(directory / "frozen-checkpoints.json"):
            raise ValueError("test_freeze_changed")
        dataset = Path(config["dataset"])
        if result["test_receipt_sha256"] != sha256(dataset / "test-receipt.json"):
            raise ValueError("test_data_receipt_changed")
        files.extend(
            path / name
            for name in (
                "best.pt",
                "result.json",
                "val-predictions.json",
                "val-cases.json",
                "test-result.json",
                "test-predictions.json",
                "test-cases.json",
            )
        )
    if sha256(Path(config["dataset"]) / "trainval-receipt.json") != experiment["trainval_receipt_sha256"]:
        raise ValueError("training_data_receipt_changed")
    for name in ("trainval", "test"):
        receipt = Path(config["dataset"]) / f"{name}-receipt.json"
        files.append(receipt)
        for file, checksum in read(receipt)["files"].items():
            path = receipt.parent / file
            if sha256(path) != checksum:
                raise ValueError("data_artifact_changed")
            files.append(path)
    result = {
        "files": {str(p.relative_to(ROOT)) if p.is_absolute() else str(p): sha256(p) for p in sorted(set(files))},
        "runs": len(freeze["runs"]),
        "device_workloads": 0,
        "verifier_code_sha256": sha256(Path(__file__)),
    }
    if Path(receipt_path).exists() and read(receipt_path) != result:
        raise ValueError("completed_artifact_receipt_changed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/emnlp2020-task-b-gpu-v3.json"))
    parser.add_argument(
        "--receipt", type=Path, default=Path("reports/level2/emnlp2020-task-b-gpu-v3/artifact-receipt.json")
    )
    parser.add_argument("--record", action="store_true", help="Record once; changed receipt is refused.")
    args = parser.parse_args()
    result = verify(args.config, args.receipt)
    if args.record:
        write(args.receipt, result)
    print(f"Verified {result['runs']} frozen runs and {len(result['files'])} artifact hashes; no GPU work.")
